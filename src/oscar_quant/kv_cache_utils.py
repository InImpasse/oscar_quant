"""Shared helpers for runtime OScaR KV-cache attention patches.

The Granite and Gemma4 adapters both need the same low-level behaviors:
initialize the upstream OScaR quantizer lazily, update Hugging Face cache
objects across minor API differences, read/write one layer's cached key/value
tensors, and run OScaR's prefill/decode quantization step after attention has
used full-precision tensors for the current call.

Keeping those pieces here makes the model-family patch modules smaller and
keeps the fragile cache compatibility logic in one place.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import OscarKVConfig

OSCAR_CONFIG_ATTR = "_oscar_quant_config"
OSCAR_CACHE_SUMMARY_ATTR = "_oscar_quant_cache_summary"


def ensure_oscar_quantizer(module: torch.nn.Module) -> None:
    """Create or reset the OScaR quantizer attached to an attention module.

    What it does:
        Reads the module's attached `OscarKVConfig`, imports upstream OScaR's
        `init_quarot` helper on demand, and attaches `module.quarot_quantizer`
        the first time the module sees a prefill. If the quantizer already
        exists, its committed-token counters are reset for a fresh generation
        cache.

    Why it exists:
        OScaR is an optional CUDA-backed dependency. Importing and initializing
        it during package import would make simple configuration usage fail on
        machines that only want the `.safetensors` exporter or docs. The
        quantizer also belongs to an attention layer because it tracks committed
        cache length for that layer.

    How it helps:
        Granite and Gemma4 can share the same initialization behavior while
        keeping OScaR failures close to the runtime path that actually needs
        them.
    """
    config: OscarKVConfig = getattr(module, OSCAR_CONFIG_ATTR)
    args = config.as_namespace()

    if not hasattr(module, "quarot_quantizer"):
        try:
            from kv_cache_compression.quarot_utils import init_quarot  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OScaR-KV-Quant is not installed in this environment. Run "
                "`bash scripts/install_oscar_dependency.sh` from this repo, or "
                "install https://github.com/ZunhaiSu/OScaR-KV-Quant manually."
            ) from exc

        init_quarot(
            module,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            k_groupsize=args.k_groupsize,
            v_groupsize=args.v_groupsize,
            k_sym=args.k_sym,
            v_sym=args.v_sym,
            k_clip_ratio=args.k_clip_ratio,
            v_clip_ratio=args.v_clip_ratio,
            residual_length=args.residual_length,
            k_token_rotation=args.k_token_rotation,
            k_norm_factoring=args.k_norm_factoring,
            use_hadamard=args.use_hadamard,
            offline_v_hadamard=args.offline_v_hadamard,
        )
        return

    quantizer = module.quarot_quantizer
    for attr in ("committed_k_len", "committed_v_len"):
        if hasattr(quantizer, attr):
            setattr(quantizer, attr, 0)


def rotate_query_like_oscar(module: torch.nn.Module, query_states: torch.Tensor) -> torch.Tensor:
    """Apply OScaR's Q-side Hadamard rotation without touching K/V tensors.

    What it does:
        Mirrors the query rotation performed by upstream OScaR's
        `QuaRotKVCacheQuantizer.process_kv`, but only for the query tensor.

    Why it exists:
        Gemma4 has shared-KV attention layers. Those layers reuse already
        rotated key tensors from an earlier source layer, so running
        `process_kv(query, key, value)` would rotate K a second time. They still
        need their query tensor in the same rotated space as the shared keys.

    How it helps:
        Shared-KV Gemma4 layers can participate in OScaR attention math without
        owning or modifying a KV cache entry themselves.
    """
    config: OscarKVConfig = getattr(module, OSCAR_CONFIG_ATTR)
    if config.k_bits >= 16 or not config.use_hadamard:
        return query_states

    try:
        from kv_cache_compression.quarot_utils import hadamard_rotation  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OScaR-KV-Quant is not installed in this environment. Run "
            "`bash scripts/install_oscar_dependency.sh` from this repo, or "
            "install https://github.com/ZunhaiSu/OScaR-KV-Quant manually."
        ) from exc

    dtype = query_states.dtype
    return hadamard_rotation(query_states).to(dtype)


def update_cache(
    cache: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a Transformers cache across cache API variants.

    What it does:
        Calls `cache.update` with cache kwargs when provided, then retries with
        the older no-kwargs call shape if the installed cache implementation
        does not accept the newer signature.

    Why it exists:
        Transformers cache internals have changed while Granite 4 and Gemma4
        support have been moving. Some families pass `cache_position` or RoPE
        values into `update`; others do not.

    How it helps:
        Model-family patch modules can follow their upstream forward logic and
        let this helper absorb minor cache signature drift.
    """
    if cache_kwargs is None:
        cache_kwargs = {}

    if cache_kwargs:
        try:
            return cache.update(key_states, value_states, layer_idx, cache_kwargs)
        except TypeError as first_error:
            try:
                return cache.update(key_states, value_states, layer_idx)
            except TypeError:
                raise first_error

    return cache.update(key_states, value_states, layer_idx)


def quantize_layer_cache_after_attention(
    module: torch.nn.Module,
    cache: Any,
    layer_idx: int,
    q_len: int,
) -> dict[str, Any] | None:
    """Quantize one layer's stored KV cache after attention has consumed it.

    What it does:
        Reads the cache tensors for `layer_idx`, calls OScaR's prefill
        quantizer when `q_len > 1` or decode quantizer when `q_len == 1`, and
        writes the returned tensors back into the cache object.

    Why it exists:
        OScaR's intended flow keeps the current attention computation in
        high-precision tensor space, then stores a compressed approximation for
        future decode steps.

    How it helps:
        Granite and Gemma4 source attention layers can use identical
        prefill/decode quantization behavior after their model-specific forward
        math has run.
    """
    if not hasattr(module, "quarot_quantizer"):
        return None

    cache_key_states, cache_value_states = get_layer_cache(cache, layer_idx)
    before = describe_cache_tensors(cache_key_states, cache_value_states)
    if q_len > 1:
        cache_key_states, cache_value_states = module.quarot_quantizer.quantize_prefill(
            cache_key_states,
            cache_value_states,
        )
    else:
        cache_key_states, cache_value_states = module.quarot_quantizer.quantize_kv_cache(
            cache_key_states,
            cache_value_states,
        )
    after = describe_cache_tensors(cache_key_states, cache_value_states)
    set_layer_cache(cache, layer_idx, cache_key_states, cache_value_states)

    summary = {
        "layer_idx": layer_idx,
        "q_len": q_len,
        "mode": "prefill" if q_len > 1 else "decode",
        "before": before,
        "after": after,
        "physical_bytes_changed": before["bytes"] != after["bytes"],
        "dtype_or_shape_changed": (
            before["key_dtype"] != after["key_dtype"]
            or before["value_dtype"] != after["value_dtype"]
            or before["key_shape"] != after["key_shape"]
            or before["value_shape"] != after["value_shape"]
        ),
    }
    summary["storage_note"] = (
        "physical cache tensor storage changed"
        if summary["physical_bytes_changed"] or summary["dtype_or_shape_changed"]
        else "physical cache tensor storage unchanged; upstream path is fake quantize/dequantize"
    )
    setattr(module, OSCAR_CACHE_SUMMARY_ATTR, summary)
    return summary


def describe_cache_tensors(key_states: torch.Tensor, value_states: torch.Tensor) -> dict[str, Any]:
    """Return physical storage metadata for one layer's cached K/V tensors."""
    return {
        "bytes": tensor_nbytes(key_states) + tensor_nbytes(value_states),
        "key_dtype": str(key_states.dtype),
        "value_dtype": str(value_states.dtype),
        "key_shape": tuple(int(dim) for dim in key_states.shape),
        "value_shape": tuple(int(dim) for dim in value_states.shape),
        "key_bytes": tensor_nbytes(key_states),
        "value_bytes": tensor_nbytes(value_states),
    }


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the physical bytes occupied by a tensor's visible elements."""
    return int(tensor.numel() * tensor.element_size())


def get_layer_cache(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read key and value cache tensors for one decoder layer.

    What it does:
        Delegates to `read_cache_tensor` for both the key and value entries at
        the requested layer index.

    Why it exists:
        After attention has run, OScaR must quantize exactly the tensors that
        Transformers will reuse on subsequent decode steps.

    How it helps:
        Keeps each patch module focused on model-family attention math instead
        of cache layout details.
    """
    key_cache = read_cache_tensor(cache, layer_idx, "key")
    value_cache = read_cache_tensor(cache, layer_idx, "value")
    return key_cache, value_cache


def set_layer_cache(
    cache: Any,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    """Write quantized key and value tensors back into one cache layer.

    What it does:
        Supports legacy cache lists, newer layered caches, and layer objects
        that expose either `keys`/`values` or `key_cache`/`value_cache`.

    Why it exists:
        The adapter reads full-precision cache tensors, lets OScaR quantize
        them, and must then replace the stored cache tensors in whatever cache
        representation the installed Transformers version uses.

    How it helps:
        Quantized K/V tensors become the source for future decode tokens without
        requiring a custom generation loop.
    """
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
        return

    layer = cache_layer(cache, layer_idx)
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        layer.keys = key_states
        layer.values = value_states
        return
    if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
        layer.key_cache = key_states
        layer.value_cache = value_states
        return

    raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")


def read_cache_tensor(cache: Any, layer_idx: int, kind: str) -> torch.Tensor:
    """Read one cache tensor while tolerating cache layout differences.

    What it does:
        Looks for legacy top-level `key_cache` or `value_cache` lists first,
        then checks the requested layer object for newer attribute names.

    Why it exists:
        Transformers cache internals differ across versions and model families,
        but OScaR only needs the actual tensor for a specific layer and kind.

    How it helps:
        The adapter avoids pinning itself to one narrow cache representation,
        which is useful while Granite 4 and Gemma4 support are still moving.
    """
    legacy_attr = f"{kind}_cache"
    if hasattr(cache, legacy_attr):
        return getattr(cache, legacy_attr)[layer_idx]

    layer = cache_layer(cache, layer_idx)
    for attr in (f"{kind}s", legacy_attr):
        if hasattr(layer, attr):
            tensor = getattr(layer, attr)
            if tensor is not None:
                return tensor

    raise TypeError(f"Could not read {kind} cache from {type(cache)!r}")


def cache_layer(cache: Any, layer_idx: int) -> Any:
    """Return a layer object from a modern Transformers cache.

    What it does:
        Accesses `cache.layers[layer_idx]` when the cache exposes a layered
        representation.

    Why it exists:
        Several helper functions need the same layer lookup before reading or
        writing key/value tensors.

    How it helps:
        Centralizing this lookup gives unsupported cache types one clear error
        message instead of scattered `AttributeError`s.
    """
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx]
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")
