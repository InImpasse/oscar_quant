"""Packed OScaR KV-cache storage for experimental runtime validation."""

from __future__ import annotations

from typing import Any

import torch

try:
    from transformers.cache_utils import Cache as _TransformersCache
except Exception:  # pragma: no cover - transformers may be absent in lightweight checks.
    _TransformersCache = object


class PackedOscarCache(_TransformersCache):
    """Hybrid cache with packed main KV storage and fp residual/fallback tensors."""

    def __init__(self, *, residual_evict_size: int = 128) -> None:
        self.residual_evict_size = residual_evict_size
        self.key_cache: list[torch.Tensor | None] = []
        self.value_cache: list[torch.Tensor | None] = []
        self.key_cache_pack: list[torch.Tensor | None] = []
        self.key_cache_params: list[torch.Tensor | None] = []
        self.value_cache_pack: list[torch.Tensor | None] = []
        self.value_cache_params: list[torch.Tensor | None] = []
        self.key_cache_norm_pack: list[torch.Tensor | None] = []
        self.key_cache_residual: list[torch.Tensor | None] = []
        self.value_cache_residual: list[torch.Tensor | None] = []
        self.key_cache_norm_residual: list[torch.Tensor | None] = []
        self.shared_source_layers: dict[str, int] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append fp K/V tensors for layers that do not use packed storage."""
        self._ensure_layer(layer_idx)
        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]
        if cached_key is None:
            new_key = key_states
            new_value = value_states
        else:
            new_key = torch.cat([cached_key, key_states], dim=2)
            new_value = torch.cat([cached_value, value_states], dim=2)
        self.key_cache[layer_idx] = new_key
        self.value_cache[layer_idx] = new_value
        return new_key, new_value

    def update_pack(
        self,
        key_pack: torch.Tensor | None,
        key_params: torch.Tensor | None,
        value_pack: torch.Tensor | None,
        value_params: torch.Tensor | None,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read or write one layer's packed main cache tensors."""
        self._ensure_layer(layer_idx)
        if key_pack is not None:
            self.key_cache_pack[layer_idx] = key_pack
            self.key_cache_params[layer_idx] = key_params
            self.value_cache_pack[layer_idx] = value_pack
            self.value_cache_params[layer_idx] = value_params
            key_norm = (cache_kwargs or {}).get("key_norm_states")
            self.key_cache_norm_pack[layer_idx] = key_norm

        tensors = (
            self.key_cache_pack[layer_idx],
            self.key_cache_params[layer_idx],
            self.value_cache_pack[layer_idx],
            self.value_cache_params[layer_idx],
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"Packed cache for layer {layer_idx} has not been initialized.")
        return tensors  # type: ignore[return-value]

    def update_residual(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append fp residual K/V states for the recent decode window."""
        self._ensure_layer(layer_idx)
        cached_key = self.key_cache_residual[layer_idx]
        cached_value = self.value_cache_residual[layer_idx]
        if cached_key is None:
            new_key = key_states
            new_value = value_states
        else:
            new_key = torch.cat([cached_key, key_states], dim=1)
            new_value = torch.cat([cached_value, value_states], dim=1)
        self.key_cache_residual[layer_idx] = new_key
        self.value_cache_residual[layer_idx] = new_value

        key_norm = (cache_kwargs or {}).get("key_norm_states")
        if isinstance(key_norm, torch.Tensor):
            cached_norm = self.key_cache_norm_residual[layer_idx]
            self.key_cache_norm_residual[layer_idx] = (
                key_norm if cached_norm is None else torch.cat([cached_norm, key_norm], dim=-1)
            )
        return new_key, new_value

    def clear_residual(self, layer_idx: int) -> None:
        self._ensure_layer(layer_idx)
        self.key_cache_residual[layer_idx] = None
        self.value_cache_residual[layer_idx] = None
        self.key_cache_norm_residual[layer_idx] = None

    def get_pack_norm(self, layer_idx: int) -> torch.Tensor | None:
        self._ensure_layer(layer_idx)
        return self.key_cache_norm_pack[layer_idx]

    def get_residual_norm(self, layer_idx: int) -> torch.Tensor | None:
        self._ensure_layer(layer_idx)
        return self.key_cache_norm_residual[layer_idx]

    def register_shared_source(self, layer_type: str, layer_idx: int) -> None:
        self.shared_source_layers[layer_type] = layer_idx

    def shared_source(self, layer_type: str) -> int | None:
        return self.shared_source_layers.get(layer_type)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._ensure_layer(layer_idx)
        fp_cache = self.key_cache[layer_idx]
        if isinstance(fp_cache, torch.Tensor):
            return int(fp_cache.shape[2])
        pack = self.value_cache_pack[layer_idx]
        residual = self.key_cache_residual[layer_idx]
        pack_len = int(pack.shape[1]) if isinstance(pack, torch.Tensor) else 0
        residual_len = int(residual.shape[1]) if isinstance(residual, torch.Tensor) else 0
        return pack_len + residual_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> None:
        return None

    def get_max_length(self) -> None:
        return None

    def crop(self, max_length: int) -> None:
        return None

    def to_legacy_cache(self) -> tuple:
        return ()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for cache_list in (
            self.key_cache,
            self.value_cache,
            self.key_cache_pack,
            self.key_cache_params,
            self.value_cache_pack,
            self.value_cache_params,
            self.key_cache_norm_pack,
            self.key_cache_residual,
            self.value_cache_residual,
            self.key_cache_norm_residual,
        ):
            for idx, tensor in enumerate(cache_list):
                if isinstance(tensor, torch.Tensor):
                    cache_list[idx] = tensor.index_select(0, beam_idx.to(tensor.device))

    def iter_tensors(self):
        for cache_list in (
            self.key_cache,
            self.value_cache,
            self.key_cache_pack,
            self.key_cache_params,
            self.value_cache_pack,
            self.value_cache_params,
            self.key_cache_norm_pack,
            self.key_cache_residual,
            self.value_cache_residual,
            self.key_cache_norm_residual,
        ):
            for tensor in cache_list:
                if isinstance(tensor, torch.Tensor):
                    yield tensor

    def storage_summary(self) -> dict[str, Any]:
        physical_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in self.iter_tensors())
        estimated_fp_bytes = 0
        for k_pack, v_pack, k_residual, v_residual, k_fp, v_fp in zip(
            self.key_cache_pack,
            self.value_cache_pack,
            self.key_cache_residual,
            self.value_cache_residual,
            self.key_cache,
            self.value_cache,
        ):
            if isinstance(k_pack, torch.Tensor) and isinstance(v_pack, torch.Tensor):
                batch_size, pack_tokens, nheads_k, _ = v_pack.shape
                head_dim = k_pack.shape[-1]
                estimated_fp_bytes += batch_size * pack_tokens * nheads_k * head_dim * 2 * 2
            if isinstance(k_residual, torch.Tensor) and isinstance(v_residual, torch.Tensor):
                estimated_fp_bytes += int(k_residual.numel() * k_residual.element_size())
                estimated_fp_bytes += int(v_residual.numel() * v_residual.element_size())
            if isinstance(k_fp, torch.Tensor) and isinstance(v_fp, torch.Tensor):
                estimated_fp_bytes += int(k_fp.numel() * k_fp.element_size())
                estimated_fp_bytes += int(v_fp.numel() * v_fp.element_size())
        packed_layers = sum(1 for tensor in self.key_cache_pack if isinstance(tensor, torch.Tensor))
        residual_layers = sum(1 for tensor in self.key_cache_residual if isinstance(tensor, torch.Tensor))
        fp_layers = sum(1 for tensor in self.key_cache if isinstance(tensor, torch.Tensor))
        compression_ratio = round(estimated_fp_bytes / physical_bytes, 6) if physical_bytes else None
        return {
            "cache_class": type(self).__name__,
            "packed_layers": packed_layers,
            "residual_layers": residual_layers,
            "fp_fallback_layers": fp_layers,
            "estimated_full_precision_bytes": estimated_fp_bytes,
            "estimated_full_precision_gib": round(estimated_fp_bytes / 1024**3, 6),
            "physical_bytes": physical_bytes,
            "physical_gib": round(physical_bytes / 1024**3, 6),
            "physical_compression_ratio": compression_ratio,
        }

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.key_cache_pack.append(None)
            self.key_cache_params.append(None)
            self.value_cache_pack.append(None)
            self.value_cache_params.append(None)
            self.key_cache_norm_pack.append(None)
            self.key_cache_residual.append(None)
            self.value_cache_residual.append(None)
            self.key_cache_norm_residual.append(None)


def new_packed_oscar_cache(*, residual_evict_size: int = 128) -> PackedOscarCache:
    return PackedOscarCache(residual_evict_size=residual_evict_size)
