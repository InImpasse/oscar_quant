"""Command-line generation entry point for OScaR KV quantization.

This module loads a supported Hugging Face model, applies its runtime OScaR
attention patch, and then runs a single text-generation request. It is the
shortest end-to-end path for checking that OScaR is installed and that the
model-family adapter can participate in normal `model.generate` flows.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

from .config import OscarKVConfig, validate_runtime_kv_cache_mode
from .loader import load_oscar_patched_gemma4, load_oscar_patched_granite
from .models import DEFAULT_GEMMA4_E2B_MODEL_ID, DEFAULT_GRANITE_MODEL_ID, ModelProfileName


def main(argv: list[str] | None = None) -> int:
    """Run one OScaR-patched generation request.

    What it does:
        Parses CLI arguments, loads the processor-or-tokenizer/model pair,
        applies the selected model family's OScaR attention patch, tokenizes the
        prompt, generates new tokens, and prints the decoded continuation.

    Why it exists:
        Users need a direct smoke test before investing in longer benchmark
        runs. This function keeps that path close to the library API so CLI
        behavior and Python usage exercise the same patching code.

    How it helps:
        A successful run proves three things at once: the model can load, the
        correct attention modules can be found, and OScaR can quantize the KV
        cache during generation.
    """
    args = _parse_args(argv)
    dtype = _dtype(args.dtype)
    validate_runtime_kv_cache_mode(args.kv_cache_mode, args.profile)
    if args.kv_cache_mode == "packed":
        raise NotImplementedError(
            "Packed KV cache mode is currently wired through "
            "`scripts/run_long_context_case.py` for Gemma4 validation, not `oscar-generate`."
        )

    model_id = _resolved_model_id(args.profile, args.model_id)
    patched_model = _load_patched_model(args.profile, model_id, dtype, args)
    print(
        f"patched_{args.profile}_attention_layers={patched_model.patched_attention_layers}",
        file=sys.stderr,
    )
    print(f"kv_cache_mode={args.kv_cache_mode}", file=sys.stderr)

    prompt = args.prompt
    if args.chat_template:
        prompt = _apply_chat_template(patched_model, prompt)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(patched_model.generate_text(prompt, **generate_kwargs))

    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"cuda_peak_allocated_gib={peak_gib:.2f}", file=sys.stderr)

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options into a namespace.

    What it does:
        Defines model-loading, generation, and OScaR quantization flags for the
        generation CLI.

    Why it exists:
        Keeping all flags in one helper makes it obvious which user-facing
        values can influence model loading versus cache quantization.

    How it helps:
        The parsed namespace can be passed to `_dtype` and `_oscar_config`,
        avoiding duplicate parsing logic in the main execution path.
    """
    parser = argparse.ArgumentParser(description="Run a supported model with OScaR KV-cache quantization.")
    parser.add_argument("--profile", choices=("granite-4.0-1b-base", "gemma4-e2b"), default="granite-4.0-1b-base")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--k-groupsize", type=int, default=32)
    parser.add_argument("--v-groupsize", type=int, default=32)
    parser.add_argument("--k-sym", action="store_true")
    parser.add_argument("--v-sym", action="store_true")
    parser.add_argument("--k-clip-ratio", type=float, default=1.0)
    parser.add_argument("--v-clip-ratio", type=float, default=1.0)
    parser.add_argument("--residual-length", type=int, default=0)
    parser.add_argument("--k-token-rotation", action="store_true")
    parser.add_argument("--disable-k-norm-factoring", action="store_true")
    parser.add_argument("--disable-hadamard", action="store_true")
    parser.add_argument("--disable-offline-v-hadamard", action="store_true")
    parser.add_argument(
        "--kv-cache-mode",
        choices=("fake", "packed"),
        default="fake",
        help=(
            "KV-cache storage mode. Current Granite/Gemma4 runtime supports only "
            "fake quantize/dequantize; packed fails fast until implemented."
        ),
    )
    return parser.parse_args(argv)


def _resolved_model_id(profile: ModelProfileName, model_id: str | None) -> str:
    """Return the model id selected by a profile plus optional override.

    What it does:
        Uses an explicit `--model-id` when provided, otherwise returns the
        built-in Hugging Face id for the requested profile.

    Why it exists:
        Granite and Gemma4 have different defaults, but the CLI should keep one
        consistent `--model-id` override flag.

    How it helps:
        Users can switch from Granite to Gemma4 with only `--profile gemma4-e2b`
        while advanced users can still point at local or fine-tuned checkpoints.
    """
    if model_id is not None:
        return model_id
    if profile == "gemma4-e2b":
        return DEFAULT_GEMMA4_E2B_MODEL_ID
    return DEFAULT_GRANITE_MODEL_ID


def _load_patched_model(
    profile: ModelProfileName,
    model_id: str,
    dtype: str | torch.dtype,
    args: argparse.Namespace,
) -> Any:
    """Load and patch the selected runtime model family.

    What it does:
        Dispatches to the Granite or Gemma4 high-level loader with the shared
        dtype, device map, trust flag, eager attention implementation, and
        validated OScaR config.

    Why it exists:
        The generation CLI is shared, but runtime attention patching is
        intentionally model-family-specific.

    How it helps:
        One command can smoke-test both supported OScaR adapters without hiding
        the fact that Granite and Gemma4 use different loader classes.
    """
    loader = load_oscar_patched_gemma4 if profile == "gemma4-e2b" else load_oscar_patched_granite
    return loader(
        model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
        kv_config=_oscar_config(args),
    )


def _apply_chat_template(patched_model: Any, prompt: str) -> str:
    """Apply a chat template using whichever tokenizer-like object is present.

    What it does:
        Finds a tokenizer on the patched wrapper, either directly for Granite or
        through the Gemma4 processor, and calls `apply_chat_template`.

    Why it exists:
        The two supported loaders expose preprocessing assets differently, but
        chat-template prompting is a model-facing concern rather than a patching
        concern.

    How it helps:
        The CLI can keep one `--chat-template` flag for both Granite and Gemma4
        without duplicating generation setup.
    """
    tokenizer = getattr(patched_model, "tokenizer", None)
    if tokenizer is None:
        processor = getattr(patched_model, "processor", None)
        tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("The selected model assets do not expose apply_chat_template.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _dtype(name: str) -> str | torch.dtype:
    """Map a CLI dtype name to the value expected by Transformers.

    What it does:
        Returns `"auto"` unchanged or converts concrete precision names to
        torch dtype objects.

    Why it exists:
        `AutoModelForCausalLM.from_pretrained` accepts both `"auto"` and dtype
        objects, while argparse only deals in strings.

    How it helps:
        The CLI can expose a small, friendly set of dtype choices without
        leaking torch internals into argument parsing.
    """
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _oscar_config(args: argparse.Namespace) -> OscarKVConfig:
    """Build a validated OScaR config from parsed CLI flags.

    What it does:
        Translates argparse field names and boolean disabling flags into an
        `OscarKVConfig` instance.

    Why it exists:
        Some CLI flags are expressed as negative toggles, such as
        `--disable-hadamard`, because that is clearer for users when defaults
        are enabled. OScaR needs the positive boolean form.

    How it helps:
        Pydantic validation happens before the model patch is applied, so bad
        quantization settings fail early with a useful error.
    """
    return OscarKVConfig(
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
        k_norm_factoring=not args.disable_k_norm_factoring,
        use_hadamard=not args.disable_hadamard,
        offline_v_hadamard=not args.disable_offline_v_hadamard,
        kv_cache_mode=args.kv_cache_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
