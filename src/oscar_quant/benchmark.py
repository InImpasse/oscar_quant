"""Benchmark Granite generation with and without OScaR KV-cache quantization.

The benchmark CLI loads the baseline Granite 4.0 1B model once, measures normal
generation, then patches attention in place and measures the OScaR path. It is
designed to answer the first practical integration question: what changes in
latency, throughput, and peak CUDA memory when KV-cache quantization is enabled?
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cli import _dtype, _oscar_config
from .config import validate_runtime_kv_cache_mode
from .granite_patch import apply_oscar_to_granite
from .kv_cache_utils import OSCAR_CACHE_SUMMARY_ATTR
from .models import DEFAULT_GRANITE_MODEL_ID
from .schemas import BenchmarkReport, BenchmarkRun


def main(argv: list[str] | None = None) -> int:
    """Run baseline and optional OScaR-patched generation measurements.

    What it does:
        Loads model/tokenizer, prepares generation inputs, times a vanilla
        generation run, patches the attention modules unless `--baseline-only`
        is set, and emits a Pydantic-validated JSON report.

    Why it exists:
        OScaR is primarily useful when it improves long-context cache memory
        behavior. A repeatable command makes it easier to capture before/after
        measurements while changing prompts, token counts, and quantization
        settings.

    How it helps:
        The JSON output can be saved in CI artifacts, notebooks, or experiment
        logs and compared across hardware, prompts, and model revisions.
    """
    args = _parse_args(argv)
    validate_runtime_kv_cache_mode(args.kv_cache_mode, "granite-4.0-1b-base")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    generation_kwargs = _generation_kwargs(args, tokenizer)

    runs: list[BenchmarkRun] = []

    _warmup(model, inputs, generation_kwargs, args.warmup_runs)
    runs.append(_timed_generate("baseline", model, tokenizer, inputs, generation_kwargs))

    if not args.baseline_only:
        patched_layers = apply_oscar_to_granite(model, _oscar_config(args))
        print(f"patched_granite_attention_layers={patched_layers}", file=sys.stderr)
        _warmup(model, inputs, generation_kwargs, args.warmup_runs)
        oscar_result = _timed_generate("oscar_kv_quant", model, tokenizer, inputs, generation_kwargs)
        oscar_result = oscar_result.model_copy(update={"patched_attention_layers": patched_layers})
        runs.append(oscar_result)

    report = BenchmarkReport(
        model_id=args.model_id,
        prompt_tokens=int(inputs["input_ids"].shape[-1]),
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        kv_cache_mode=args.kv_cache_mode,
        runs=runs,
    )
    print(report.model_dump_json(indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse benchmark command-line arguments.

    What it does:
        Defines flags for model loading, generation length, sampling behavior,
        warmup count, baseline-only mode, and OScaR quantizer settings.

    Why it exists:
        Benchmarking has a few controls that are not needed by the simpler
        generation CLI, especially warmups and baseline-only measurement.

    How it helps:
        A single namespace can drive both the unpatched and patched run, keeping
        the comparison fair because both paths share the same prompt and
        generation parameters.
    """
    parser = argparse.ArgumentParser(
        description="Baseline Granite generation with and without OScaR KV-cache quantization."
    )
    parser.add_argument("--model-id", default=DEFAULT_GRANITE_MODEL_ID)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--baseline-only", action="store_true")
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
            "KV-cache storage mode. Current Granite benchmark supports only fake "
            "quantize/dequantize; packed fails fast until implemented."
        ),
    )
    return parser.parse_args(argv)


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    """Create keyword arguments for `model.generate`.

    What it does:
        Converts parsed generation flags into the kwargs used for each timed
        run, including cache use, token limits, sampling controls, and token ids.

    Why it exists:
        Both the baseline and OScaR-patched run must use exactly the same
        generation settings for the comparison to be meaningful.

    How it helps:
        Centralizing these kwargs prevents subtle benchmark drift, such as one
        path sampling while the other runs greedily.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _move_inputs_to_model(inputs: Any, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Move tokenized prompt tensors onto the model's first parameter device.

    What it does:
        Finds the model device from its parameters and transfers every tokenizer
        output tensor to that device.

    Why it exists:
        `device_map="auto"` may place a loaded model on CUDA, CPU, or another
        available device. Tokenizer outputs start on CPU by default.

    How it helps:
        Generation starts with tensors on a compatible device, avoiding a common
        runtime error before the benchmark reaches the attention patch.
    """
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _warmup(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
    runs: int,
) -> None:
    """Run untimed generation passes before measurement.

    What it does:
        Executes `model.generate` a configurable number of times without
        collecting timings.

    Why it exists:
        First runs often include one-time costs such as kernel loading, graph
        setup, allocator growth, or cache initialization.

    How it helps:
        Warmups let users measure steadier-state generation when comparing the
        vanilla and OScaR paths.
    """
    for _ in range(runs):
        with torch.inference_mode():
            model.generate(**inputs, **generation_kwargs)


def _timed_generate(
    label: str,
    model: torch.nn.Module,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
) -> BenchmarkRun:
    """Measure one generation call and return a structured benchmark row.

    What it does:
        Optionally resets CUDA memory counters, synchronizes around generation,
        records elapsed wall time, decodes the newly generated tokens, and
        returns a `BenchmarkRun`.

    Why it exists:
        The benchmark needs identical timing logic for the baseline and patched
        model. CUDA synchronization is especially important because GPU work is
        asynchronous from Python's point of view.

    How it helps:
        Each measured run reports comparable latency, throughput, text output,
        and optional memory usage in the same Pydantic schema.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    new_token_count = int(new_tokens.shape[-1])
    cache_summary = _oscar_cache_summary(model)
    return BenchmarkRun(
        label=label,
        elapsed_seconds=round(elapsed_seconds, 6),
        new_tokens=new_token_count,
        tokens_per_second=round(new_token_count / elapsed_seconds, 3) if elapsed_seconds > 0 else None,
        text=tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        cuda_peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if torch.cuda.is_available()
        else None,
        kv_cache_observed_gib=cache_summary["gib"],
        kv_cache_storage_note=cache_summary["note"],
    )


def _oscar_cache_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize physical KV-cache tensor storage recorded by OScaR patches."""
    summaries = [
        getattr(module, OSCAR_CACHE_SUMMARY_ATTR)
        for module in model.modules()
        if hasattr(module, OSCAR_CACHE_SUMMARY_ATTR)
    ]
    if not summaries:
        return {"gib": None, "note": None}

    observed_bytes = sum(int(summary["after"]["bytes"]) for summary in summaries)
    changed_layers = sum(
        1
        for summary in summaries
        if summary["physical_bytes_changed"] or summary["dtype_or_shape_changed"]
    )
    note = (
        f"{changed_layers}/{len(summaries)} layers changed physical cache tensor storage"
        if changed_layers
        else (
            "0/{layers} layers changed physical cache tensor storage; "
            "OScaR upstream fake quantize/dequantize keeps fp cache tensors"
        ).format(layers=len(summaries))
    )
    return {"gib": round(observed_bytes / 1024**3, 6), "note": note}


if __name__ == "__main__":
    raise SystemExit(main())
