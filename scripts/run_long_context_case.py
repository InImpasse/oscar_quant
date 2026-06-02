#!/usr/bin/env python3
"""Run one long-context baseline or OScaR KV-cache validation case."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from oscar_quant.config import OscarKVConfig, validate_runtime_kv_cache_mode
from oscar_quant.gemma4_patch import apply_oscar_to_gemma4
from oscar_quant.granite_patch import apply_oscar_to_granite
from oscar_quant.kv_cache_utils import OSCAR_CACHE_SUMMARY_ATTR
from oscar_quant.models import DEFAULT_GEMMA4_E2B_MODEL_ID, DEFAULT_GRANITE_MODEL_ID

PROFILES = {
    "granite-4.0-1b-base": {
        "model_id": DEFAULT_GRANITE_MODEL_ID,
        "auto_model": "causal-lm",
        "patch": apply_oscar_to_granite,
    },
    "gemma4-e2b": {
        "model_id": DEFAULT_GEMMA4_E2B_MODEL_ID,
        "auto_model": "image-text-to-text",
        "patch": apply_oscar_to_gemma4,
    },
}

PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

OSCAR_BITS = {
    "int8": 8,
    "int4": 4,
    "int2": 2,
}


class NvidiaSmiMonitor:
    """Poll nvidia-smi memory.used while a generation run is active."""

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if values:
                    current = max(values)
                    self.peak_mib = current if self.peak_mib is None else max(self.peak_mib, current)
            except Exception:
                return
            time.sleep(self.interval_sec)


def main() -> int:
    args = parse_args()
    started = time.time()
    result: dict[str, Any] = {
        "profile": args.profile,
        "run_type": args.run_type,
        "precision": args.precision,
        "kv_cache_mode": args.kv_cache_mode,
        "context_target": args.context_target,
        "max_new_tokens": args.max_new_tokens,
        "status": "error",
    }

    try:
        validate_args(args)
        run_result = run_case(args)
        result.update(run_result)
        result["status"] = "success"
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "oom", "error": str(exc)})
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        result.update({"status": status, "error": str(exc)})
        if status == "oom" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        result["elapsed_wall_sec"] = round(time.time() - started, 6)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one long-context validation case.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--run-type", choices=("baseline", "oscar"), required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "int8", "int4", "int2"), required=True)
    parser.add_argument("--kv-cache-mode", choices=("fake", "packed"), default="fake")
    parser.add_argument("--context-target", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--k-groupsize", type=int, default=32)
    parser.add_argument("--v-groupsize", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.run_type == "baseline" and args.precision not in PRECISION_TO_DTYPE:
        raise ValueError("Baseline runs support only bf16/fp16 in this script; int* precisions are OScaR KV bits.")
    if args.run_type == "oscar" and args.precision not in OSCAR_BITS:
        raise ValueError("OScaR runs in this script use int8/int4/int2 KV-cache precision.")
    validate_runtime_kv_cache_mode(args.kv_cache_mode, args.profile)


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILES[args.profile]
    model_id = args.model_id or profile["model_id"]
    load_dtype = PRECISION_TO_DTYPE.get(args.precision, torch.bfloat16)
    assets = load_assets(args.profile, model_id, args.trust_remote_code)
    tokenizer = tokenizer_from_assets(assets)
    model = load_model(profile["auto_model"], model_id, load_dtype, args)
    model.eval()

    patched_layers = 0
    if args.run_type == "oscar":
        bits = OSCAR_BITS[args.precision]
        kv_config = OscarKVConfig(
            k_bits=bits,
            v_bits=bits,
            k_groupsize=args.k_groupsize,
            v_groupsize=args.v_groupsize,
            residual_length=args.residual_length,
            kv_cache_mode=args.kv_cache_mode,
        )
        patched_layers = profile["patch"](model, kv_config)

    prompt = build_prompt(tokenizer, args.context_target)
    inputs = make_inputs(args.profile, assets, prompt)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs_to_model(inputs, model)
    generation_kwargs = generation_kwargs_for(tokenizer, args.max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    monitor = NvidiaSmiMonitor()
    monitor.start()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_elapsed = time.perf_counter() - start
    monitor.stop()

    generated_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    kv_summary = oscar_cache_summary(model)
    theoretical = theoretical_kv_gib(model, prompt_tokens + generated_tokens, args)

    return {
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "patched_attention_layers": patched_layers,
        "decode_time_sec": round(decode_elapsed, 6),
        "decode_tokens_per_sec": round(generated_tokens / decode_elapsed, 6) if decode_elapsed > 0 else None,
        "torch_peak_allocated_gib": cuda_gib(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "torch_peak_reserved_gib": cuda_gib(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "nvidia_smi_peak_used_gib": round((monitor.peak_mib or 0) / 1024, 6) if monitor.peak_mib is not None else None,
        "kv_observed_tensor_gib": kv_summary["gib"],
        "kv_cache_storage_note": kv_summary["note"],
        **theoretical,
    }


def load_assets(profile: str, model_id: str, trust_remote_code: bool) -> Any:
    if profile == "gemma4-e2b":
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def tokenizer_from_assets(assets: Any) -> Any:
    return getattr(assets, "tokenizer", assets)


def load_model(auto_model: str, model_id: str, dtype: torch.dtype, args: argparse.Namespace) -> Any:
    if auto_model == "image-text-to-text":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )


def build_prompt(tokenizer: Any, target_tokens: int) -> str:
    seed = (
        "Oscar KV cache benchmark context filler. Keep every detail stable across "
        "baseline and patched runs. "
    )
    text = seed
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(token_ids) < target_tokens:
        text += seed
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)


def make_inputs(profile: str, assets: Any, prompt: str) -> dict[str, Any]:
    if profile == "gemma4-e2b":
        try:
            return dict(assets(text=prompt, return_tensors="pt"))
        except TypeError:
            return dict(assets(prompt, return_tensors="pt"))
    return dict(assets(prompt, return_tensors="pt"))


def move_inputs_to_model(inputs: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    device = next(model.parameters()).device
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generation_kwargs_for(tokenizer: Any, max_new_tokens: int) -> dict[str, Any]:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def oscar_cache_summary(model: torch.nn.Module) -> dict[str, Any]:
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
            "fake quantize/dequantize kept fp cache tensors"
        ).format(layers=len(summaries))
    )
    return {"gib": round(observed_bytes / 1024**3, 6), "note": note}


def theoretical_kv_gib(model: torch.nn.Module, tokens: int, args: argparse.Namespace) -> dict[str, float | None]:
    config = getattr(model.config, "text_config", model.config)
    layers = getattr(config, "num_hidden_layers", None)
    kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        attention_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is not None and attention_heads:
            head_dim = hidden_size // attention_heads
    if not all(value is not None for value in (layers, kv_heads, head_dim)):
        return {"kv_theoretical_bf16_gib": None, "kv_theoretical_quantized_gib": None}

    bf16_bytes = int(layers) * 2 * int(kv_heads) * tokens * int(head_dim) * 2
    if args.run_type == "oscar":
        quant_bytes = bf16_bytes * OSCAR_BITS[args.precision] / 16
    else:
        quant_bytes = bf16_bytes
    return {
        "kv_theoretical_bf16_gib": round(bf16_bytes / 1024**3, 6),
        "kv_theoretical_quantized_gib": round(quant_bytes / 1024**3, 6),
    }


def cuda_gib(bytes_value: int) -> float:
    return round(bytes_value / 1024**3, 6)


if __name__ == "__main__":
    raise SystemExit(main())
