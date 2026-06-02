"""Pydantic schemas for benchmark output.

The baseline command prints JSON because it is intended to be consumed by both
humans and automation. These models document that JSON contract in Python, keep
numeric fields bounded, and make it easier to compare runs over time.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkRun(BaseModel):
    """One timed generation run in the benchmark report.

    What it does:
        Stores the label, timing, generated token count, throughput, decoded
        text, optional CUDA peak memory, and optional patched-layer count for a
        single model invocation.

    Why it exists:
        The benchmark runs both unpatched and OScaR-patched generation. A typed
        result object prevents the two paths from drifting into slightly
        different JSON shapes.

    How it helps:
        Consumers can compare `baseline` and `oscar_kv_quant` entries without
        guessing which fields are present or what units are used.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    elapsed_seconds: float = Field(ge=0)
    new_tokens: int = Field(ge=0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0)
    text: str
    cuda_peak_allocated_gib: Optional[float] = Field(default=None, ge=0)
    patched_attention_layers: Optional[int] = Field(default=None, ge=0)
    kv_cache_observed_gib: Optional[float] = Field(default=None, ge=0)
    kv_cache_storage_note: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Top-level JSON payload emitted by `oscar-baseline`.

    What it does:
        Wraps model identity, prompt length, KV quantization bit widths, and a
        list of per-run measurements.

    Why it exists:
        Benchmark output should be stable enough to save, diff, or feed into a
        dashboard. A Pydantic schema gives that output a versionable structure
        without adding a heavier reporting system.

    How it helps:
        If a future change accidentally emits negative token counts, missing
        runs, or unsupported bit widths, validation fails near the benchmark
        code instead of after results have been logged somewhere else.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    prompt_tokens: int = Field(ge=0)
    k_bits: int = Field(ge=2)
    v_bits: int = Field(ge=2)
    kv_cache_mode: str = "fake"
    runs: List[BenchmarkRun]
