"""Configuration models for OScaR KV-cache quantization.

The adapter hands these values to upstream OScaR's `init_quarot` helper when an
attention layer first sees a prefill prompt. Keeping the options in a Pydantic
model gives the CLI, Python API, and future config-file loaders one validated
schema instead of several lightly checked dictionaries.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KVCacheMode = Literal["fake", "packed"]


class OscarKVConfig(BaseModel):
    """Validated OScaR KV-cache quantization settings.

    What it does:
        Captures the bit widths, group sizes, clipping ratios, and rotation
        toggles needed to initialize OScaR's quantizer on each Granite attention
        layer.

    Why it exists:
        OScaR expects a cluster of tightly related options. A frozen Pydantic
        model keeps those options immutable after construction and catches bad
        values, such as one-bit quantization or zero group sizes, before a model
        has been loaded onto GPU memory.

    How it helps:
        Callers can pass `OscarKVConfig` directly to `apply_oscar_to_granite`,
        while CLI code can build the same model from parsed arguments. This
        keeps the public API and command-line behavior aligned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_bits: int = Field(default=2, ge=2)
    v_bits: int = Field(default=2, ge=2)
    k_groupsize: int = Field(default=32, gt=0)
    v_groupsize: int = Field(default=32, gt=0)
    k_sym: bool = False
    v_sym: bool = False
    k_clip_ratio: float = Field(default=1.0, gt=0)
    v_clip_ratio: float = Field(default=1.0, gt=0)
    residual_length: int = Field(default=0, ge=0)
    k_token_rotation: bool = False
    k_norm_factoring: bool = True
    use_hadamard: bool = True
    offline_v_hadamard: bool = True
    kv_cache_mode: KVCacheMode = "fake"

    def as_namespace(self) -> SimpleNamespace:
        """Return a namespace shaped like upstream OScaR helper arguments.

        What it does:
            Converts the Pydantic model into a `types.SimpleNamespace` with one
            attribute per quantization option.

        Why it exists:
            The upstream OScaR codebase uses object attributes for many helper
            settings. Returning a namespace preserves that calling convention
            while still letting this project own validation through Pydantic.

        How it helps:
            The patching layer can pass settings into OScaR without leaking
            Pydantic-specific APIs into the integration boundary.
        """
        return SimpleNamespace(**self.model_dump())


def validate_runtime_kv_cache_mode(mode: KVCacheMode, profile: str) -> None:
    """Fail early when a requested KV-cache storage mode is not implemented."""
    if mode == "fake":
        return
    if profile == "gemma4-e2b":
        return
    raise NotImplementedError(
        f"Packed KV cache mode is not implemented for profile {profile!r} in this repo. "
        "Use --kv-cache-mode fake, or add a model-family-specific packed cache implementation."
    )
