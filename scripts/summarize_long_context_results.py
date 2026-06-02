#!/usr/bin/env python3
"""Summarize long-context JSON results into a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "profile",
    "model_id",
    "run_type",
    "precision",
    "kv_cache_mode",
    "context_target",
    "prompt_tokens",
    "generated_tokens",
    "status",
    "decode_time_sec",
    "decode_tokens_per_sec",
    "torch_peak_allocated_gib",
    "torch_peak_reserved_gib",
    "nvidia_smi_peak_used_gib",
    "kv_theoretical_bf16_gib",
    "kv_theoretical_quantized_gib",
    "kv_observed_tensor_gib",
    "kv_cache_storage_note",
    "patched_attention_layers",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize long-context result JSON files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/long_context/raw"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/long_context/summary.csv"))
    args = parser.parse_args()

    rows = [read_row(path) for path in sorted(args.raw_dir.glob("*.json"))]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    print(f"wrote {len(rows)} rows to {args.output_csv}")
    return 0


def read_row(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    row.setdefault("error", "")
    return row


if __name__ == "__main__":
    raise SystemExit(main())
