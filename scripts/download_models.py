#!/usr/bin/env python3
"""Download supported Hugging Face model snapshots for remote validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROFILES = {
    "granite-4.0-1b-base": "ibm-granite/granite-4.0-1b-base",
    "gemma4-e2b": "google/gemma-4-E2B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Granite/Gemma4 model snapshots.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES),
        help="Profile to download. Repeat for multiple profiles. Defaults to all profiles.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("checkpoints/huggingface"))
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated models.")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    profiles = args.profile or sorted(PROFILES)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        model_id = PROFILES[profile]
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            revision=args.revision,
            local_dir=None,
        )
        print(f"{profile}\t{model_id}\t{path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
