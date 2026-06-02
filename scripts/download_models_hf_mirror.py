#!/usr/bin/env python3
"""Download supported model snapshots from a Hugging Face mirror endpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROFILES = {
    "granite-4.0-1b-base": "ibm-granite/granite-4.0-1b-base",
    "gemma4-e2b": "google/gemma-4-E2B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Granite/Gemma4 models from hf-mirror.com.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES),
        help="Profile to download. Repeat for multiple profiles. Defaults to all profiles.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/hf-mirror"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint

    # Import after setting HF_ENDPOINT so huggingface_hub picks up the mirror.
    from huggingface_hub import snapshot_download

    profiles = args.profile or sorted(PROFILES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        model_id = PROFILES[profile]
        local_dir = args.output_dir / profile
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {profile} from {args.endpoint} -> {local_dir}", flush=True)
        path = snapshot_download(
            repo_id=model_id,
            revision=args.revision,
            token=args.token,
            local_dir=local_dir,
            max_workers=args.max_workers,
        )
        print(f"{profile}\t{model_id}\t{path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
