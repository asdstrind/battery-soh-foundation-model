"""Download a pinned private dataset release from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", "coinlearner/battery-soh-benchmark"),
    )
    parser.add_argument("--revision", default="data-v1.0.0")
    parser.add_argument("--output", type=Path, default=Path("data/battery-soh-benchmark"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {args.repo_id}@{args.revision} to {args.output}")


if __name__ == "__main__":
    main()

