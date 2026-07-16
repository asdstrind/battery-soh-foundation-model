"""Download a pinned private dataset release from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", "coinlearner/battery-soh-benchmark"),
    )
    parser.add_argument("--revision", default="data-v1.1.0")
    parser.add_argument("--output", type=Path, default=Path("data/battery-soh-benchmark"))
    parser.add_argument(
        "--content",
        choices=("canonical", "full-legacy", "all"),
        default="canonical",
        help="Choose Parquet, byte-exact full transformed_data, or both",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    common_patterns = [
        "README.md",
        "DATASET_CARD_FULL*.md",
        "LICENSE_DATA.md",
        "scripts/**",
        "splits/**",
    ]
    canonical_patterns = [
        "data/**",
        "metadata/manifest.json",
        "metadata/checksums.sha256",
    ]
    legacy_patterns = [
        "legacy_full/transformed_data/**",
        "metadata/full_transformed_data_manifest.json",
    ]
    allow_patterns = list(common_patterns)
    if args.content in ("canonical", "all"):
        allow_patterns.extend(canonical_patterns)
    if args.content in ("full-legacy", "all"):
        allow_patterns.extend(legacy_patterns)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )
    print(
        f"Downloaded {args.content} content from "
        f"{args.repo_id}@{args.revision} to {args.output}"
    )


if __name__ == "__main__":
    main()
