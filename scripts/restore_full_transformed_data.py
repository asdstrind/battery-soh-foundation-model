"""Restore the complete historical transformed_data tree byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    if mode == "hardlink":
        os.link(source, temporary)
    else:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore all published transformed_data pickle files exactly."
    )
    parser.add_argument("dataset_root", type=Path, help="Downloaded HF dataset directory")
    parser.add_argument("output", type=Path, help="Destination transformed_data directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use hardlink to avoid duplicate disk usage when both paths share a filesystem",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Hash the restored files as well as the downloaded source files",
    )
    args = parser.parse_args()

    manifest_path = args.dataset_root / "metadata" / "full_transformed_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = args.dataset_root / str(manifest["repository_path"])
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Complete legacy data not found at {source_root}. "
            "Download with scripts/download_data.py --content full-legacy first."
        )

    planned: list[tuple[Path, Path, dict[str, object]]] = []
    for item in manifest["files"]:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        source = source_root / relative
        destination = args.output / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Downloaded size mismatch: {source}")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {destination}; pass --overwrite to replace files"
            )
        planned.append((source, destination, item))

    total_bytes = 0
    for source, destination, item in planned:
        print(f"Verifying source {item['path']}...", flush=True)
        expected_hash = str(item["sha256"])
        if sha256(source) != expected_hash:
            raise ValueError(f"Downloaded checksum mismatch: {source}")
        materialize(source, destination, args.mode)
        if destination.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Restored size mismatch: {destination}")
        if args.verify and sha256(destination) != expected_hash:
            raise ValueError(f"Restored checksum mismatch: {destination}")
        total_bytes += destination.stat().st_size

    if len(planned) != int(manifest["totals"]["files"]):
        raise ValueError("Restored file count mismatch")
    if total_bytes != int(manifest["totals"]["bytes"]):
        raise ValueError("Restored byte count mismatch")
    print(
        f"Restored {len(planned)} files and {total_bytes} bytes to {args.output} "
        f"using {args.mode} mode.",
        flush=True,
    )


if __name__ == "__main__":
    main()
