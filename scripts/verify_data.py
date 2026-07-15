"""Verify a local Hugging Face dataset release against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "metadata/manifest.json").read_text(encoding="utf-8"))
    cycles = 0
    expected_columns = {
        "dataset", "package_id", "battery_id", "cycle_index",
        "voltage", "current", "elapsed_time", "soh",
    }
    for package in manifest["packages"]:
        path = args.root / package["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != package["bytes"]:
            raise ValueError(f"Size mismatch: {path}")
        if sha256(path) != package["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        metadata = pq.read_metadata(path)
        if set(pq.read_schema(path).names) != expected_columns:
            raise ValueError(f"Schema mismatch: {path}")
        if metadata.num_rows != package["cycles"]:
            raise ValueError(f"Row count mismatch: {path}")
        cycles += metadata.num_rows
    if cycles != manifest["totals"]["cycles"]:
        raise ValueError("Total cycle count mismatch")
    print(f"Verified {len(manifest['packages'])} packages and {cycles} cycles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
