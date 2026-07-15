"""Build a deterministic checksum index for a staged private model repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted((args.root / "checkpoints").rglob("*.pt"))
    with (args.root / "checkpoint_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in files:
            relative = path.relative_to(args.root).as_posix()
            print(f"Hashing {relative}...", flush=True)
            writer.writerow([relative, path.stat().st_size, sha256(path)])
    print(f"Indexed {len(files)} checkpoints.")


if __name__ == "__main__":
    main()

