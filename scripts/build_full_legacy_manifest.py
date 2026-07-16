"""Build a byte-level manifest for the complete historical transformed_data tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Complete transformed_data directory")
    parser.add_argument("output", type=Path, help="Manifest JSON output path")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)

    files: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {path}")
        relative = path.relative_to(source).as_posix()
        print(f"Hashing {relative}...", flush=True)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise ValueError(f"No files found under {source}")

    manifest = {
        "schema_version": 1,
        "repository_path": "legacy_full/transformed_data",
        "restore_root": "transformed_data",
        "serialization": "python-pickle",
        "files": files,
        "totals": {
            "files": len(files),
            "bytes": sum(int(item["bytes"]) for item in files),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Manifested {manifest['totals']['files']} files and "
        f"{manifest['totals']['bytes']} bytes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
