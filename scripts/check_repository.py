"""Fail CI when files that belong on Hugging Face enter the GitHub repository."""

from __future__ import annotations

import argparse
from pathlib import Path


BLOCKED_DIRECTORIES = {
    "original_battery_data",
    "transformed_data",
    "result",
    "results",
    "checkpoints",
    "artifacts",
    "outputs",
}
BLOCKED_SUFFIXES = {
    ".pkl",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".mat",
    ".gz",
    ".zip",
    ".rar",
}
MAX_FILE_BYTES = 50 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        if any(part in BLOCKED_DIRECTORIES for part in relative.parts):
            failures.append(f"blocked directory: {relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked file type: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 50 MiB: {relative}")

    if failures:
        print("Repository integrity check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
