"""Compare a restored legacy export with the trusted original pickle files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path, help="Trusted original transformed_data")
    parser.add_argument("restored", type=Path, help="Restored transformed_data")
    args = parser.parse_args()
    manifest = json.loads(
        (args.restored / "legacy_export_manifest.json").read_text(encoding="utf-8")
    )

    batteries = 0
    cycles = 0
    for package in manifest["packages"]:
        for kind in ("data", "soh"):
            relative = Path(str(package[f"{kind}_file"]))
            original = load_pickle(args.original / relative)
            restored = load_pickle(args.restored / relative)
            if set(original) != set(restored):
                raise ValueError(f"Battery key mismatch: {relative}")
            for battery_id in original:
                original_array = np.asarray(original[battery_id], dtype=np.float32)
                restored_array = np.asarray(restored[battery_id], dtype=np.float32)
                if original_array.shape != restored_array.shape:
                    raise ValueError(f"Shape mismatch: {relative}/{battery_id}")
                if not np.array_equal(original_array, restored_array):
                    raise ValueError(f"Float32 value mismatch: {relative}/{battery_id}")
        data = load_pickle(args.restored / Path(str(package["data_file"])))
        batteries += len(data)
        cycles += sum(len(value) for value in data.values())
        print(
            f"Matched {package['dataset']}/package_{package['package_id']}: "
            f"{len(data)} batteries, {package['cycles']} cycles",
            flush=True,
        )
    print(
        f"Round-trip comparison passed: {len(manifest['packages'])} packages, "
        f"{batteries} batteries, {cycles} cycles.",
        flush=True,
    )


if __name__ == "__main__":
    main()
