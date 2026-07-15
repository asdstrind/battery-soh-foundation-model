"""Convert canonical 256-point pickle files to validated Parquet packages.

Only load pickle files from a trusted local copy of this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_DATASETS = (
    "CALCE_dataset",
    "HNEL_dataset",
    "IECON_dataset",
    "NASA_dataset",
    "Oxford_dataset",
    "SNL_LFP_dataset",
    "SNL_NCA_dataset",
    "SNL_NMC_dataset",
    "TongJi_dataset",
    "Toyota_MIT_dataset",
    "XJTU_battery_dataset",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 256)


def convert_package(dataset_name: str, package_dir: Path, output_root: Path) -> dict:
    data_files = sorted(package_dir.glob("*_data.pkl"))
    soh_files = sorted(package_dir.glob("*_soh.pkl"))
    if len(data_files) != 1 or len(soh_files) != 1:
        raise ValueError(f"Expected one data/SOH pair in {package_dir}")
    data_path, soh_path = data_files[0], soh_files[0]

    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    with soh_path.open("rb") as handle:
        soh = pickle.load(handle)
    if set(data) != set(soh):
        raise ValueError(f"Battery key mismatch in {package_dir}")

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    battery_ids: list[np.ndarray] = []
    cycle_ids: list[np.ndarray] = []
    battery_stats: dict[str, int] = {}

    def battery_number(value: str) -> int:
        return int(value.rsplit("_", 1)[-1])

    for battery_id in sorted(data, key=battery_number):
        x = np.asarray(data[battery_id], dtype=np.float32)
        y = np.asarray(soh[battery_id], dtype=np.float32).reshape(-1)
        if x.ndim != 3 or x.shape[1:] != (3, 256):
            raise ValueError(f"Unexpected shape for {dataset_name}/{battery_id}: {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"Cycle count mismatch for {dataset_name}/{battery_id}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite values for {dataset_name}/{battery_id}")
        features.append(x)
        targets.append(y)
        battery_ids.append(np.repeat(battery_id, len(x)))
        cycle_ids.append(np.arange(len(x), dtype=np.int32))
        battery_stats[battery_id] = len(x)

    x_all = np.concatenate(features, axis=0)
    y_all = np.concatenate(targets, axis=0)
    battery_all = np.concatenate(battery_ids)
    cycle_all = np.concatenate(cycle_ids)
    package_id = int(package_dir.name.rsplit("_", 1)[-1])
    short_name = dataset_name.removesuffix("_dataset")

    table = pa.table(
        {
            "dataset": pa.array(np.repeat(short_name, len(x_all))),
            "package_id": pa.array(np.repeat(package_id, len(x_all)), type=pa.int16()),
            "battery_id": pa.array(battery_all),
            "cycle_index": pa.array(cycle_all, type=pa.int32()),
            "voltage": fixed_list(x_all[:, 0, :]),
            "current": fixed_list(x_all[:, 1, :]),
            "elapsed_time": fixed_list(x_all[:, 2, :]),
            "soh": pa.array(y_all, type=pa.float32()),
        }
    )

    output_dir = output_root / "data" / short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"package_{package_id}.parquet"
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        compression_level=6,
        row_group_size=4096,
        use_dictionary=["dataset", "battery_id"],
    )
    return {
        "dataset": short_name,
        "package_id": package_id,
        "batteries": len(battery_stats),
        "cycles": len(x_all),
        "battery_cycles": battery_stats,
        "file": output_path.relative_to(output_root).as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        "source": {
            "data_file": data_path.as_posix(),
            "data_sha256": sha256(data_path),
            "soh_file": soh_path.as_posix(),
            "soh_sha256": sha256(soh_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to transformed_data")
    parser.add_argument("output", type=Path, help="HF dataset staging directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    packages: list[dict] = []
    for dataset in CANONICAL_DATASETS:
        root = args.source / dataset / "downsampled_data_256"
        package_dirs = sorted(root.glob("package_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        if not package_dirs:
            raise FileNotFoundError(f"No packages found under {root}")
        for package_dir in package_dirs:
            print(f"Converting {dataset}/{package_dir.name}...", flush=True)
            packages.append(convert_package(dataset, package_dir, args.output))

    manifest = {
        "schema_version": 1,
        "format": "parquet",
        "feature_shape": [3, 256],
        "packages": packages,
        "totals": {
            "packages": len(packages),
            "batteries": sum(item["batteries"] for item in packages),
            "cycles": sum(item["cycles"] for item in packages),
            "bytes": sum(item["bytes"] for item in packages),
        },
    }
    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums = "".join(f'{item["sha256"]}  {item["file"]}\n' for item in packages)
    (metadata_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")
    print(json.dumps(manifest["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()

