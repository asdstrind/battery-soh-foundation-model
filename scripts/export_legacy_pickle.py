"""Export the Parquet release to the historical downsampled pickle layout.

The generated pickle files are intended only for compatibility with the
historical training code. Pickle files must only be loaded from trusted paths.
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


REQUIRED_COLUMNS = (
    "battery_id",
    "cycle_index",
    "voltage",
    "current",
    "elapsed_time",
    "soh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def battery_number(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Unsupported battery identifier: {value}") from error


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type) or array.type.list_size != 256:
        raise ValueError(f"Expected fixed-size list[256], got {array.type}")
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(array), 256)


def load_package(parquet_path: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    table = pq.read_table(parquet_path, columns=list(REQUIRED_COLUMNS))
    if tuple(table.column_names) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns in {parquet_path}: {table.column_names}")

    battery_ids = np.asarray(table["battery_id"].to_pylist(), dtype=object)
    cycle_indices = table["cycle_index"].combine_chunks().to_numpy(zero_copy_only=False)
    voltage = fixed_list_to_numpy(table["voltage"])
    current = fixed_list_to_numpy(table["current"])
    elapsed_time = fixed_list_to_numpy(table["elapsed_time"])
    targets = np.asarray(
        table["soh"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float32
    )
    features = np.stack((voltage, current, elapsed_time), axis=1)

    data: dict[str, list[np.ndarray]] = {}
    soh: dict[str, np.ndarray] = {}
    for battery_id in sorted(set(battery_ids), key=battery_number):
        positions = np.flatnonzero(battery_ids == battery_id)
        order = np.argsort(cycle_indices[positions], kind="stable")
        positions = positions[order]
        actual_cycles = np.asarray(cycle_indices[positions], dtype=np.int64)
        expected_cycles = np.arange(len(positions), dtype=np.int64)
        if not np.array_equal(actual_cycles, expected_cycles):
            raise ValueError(
                f"Non-contiguous cycle indices for {battery_id} in {parquet_path}"
            )
        ordered_features = np.asarray(features[positions], dtype=np.float32)
        data[str(battery_id)] = [cycle.copy() for cycle in ordered_features]
        soh[str(battery_id)] = np.asarray(targets[positions], dtype=np.float32).copy()
    return data, soh


def write_pickle(path: Path, value: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def verify_pickles(
    data_path: Path,
    soh_path: Path,
    expected_data: dict[str, list[np.ndarray]],
    expected_soh: dict[str, np.ndarray],
) -> int:
    with data_path.open("rb") as handle:
        actual_data = pickle.load(handle)
    with soh_path.open("rb") as handle:
        actual_soh = pickle.load(handle)
    if set(actual_data) != set(expected_data) or set(actual_soh) != set(expected_soh):
        raise ValueError(f"Battery key mismatch after writing {data_path.parent}")

    cycles = 0
    for battery_id in sorted(expected_data, key=battery_number):
        actual_x = np.asarray(actual_data[battery_id], dtype=np.float32)
        expected_x = np.asarray(expected_data[battery_id], dtype=np.float32)
        actual_y = np.asarray(actual_soh[battery_id], dtype=np.float32)
        expected_y = np.asarray(expected_soh[battery_id], dtype=np.float32)
        if actual_x.shape[1:] != (3, 256) or actual_x.shape != expected_x.shape:
            raise ValueError(f"Feature shape mismatch for {battery_id}")
        if actual_y.shape != expected_y.shape or len(actual_x) != len(actual_y):
            raise ValueError(f"SOH shape mismatch for {battery_id}")
        if not np.array_equal(actual_x, expected_x) or not np.array_equal(actual_y, expected_y):
            raise ValueError(f"Value mismatch for {battery_id}")
        cycles += len(actual_x)
    return cycles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the historical downsampled pickle directory structure."
    )
    parser.add_argument("dataset_root", type=Path, help="Downloaded HF dataset directory")
    parser.add_argument("output", type=Path, help="Output transformed_data directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Reload and validate every pickle")
    parser.add_argument(
        "--skip-input-checksums",
        action="store_true",
        help="Skip Parquet SHA-256 validation (not recommended)",
    )
    args = parser.parse_args()

    manifest_path = args.dataset_root / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported: list[dict[str, object]] = []
    total_cycles = 0

    for package in manifest["packages"]:
        parquet_path = args.dataset_root / package["file"]
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        if not args.skip_input_checksums and sha256(parquet_path) != package["sha256"]:
            raise ValueError(f"Input checksum mismatch: {parquet_path}")

        dataset = str(package["dataset"])
        package_id = int(package["package_id"])
        print(f"Exporting {dataset}/package_{package_id}...", flush=True)
        data, soh = load_package(parquet_path)
        destination = (
            args.output
            / f"{dataset}_dataset"
            / "downsampled_data_256"
            / f"package_{package_id}"
        )
        prefix = f"256_{dataset}_all_battery_id"
        data_path = destination / f"{prefix}_data.pkl"
        soh_path = destination / f"{prefix}_soh.pkl"
        write_pickle(data_path, data, args.overwrite)
        write_pickle(soh_path, soh, args.overwrite)

        package_cycles = sum(len(values) for values in data.values())
        if args.verify:
            verified_cycles = verify_pickles(data_path, soh_path, data, soh)
            if verified_cycles != package_cycles:
                raise ValueError(f"Verification cycle count mismatch in {destination}")
        if package_cycles != int(package["cycles"]):
            raise ValueError(f"Manifest cycle count mismatch in {destination}")
        total_cycles += package_cycles
        exported.append(
            {
                "dataset": dataset,
                "package_id": package_id,
                "batteries": len(data),
                "cycles": package_cycles,
                "data_file": data_path.relative_to(args.output).as_posix(),
                "data_sha256": sha256(data_path),
                "soh_file": soh_path.relative_to(args.output).as_posix(),
                "soh_sha256": sha256(soh_path),
            }
        )

    if total_cycles != int(manifest["totals"]["cycles"]):
        raise ValueError("Total cycle count mismatch")
    export_manifest = {
        "schema_version": 1,
        "source_manifest_sha256": sha256(manifest_path),
        "numeric_dtype": "float32",
        "packages": exported,
        "totals": {"packages": len(exported), "cycles": total_cycles},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "legacy_export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(exported)} packages and {total_cycles} cycles to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
