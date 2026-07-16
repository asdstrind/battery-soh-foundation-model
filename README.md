# Battery SOH Foundation Model

Research code for multi-source lithium-ion battery state-of-health prediction with distribution metric learning.

This repository intentionally contains code and documentation only. Processed datasets and model checkpoints are versioned in separate **private Hugging Face repositories**:

- Dataset: `coinlearner/battery-soh-benchmark`
- Model checkpoints: `coinlearner/battery-soh-foundation-model`

The exact GitHub, dataset, and model revisions for a release are pinned in `metadata/release-manifest.yaml`.

## Repository contents

- `test_version/functions/`: model, loss, loading, and training implementations.
- `test_version/*.ipynb`: experiment and plotting notebooks.
- `test_version/data_process/`: source-dataset preprocessing notebooks.
- `fig_dir/`: figures retained with the research snapshot.
- `scripts/`: dataset conversion, download, validation, and release utilities.
- `docs/`: dataset documentation and cell-level train/test assignments.

## Environment

Create an isolated Python environment and install the research dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch should be installed using the command appropriate for the local CPU/CUDA environment.

## Download the private dataset

Request or configure access to the private Hugging Face repository, then authenticate without putting a token in this repository:

```bash
hf auth login
python scripts/download_data.py --revision data-v1.0.1
python scripts/verify_data.py data/battery-soh-benchmark
```

`HF_TOKEN` may be supplied as an environment variable in CI. Never commit it.

## Restore the historical pickle layout

The Hub uses Parquet as the canonical, safe, language-neutral release format. To run historical code that expects the old nested pickle dictionaries, export a local compatibility copy:

```bash
python scripts/export_legacy_pickle.py \
  data/battery-soh-benchmark \
  transformed_data \
  --verify
```

The command verifies all input Parquet checksums, recreates the original canonical directory and filename pattern, reloads every generated pickle, and checks battery keys, cycle order, `(3, 256)` shapes, SOH lengths, and values. Existing output files are never replaced unless `--overwrite` is supplied.

The restored files contain float32 values because float32 is the canonical dtype of the Parquet release. They reproduce the historical downsampled data structure and float32 numerical values, but they are not byte-identical to the original pickle files. They do not recreate variable-length intermediate files or historical `T_<target>_S10_T1` aggregation folders.

> Pickle deserialization can execute code. Only load pickle files generated locally from a trusted, checksum-verified release.

## Restore the complete historical `transformed_data`

The private `data-v1.1.0` release also preserves all 74 historical pickle files byte-for-byte, including the non-downsampled variable-length curves and the auxiliary TongJi exports. Download only this exact legacy tree with:

```bash
python scripts/download_data.py \
  --revision data-v1.1.0 \
  --content full-legacy
```

Then restore and verify the complete directory:

```bash
python scripts/restore_full_transformed_data.py \
  data/battery-soh-benchmark \
  transformed_data \
  --verify
```

This downloads 12,253,820,162 bytes (11.412 GiB). Copy mode requires enough space for both the downloaded files and the restored directory. If both locations are on the same filesystem, `--mode hardlink --verify` avoids the second physical copy; do not modify hard-linked output files because they share storage with the download.

The full legacy tree is covered by `metadata/full_transformed_data_manifest.json`, which records the relative path, byte size, and SHA-256 of every file. This route restores exact historical bytes; the Parquet exporter above remains the recommended route when only canonical 256-point data are needed.

## Reproducibility status

The included notebooks are a cleaned historical research snapshot. Notebook outputs were removed before publication. Some preprocessing notebooks still document original source-specific layouts; the release dataset and its manifest are the canonical input for new work.

## Citation

If this work is useful, cite the paper in `CITATION.cff` and cite every original dataset used in an analysis. Full source attribution is in `docs/Battery_SOH_Data_README.md`.

## License and access

This repository is private and currently distributed under an all-rights-reserved research notice. Dataset and model access do not grant rights to redistribute the original third-party datasets.

Chinese documentation: [README_zh.md](README_zh.md)
