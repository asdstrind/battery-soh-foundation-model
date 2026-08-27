# Battery SOH Foundation Model

This public GitHub repository contains academic research code for multi-source lithium-ion battery state-of-health prediction with distribution metric learning.

This repository intentionally contains code and documentation only. Processed datasets and model checkpoints are versioned in separate **private Hugging Face repositories**:

- Dataset: `coinlearner/battery-soh-benchmark`
- Model checkpoints: `coinlearner/battery-soh-foundation-model`

The exact GitHub, dataset, and model revisions for a release are pinned in `metadata/release-manifest.yaml`.

## Repository contents

- `test_version/functions/`: model, loss, loading, and training implementations.
- `test_version/*.ipynb`: experiment and plotting notebooks.
- `test_version/data_process/`: source-dataset preprocessing notebooks.
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
python scripts/download_data.py --revision data-v1.0.0
python scripts/verify_data.py data/battery-soh-benchmark
```

`HF_TOKEN` may be supplied as an environment variable in CI. Never commit it.

## Reproducibility status

The included notebooks are a cleaned historical research snapshot. Notebook outputs were removed before publication. Some preprocessing notebooks still document original source-specific layouts; the release dataset and its manifest are the canonical input for new work.

Raw third-party datasets are not distributed in this public GitHub repository. Their sources and attribution requirements are documented in `docs/Battery_SOH_Data_README.md`.

Published-paper figures are intentionally excluded from this public repository. The paper/publisher-hosted version should be used as the authoritative source for figures.

## Citation

If this work is useful, cite the paper in `CITATION.cff` and cite every original dataset used in an analysis. Full source attribution is in `docs/Battery_SOH_Data_README.md`.

## License and access

This public repository is distributed under an all-rights-reserved academic inspection notice. The private dataset and model repositories, third-party datasets, and published materials remain subject to their respective access terms, licenses, copyright, and attribution requirements. See `LICENSE` for details.

Chinese documentation: [README_zh.md](README_zh.md)
