# Processed Multi-Source Lithium-Ion Battery Dataset

Version: 1.2 (15 July 2026)  
Associated paper: *Generalized Foundation Model for Lithium-Ion Battery State-of-Health Prediction with Distribution Metric Learning*  
DOI: https://doi.org/10.1016/j.est.2026.120566

## 1. Scope

This archive contains processed cycle-level data derived from public lithium-ion battery ageing datasets. The data were standardized into a common voltage-current-time representation for multi-source state-of-health (SOH) prediction.

The archive contains per-dataset intermediate files and 256-point downsampled files. The `downsampled_data_256` files are the recommended inputs for reproducing or extending the machine-learning experiments.

The archive contains per-cell data. It does **not** contain the previously aggregated `T_<target>_S10_T1` task folders. The cell-level target-domain assignments used in the study are documented in Section 7 and in the accompanying `train_test_splits.json` file.

## 2. Canonical datasets

The study uses the following 11 canonical domains. A `package` represents a battery batch, chemistry, or experimental subset within a source dataset.

| Dataset directory | Package | Cells | Cycles in the 256-point files | Notes |
|---|---:|---:|---:|---|
| `CALCE_dataset` | 1 | 4 | 3,895 | CALCE CS2 cells |
| `HNEL_dataset` | 1 | 14 | 15,126 | HNEI/HNEL cells |
| `IECON_dataset` | 1 | 8 | 17,886 | IECON-2022 degradation experiment |
| `NASA_dataset` | 1 | 4 | 644 | NASA cells B0005, B0006, B0007, and B0018 |
| `Oxford_dataset` | 1 | 8 | 519 | Oxford Battery Degradation Dataset 1 |
| `SNL_LFP_dataset` | 1 | 21 | 4,531 | Sandia LFP subset |
| `SNL_NCA_dataset` | 1 | 18 | 1,639 | Sandia NCA subset |
| `SNL_NMC_dataset` | 1 | 22 | 1,804 | Sandia NMC subset |
| `TongJi_dataset` | 1 | 66 | 22,716 | NCA cells |
| `TongJi_dataset` | 2 | 55 | 27,838 | NCM cells |
| `TongJi_dataset` | 3 | 9 | 8,737 | NCM-NCA blended cells |
| `Toyota_MIT_dataset` | 1 | 46 | 38,765 | 2017-05-12 batch |
| `Toyota_MIT_dataset` | 2 | 48 | 24,872 | 2017-06-30 batch |
| `Toyota_MIT_dataset` | 3 | 46 | 50,961 | 2018-04-12 batch |
| `XJTU_battery_dataset` | 1 | 8 | 3,195 | XJTU batch/protocol 1 |
| `XJTU_battery_dataset` | 2 | 15 | 3,697 | XJTU batch/protocol 2 |
| `XJTU_battery_dataset` | 3 | 8 | 4,592 | XJTU batch/protocol 3 |
| `XJTU_battery_dataset` | 4 | 8 | 5,654 | XJTU batch/protocol 4 |
| `XJTU_battery_dataset` | 5 | 8 | 1,959 | XJTU batch/protocol 5 |
| `XJTU_battery_dataset` | 6 | 8 | 8,372 | XJTU batch/protocol 6 |

### Important note about auxiliary TongJi directories

The archive also contains these auxiliary chemistry-specific exports:

- `TongJi_NCA_dataset`
- `TongJi_NCM_dataset`
- `TongJi_NCMNCA_dataset`

They are not additional experimental domains and were not used as separate domains in the 11-domain configuration. They should **not** be combined with `TongJi_dataset`, because doing so would duplicate the same underlying cells. In addition, these auxiliary exports have shorter SOH lists than their corresponding data lists. For all analysis, use only:

```text
TongJi_dataset/package_1  # NCA
TongJi_dataset/package_2  # NCM
TongJi_dataset/package_3  # NCM-NCA
```

## 3. Directory and file structure

A typical canonical dataset directory has the following structure:

```text
<dataset>_dataset/
├── <dataset>_data.pkl
├── <dataset>_SOH.pkl
└── downsampled_data_256/
    ├── package_1/
    │   ├── 256_<dataset>_all_battery_id_data.pkl
    │   └── 256_<dataset>_all_battery_id_soh.pkl
    └── package_n/
        ├── 256_<dataset>_all_battery_id_data.pkl
        └── 256_<dataset>_all_battery_id_soh.pkl
```

The two top-level pickle files contain the intermediate, variable-length cycle curves. Their structure is:

```python
data[package_id][battery_id][cycle_index] -> ndarray with shape (3, N)
soh[package_id][battery_id][cycle_index]  -> scalar SOH value
```

The recommended downsampled pickle files omit the outer package level because each file is already stored under a package directory:

```python
data[battery_id][cycle_index] -> ndarray with shape (3, 256)
soh[battery_id][cycle_index]  -> scalar SOH value
```

Battery identifiers are one-based strings such as `battery_1`, `battery_2`, and so on. Battery identifiers are local to each package and must not be compared across datasets.

## 4. Feature definition and units

For every cycle, the three rows are ordered as follows:

| Row | Signal | Unit |
|---:|---|---|
| 0 | Voltage | V |
| 1 | Current | A |
| 2 | Elapsed time within the cycle | s |

No input normalization was applied. Current signs follow the convention of the respective source dataset after unit conversion to amperes.

SOH is stored as a dimensionless capacity ratio. A value of approximately `1.0` represents the applicable nominal/reference capacity. Because the reference capacity and experimental measurement procedures differ by source dataset, SOH values should be interpreted within their original dataset and package. Early measured capacities may be slightly greater than the reference capacity.

## 5. Downsampling procedure

Each variable-length cycle was converted to 256 points as follows:

1. The first value of the time channel was subtracted from the entire time channel, so elapsed time starts at zero.
2. The original sample positions were represented by equally spaced indices from `0` to `N-1`.
3. Voltage, current, and elapsed time were independently linearly interpolated at 256 equally spaced positions in this sample-index interval.
4. No normalization or standardization was applied after interpolation.

The resulting per-cycle shape is `(3, 256)`.

## 6. SOH range used in the paper

The files in this archive retain the available SOH trajectories and can include cycles below 80% SOH. The paper experiments retained the earlier degradation region, approximately SOH > 0.8, during task construction.

The implementation scanned each cell in chronological order and found the first index `j` for which the forward mean `mean(SOH[j:j+25]) <= 0.8`. It then retained cycles in the interval `[0, j)`. Applying a different threshold rule may produce a different number of cycles.

## 7. Target-domain train/test assignments

The target split is by complete battery/cell, not by randomly splitting individual cycles. Battery IDs below refer to the corresponding `battery_<id>` keys in the indicated package.

| Target task | Package | Labeled training cells | Held-out test cells | Cells used in the task | Other cells in the current package |
|---|---:|---|---|---|---|
| HNEL | 1 | 1, 2, 3 | 4, 5, 6, 7 | 1-7 | 8-14 excluded |
| TongJi NCM | 2 | 1, 2, 3, 6 | 4, 5, 7, 8, 9 | 1-9 | 10-55 excluded |
| Toyota-MIT | 2 | 1, 2, 3, 4 | 5-17 | 1-17 | 18-48 excluded |

There is no separate validation-cell split in the supplied assignment.

### Cells shown in the paper figures

The four batteries shown in the per-task SOH prediction figures and numerical comparison tables are a display subset of the held-out test cells. The labels `Battery 1` to `Battery 4` in the paper are figure-order labels, not the original `battery_<id>` keys.

**Uncertainty notice:** The original cell-ID mapping below was reconstructed retrospectively from the surviving historical task files, cycle counts, and plotting indices; it was not preserved as a separate contemporaneous experiment record. Because considerable time has passed since the figures were produced, and because the historical aggregation files renumbered cells internally, the original IDs cannot now be independently verified with complete certainty. The mapping should therefore be treated as a best-effort reference and may contain errors. The complete train/test assignments in the preceding table and `train_test_splits.json` are not redefined by this figure mapping. For strict reproduction of an individual published curve, the SOH trajectory and cycle count should also be cross-checked instead of relying only on the ID below.

| Target task | Figure label 1 | Figure label 2 | Figure label 3 | Figure label 4 | Held-out test cells not shown in the four-curve figure |
|---|---:|---:|---:|---:|---|
| HNEL | `battery_4` | `battery_5` | `battery_6` | `battery_7` | None |
| TongJi NCM | `battery_4` | `battery_5` | `battery_7` | `battery_8` | `battery_9` |
| Toyota-MIT | `battery_5` | `battery_6` | `battery_11` | `battery_13` | `battery_7`-`battery_10`, `battery_12`, `battery_14`-`battery_17` |

Subject to the uncertainty stated above, this figure subset does not change the train/test split: evaluation may be performed on every cell listed in the held-out test column above.

For the domain-alignment experiment, SOH labels from the training cells were used for supervised target regression. SOH labels from held-out test cells were reserved for evaluation. Input curves from both labeled and held-out target cells were used without test SOH labels in the MMD domain-alignment term; this is a transductive unsupervised-domain-adaptation setting.

For each target task, the other 10 canonical domains served as source domains. The historical aggregation script selected cells sequentially within each source package until the accumulated post-threshold cycle count first exceeded 2,000 cycles, and considered at most the first three packages of a multi-package source dataset. The archive itself is not capped and allows users to define alternative source-selection policies.

## 8. Loading example

The files use Python pickle serialization. Only load pickle files obtained from a trusted copy of this archive.

```python
import pickle
import numpy as np

data_path = (
    "transformed_data/HNEL_dataset/downsampled_data_256/package_1/"
    "256_HNEL_all_battery_id_data.pkl"
)
soh_path = (
    "transformed_data/HNEL_dataset/downsampled_data_256/package_1/"
    "256_HNEL_all_battery_id_soh.pkl"
)

with open(data_path, "rb") as f:
    data = pickle.load(f)

with open(soh_path, "rb") as f:
    soh = pickle.load(f)

x = np.asarray(data["battery_1"], dtype=np.float32)       # (cycles, 3, 256)
y = np.asarray(soh["battery_1"], dtype=np.float32)        # (cycles,)
y = y.reshape(-1, 1)                                      # (cycles, 1)

assert len(x) == len(y)
assert x.shape[1:] == (3, 256)
assert np.isfinite(x).all() and np.isfinite(y).all()
```

## 9. Validation performed on the release

The canonical `downsampled_data_256` files were checked for:

- matching battery-key sets in data and SOH files;
- matching cycle counts for each battery;
- per-cycle feature shape `(3, 256)`;
- finite SOH values;
- consistency of the package and battery counts reported in Section 2.

The three auxiliary TongJi exports described in Section 2 are excluded from this validation and should not be used.

## 10. Citation

If this processed archive is used, please cite the associated study:

> Li et al. *Generalized Foundation Model for Lithium-Ion Battery State-of-Health Prediction with Distribution Metric Learning*. Journal of Energy Storage, 150, 120566, 2026. https://doi.org/10.1016/j.est.2026.120566

Please also cite the original source corresponding to every dataset used in a derived analysis:

| Dataset | Recommended source/citation |
|---|---|
| CALCE | CALCE Battery Data Archive: https://web.calce.umd.edu/batteries/data/ ; He et al., https://doi.org/10.1016/j.jpowsour.2011.08.040 |
| HNEL/HNEI | Devie, Baure, and Dubarry, https://doi.org/10.3390/en11051031 |
| IECON-2022 | Lyu, Zhang, Zio, and Yang, https://doi.org/10.1109/IECON49645.2022.9969107 |
| NASA | Saha and Goebel, *Battery Data Set*, NASA Prognostics Data Repository, 2007 |
| Oxford | Birkl et al., https://doi.org/10.1016/j.jpowsour.2016.12.011 |
| SNL | Preger et al., https://doi.org/10.1149/1945-7111/abae37 |
| TongJi | Zhu et al., https://doi.org/10.1038/s41467-022-29837-w ; data record: https://doi.org/10.5281/zenodo.6379165 |
| Toyota-MIT | Severson et al., https://doi.org/10.1038/s41560-019-0356-8 ; data platform: https://data.matr.io/1/ |
| XJTU | Wang et al., https://doi.org/10.1038/s41467-024-48779-z ; dataset: https://doi.org/10.5281/zenodo.10963339 ; preprocessing/library paper: https://doi.org/10.1016/j.est.2023.109884 |

## 11. Use and redistribution

This copy is provided for academic, non-commercial research. The original datasets remain subject to the terms, attribution requirements, and licenses of their respective providers. Citation of this processed archive does not replace citation of the original data sources. Recipients are responsible for checking the applicable original terms before further redistribution.
