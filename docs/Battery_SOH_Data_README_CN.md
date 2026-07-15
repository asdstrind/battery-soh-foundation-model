# 多源锂离子电池 SOH 处理数据集说明

版本：1.2（2026 年 7 月 15 日）  
关联论文：*Generalized Foundation Model for Lithium-Ion Battery State-of-Health Prediction with Distribution Metric Learning*  
DOI：https://doi.org/10.1016/j.est.2026.120566

## 1. 数据范围

本压缩包包含由公开锂离子电池老化数据集处理得到的循环级数据。不同来源的数据被统一为电压—电流—时间表示，用于多源电池健康状态（SOH）预测。

压缩包内同时包含各数据集的中间处理文件和降采样至 256 点的文件。若要复现或扩展本文的机器学习实验，建议使用 `downsampled_data_256` 文件。

本压缩包提供的是单电池级数据，不包含此前生成的 `T_<target>_S10_T1` 聚合任务文件夹。论文中目标域的单电池训练/测试划分见第 7 节及随附的 `train_test_splits.json`。

## 2. 标准数据集

本文使用以下 11 个标准数据域。这里的 `package` 表示同一来源数据集中的电池批次、化学体系或实验子集。

| 数据集目录 | Package | 电池数 | 256 点文件中的循环数 | 说明 |
|---|---:|---:|---:|---|
| `CALCE_dataset` | 1 | 4 | 3,895 | CALCE CS2 电池 |
| `HNEL_dataset` | 1 | 14 | 15,126 | HNEI/HNEL 电池 |
| `IECON_dataset` | 1 | 8 | 17,886 | IECON-2022 退化实验 |
| `NASA_dataset` | 1 | 4 | 644 | NASA 电池 B0005、B0006、B0007 和 B0018 |
| `Oxford_dataset` | 1 | 8 | 519 | Oxford Battery Degradation Dataset 1 |
| `SNL_LFP_dataset` | 1 | 21 | 4,531 | Sandia LFP 子集 |
| `SNL_NCA_dataset` | 1 | 18 | 1,639 | Sandia NCA 子集 |
| `SNL_NMC_dataset` | 1 | 22 | 1,804 | Sandia NMC 子集 |
| `TongJi_dataset` | 1 | 66 | 22,716 | NCA 电池 |
| `TongJi_dataset` | 2 | 55 | 27,838 | NCM 电池 |
| `TongJi_dataset` | 3 | 9 | 8,737 | NCM-NCA 混合体系电池 |
| `Toyota_MIT_dataset` | 1 | 46 | 38,765 | 2017-05-12 批次 |
| `Toyota_MIT_dataset` | 2 | 48 | 24,872 | 2017-06-30 批次 |
| `Toyota_MIT_dataset` | 3 | 46 | 50,961 | 2018-04-12 批次 |
| `XJTU_battery_dataset` | 1 | 8 | 3,195 | XJTU 批次/工况 1 |
| `XJTU_battery_dataset` | 2 | 15 | 3,697 | XJTU 批次/工况 2 |
| `XJTU_battery_dataset` | 3 | 8 | 4,592 | XJTU 批次/工况 3 |
| `XJTU_battery_dataset` | 4 | 8 | 5,654 | XJTU 批次/工况 4 |
| `XJTU_battery_dataset` | 5 | 8 | 1,959 | XJTU 批次/工况 5 |
| `XJTU_battery_dataset` | 6 | 8 | 8,372 | XJTU 批次/工况 6 |

### 关于辅助 TongJi 目录的重要说明

压缩包中还包含以下按化学体系单独导出的辅助目录：

- `TongJi_NCA_dataset`
- `TongJi_NCM_dataset`
- `TongJi_NCMNCA_dataset`

它们不是额外的实验域，也没有在 11 域配置中作为独立数据域使用。请勿将它们与 `TongJi_dataset` 合并，否则会重复计入同一批电池。此外，这三个辅助导出目录中的 SOH 序列比对应的数据序列短。所有分析请仅使用：

```text
TongJi_dataset/package_1  # NCA
TongJi_dataset/package_2  # NCM
TongJi_dataset/package_3  # NCM-NCA
```

## 3. 目录与文件结构

一个标准数据集目录通常具有以下结构：

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

顶层的两个 pickle 文件保存长度不固定的循环曲线，其中：

```python
data[package_id][battery_id][cycle_index] -> 形状为 (3, N) 的 ndarray
soh[package_id][battery_id][cycle_index]  -> 标量 SOH
```

建议使用的降采样 pickle 文件已经位于具体 package 目录下，因此不再包含最外层的 package 层级：

```python
data[battery_id][cycle_index] -> 形状为 (3, 256) 的 ndarray
soh[battery_id][cycle_index]  -> 标量 SOH
```

电池编号是从 1 开始的字符串，例如 `battery_1`、`battery_2`。编号只在各自 package 内有效，不能跨数据集直接比较。

## 4. 特征定义与单位

每个循环的三个通道按以下顺序排列：

| 行号 | 信号 | 单位 |
|---:|---|---|
| 0 | 电压 | V |
| 1 | 电流 | A |
| 2 | 循环内的经过时间 | s |

输入数据未进行归一化。电流符号遵循各原始数据集在换算为安培后的约定。

SOH 保存为无量纲容量比值，约为 `1.0` 表示相应的标称/参考容量。由于不同来源采用的参考容量和实验测量方式不同，SOH 应在各自原始数据集和 package 内解释。早期测得的容量可能略高于参考容量，因此 SOH 可能略大于 1。

## 5. 降采样方法

每条长度不固定的循环曲线按以下步骤转换为 256 个点：

1. 时间通道的所有数值减去该循环的第一个时间值，使经过时间从零开始。
2. 将原始采样位置表示为从 `0` 到 `N-1` 的等间隔索引。
3. 在该索引区间内取 256 个等间隔位置，分别对电压、电流和经过时间进行线性插值。
4. 插值后不再进行归一化或标准化。

最终每个循环的形状为 `(3, 256)`。

## 6. 论文实验使用的 SOH 范围

本压缩包保留各电池的可用 SOH 轨迹，因此可能包含 SOH 低于 80% 的循环。论文在构造任务时保留较早的退化区间，大致对应 SOH > 0.8。

实际实现按时间顺序扫描每个电池，寻找第一个满足前向均值 `mean(SOH[j:j+25]) <= 0.8` 的索引 `j`，随后保留区间 `[0, j)` 内的循环。采用不同的阈值或截断规则可能得到不同的循环数量。

## 7. 目标域训练/测试划分

目标域按完整电池划分，而不是将单个电池的循环随机拆分到训练集和测试集。下表中的编号对应指定 package 中的 `battery_<id>` 键。

| 目标任务 | Package | 有标签训练电池 | 留出测试电池 | 任务中使用的电池 | 当前 package 中的其他电池 |
|---|---:|---|---|---|---|
| HNEL | 1 | 1、2、3 | 4、5、6、7 | 1–7 | 8–14 未使用 |
| TongJi NCM | 2 | 1、2、3、6 | 4、5、7、8、9 | 1–9 | 10–55 未使用 |
| Toyota-MIT | 2 | 1、2、3、4 | 5–17 | 1–17 | 18–48 未使用 |

本划分没有单独的验证电池集合。

### 论文图表中展示的四个电池

每个任务的 SOH 预测曲线图和单电池数值比较表只展示了留出测试集中的四个电池。论文中的 `Battery 1` 至 `Battery 4` 是作图顺序标签，并不是数据文件中的原始 `battery_<id>` 编号。

**不确定性说明：** 下表中的原始电池编号是根据目前仍保留的历史任务文件、循环数量和绘图索引事后反向恢复的，当时并未另外保存一份可直接核验的作图电池编号记录。由于论文作图距今时间较长，而且旧聚合文件曾在内部重新编号电池，目前无法对这些原始编号进行完全独立的确认。因此，下表仅作为尽力恢复的参考，可能存在编号对应错误。该不确定性不用于重新定义前表及 `train_test_splits.json` 中的完整训练/测试划分。若需要严格复现论文中的某一条曲线，建议同时核对 SOH 轨迹和循环数量，不要只依赖下表中的编号。

| 目标任务 | 图中 Battery 1 | 图中 Battery 2 | 图中 Battery 3 | 图中 Battery 4 | 留出测试集中未在四幅曲线图展示的电池 |
|---|---:|---:|---:|---:|---|
| HNEL | `battery_4` | `battery_5` | `battery_6` | `battery_7` | 无 |
| TongJi NCM | `battery_4` | `battery_5` | `battery_7` | `battery_8` | `battery_9` |
| Toyota-MIT | `battery_5` | `battery_6` | `battery_11` | `battery_13` | `battery_7`–`battery_10`、`battery_12`、`battery_14`–`battery_17` |

在上述不确定性前提下，该表只说明论文图表可能采用的展示子集，不会改变正式训练/测试划分；复现实验时可以对上表“留出测试电池”列中的全部电池进行评估。

在域对齐实验中，训练电池的 SOH 标签用于目标域有监督回归，留出测试电池的 SOH 标签仅用于评估。训练电池和测试电池的输入曲线都会在不使用测试 SOH 标签的情况下参与 MMD 域对齐项，因此属于传导式无监督域适应设置。

对于每个目标任务，其余 10 个标准数据域作为源域。历史聚合脚本在每个源 package 中按顺序选择电池，直至经过 SOH 阈值截断后的累计循环数首次超过 2,000；对于包含多个 package 的源数据集，最多考虑前三个 package。本压缩包本身没有 2,000 循环上限，使用者也可以自行定义其他源域选择策略。

## 8. 读取示例

文件采用 Python pickle 序列化。请仅加载来自可信数据副本的 pickle 文件。

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

x = np.asarray(data["battery_1"], dtype=np.float32)  # (cycles, 3, 256)
y = np.asarray(soh["battery_1"], dtype=np.float32)   # (cycles,)
y = y.reshape(-1, 1)                                  # (cycles, 1)

assert len(x) == len(y)
assert x.shape[1:] == (3, 256)
assert np.isfinite(x).all() and np.isfinite(y).all()
```

## 9. 发布数据验证

已对标准 `downsampled_data_256` 文件执行以下检查：

- data 文件和 SOH 文件中的电池键集合一致；
- 每个电池的数据循环数与 SOH 数量一致；
- 每个循环的特征形状为 `(3, 256)`；
- SOH 数值均为有限值；
- package 数量、电池数量与第 2 节中的统计一致。

第 2 节所述的三个辅助 TongJi 导出目录不包含在该验证范围内，请勿使用。

## 10. 引用规范

如果使用本处理数据，请引用关联论文：

> Li et al. *Generalized Foundation Model for Lithium-Ion Battery State-of-Health Prediction with Distribution Metric Learning*. Journal of Energy Storage, 150, 120566, 2026. https://doi.org/10.1016/j.est.2026.120566

此外，请引用后续分析实际使用的每个原始数据集：

| 数据集 | 建议引用的来源 |
|---|---|
| CALCE | CALCE Battery Data Archive：https://web.calce.umd.edu/batteries/data/；He et al.，https://doi.org/10.1016/j.jpowsour.2011.08.040 |
| HNEL/HNEI | Devie, Baure, and Dubarry，https://doi.org/10.3390/en11051031 |
| IECON-2022 | Lyu, Zhang, Zio, and Yang，https://doi.org/10.1109/IECON49645.2022.9969107 |
| NASA | Saha and Goebel，*Battery Data Set*，NASA Prognostics Data Repository，2007 |
| Oxford | Birkl et al.，https://doi.org/10.1016/j.jpowsour.2016.12.011 |
| SNL | Preger et al.，https://doi.org/10.1149/1945-7111/abae37 |
| TongJi | Zhu et al.，https://doi.org/10.1038/s41467-022-29837-w；数据记录：https://doi.org/10.5281/zenodo.6379165 |
| Toyota-MIT | Severson et al.，https://doi.org/10.1038/s41560-019-0356-8；数据平台：https://data.matr.io/1/ |
| XJTU | Wang et al.，https://doi.org/10.1038/s41467-024-48779-z；数据集：https://doi.org/10.5281/zenodo.10963339；预处理/程序库论文：https://doi.org/10.1016/j.est.2023.109884 |

## 11. 使用与再分发

本数据副本仅供学术、非商业研究使用。所有原始数据集仍受各自提供者的使用条款、署名要求和许可证约束。引用本处理数据不能代替对原始数据来源的引用。接收者在进一步分发前应自行核对并遵守相应的原始条款。
