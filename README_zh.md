# 电池 SOH 基础模型

本 GitHub 仓库为公开代码仓库，保存多源锂离子电池健康状态预测与分布度量学习的学术研究代码。

本 GitHub 仓库只保存代码和文档。处理后数据与模型检查点分别存放在两个**私有 Hugging Face 仓库**：

- 数据集：`coinlearner/battery-soh-benchmark`
- 模型：`coinlearner/battery-soh-foundation-model`

每次发布对应的 GitHub、数据集和模型版本统一记录在 `metadata/release-manifest.yaml`。

## 目录说明

- `test_version/functions/`：模型、损失函数、数据加载和训练实现。
- `test_version/*.ipynb`：实验与绘图 Notebook。
- `test_version/data_process/`：各原始数据源的预处理 Notebook。
- `scripts/`：数据转换、下载、校验与发布工具。
- `docs/`：数据说明和按电芯划分的训练/测试清单。

## 下载私有数据

获得私有仓库权限并登录 Hugging Face 后执行：

```bash
hf auth login
python scripts/download_data.py --revision data-v1.0.0
python scripts/verify_data.py data/battery-soh-benchmark
```

令牌只能通过本机凭据或环境变量 `HF_TOKEN` 提供，禁止写入代码、Notebook 或配置文件。

## 当前发布状态

仓库中的 Notebook 是清除输出后的历史研究快照。新实验应以 HF Dataset 中带校验清单的发布数据为准。数据集和模型仍存放在私有 Hugging Face 仓库中；第三方原始数据不随公开 GitHub 仓库分发，数据来源和署名要求见 `docs/Battery_SOH_Data_README_CN.md`。

论文正式发表图表不随公开仓库分发；图表请以论文或出版方托管的正式版本为准。

本项目采用保留全部权利的学术查阅声明，具体条款见 `LICENSE`。引用本项目时请参考 `CITATION.cff`。
