# 电池 SOH 基础模型

本仓库保存多源锂离子电池健康状态预测与分布度量学习的研究代码。

本 GitHub 仓库只保存代码和文档。处理后数据与模型检查点分别存放在两个**私有 Hugging Face 仓库**：

- 数据集：`coinlearner/battery-soh-benchmark`
- 模型：`coinlearner/battery-soh-foundation-model`

每次发布对应的 GitHub、数据集和模型版本统一记录在 `metadata/release-manifest.yaml`。

## 目录说明

- `test_version/functions/`：模型、损失函数、数据加载和训练实现。
- `test_version/*.ipynb`：实验与绘图 Notebook。
- `test_version/data_process/`：各原始数据源的预处理 Notebook。
- `fig_dir/`：研究快照中保留的论文和分析图片。
- `scripts/`：数据转换、下载、校验与发布工具。
- `docs/`：数据说明和按电芯划分的训练/测试清单。

## 下载私有数据

获得私有仓库权限并登录 Hugging Face 后执行：

```bash
hf auth login
python scripts/download_data.py --revision data-v1.0.1
python scripts/verify_data.py data/battery-soh-benchmark
```

## 恢复旧版 pickle 目录

Hugging Face 上的规范存储格式是 Parquet。若要运行依赖旧版嵌套 pickle 字典的历史训练代码，可以生成一份本地兼容副本：

```bash
python scripts/export_legacy_pickle.py \
  data/battery-soh-benchmark \
  transformed_data \
  --verify
```

该命令会先校验所有 Parquet 文件的 SHA-256，然后恢复原有规范目录与文件命名；`--verify` 会重新加载每个生成的 pickle，检查电池键、循环顺序、`(3, 256)` 形状、SOH 长度与数值。除非显式传入 `--overwrite`，否则不会覆盖已有文件。

恢复文件采用 Parquet 发布版的规范 `float32` 数值。它们在旧版 downsampled 数据结构和 float32 数值上等价，但不保证与历史 pickle 字节级一致，也不能恢复未发布的变长中间文件或历史 `T_<target>_S10_T1` 聚合目录。

> pickle 反序列化可能执行代码，只应加载由可信、校验通过的数据版本在本地生成的文件。

令牌只能通过本机凭据或环境变量 `HF_TOKEN` 提供，禁止写入代码、Notebook 或配置文件。

## 当前发布状态

仓库中的 Notebook 是清除输出后的历史研究快照。新实验应以 HF Dataset 中带校验清单的发布数据为准。项目暂时保持私有，代码采用保留全部权利的研究使用声明；第三方原始数据的再分发权不随项目访问权限一并授予。
