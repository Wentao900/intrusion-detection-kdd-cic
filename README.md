# KDD vs CIC-IDS-2017 入侵检测 Colab 实验

在 **Google Colab** 上对比 KDD Cup 1999 与 CIC-IDS-2017 数据集，完成数据分析、机器学习/深度学习建模、优化与创新验证。

## 快速开始（Colab）

1. 在 Colab 中打开本仓库或执行笔记本中的 `git clone`（已指向 [Wentao900/intrusion-detection-kdd-cic](https://github.com/Wentao900/intrusion-detection-kdd-cic)）
2. 打开 [`notebooks/IDS_KDD_CIC_Experiment.ipynb`](notebooks/IDS_KDD_CIC_Experiment.ipynb)
3. **运行时 → 更改运行时类型** → Python 3（可选 GPU）
4. 按顺序运行全部单元格
5. 报告输出：`reports/实验报告.md`

### 调试模式

```python
import os
os.environ["QUICK_RUN"] = "true"  # 1/5 抽样，快速验证
```

### CIC 数据下载说明

旧版 AWS 单文件链接已 **404**。本项目改为下载 UNB 官方 **`MachineLearningCSV.zip`**（约 224 MB）并自动解压，首次 Colab 运行约 5–15 分钟（视网络而定）。

若自动下载失败，请从 [CIC IDS 2017 官网](https://www.unb.ca/cic/datasets/ids-2017.html) 手动下载 zip，放到 `data/cache/cic_raw/MachineLearningCSV.zip` 后重新运行。

### Drive 缓存

笔记本会挂载 Google Drive，将 `data/cache/` 与 `artifacts/` 存到 Drive，避免重复下载 CIC。

## 逻辑校验（无需安装 ML 依赖）

```bash
python3 scripts/verify_logic.py
```

## 本地运行（可选，在 Colab 或已配置环境中执行）

```bash
pip install -r requirements-colab.txt
PYTHONPATH=. python -m src.run_experiment
PYTHONPATH=. python -m src.report_builder
```

## 项目结构

| 路径 | 说明 |
|------|------|
| `src/config.py` | 全局配置、QUICK_RUN |
| `src/data_kdd.py` | KDD 加载与标签映射 |
| `src/data_cic.py` | CIC 分块下载与抽样 |
| `src/preprocess.py` | 预处理管道 |
| `src/models_ml.py` | LR / RF / SVM / NB |
| `src/models_dl.py` | 1D-CNN |
| `src/optimize.py` | 集成、SMOTE、分层 Stack |
| `src/eda.py` | 可视化 |
| `src/report_builder.py` | 自动生成中文报告 |

## 实验内容

- **（一）** 41 vs 79 维特征对比、预处理、攻击场景与不平衡分析
- **（二）** 4 种经典 ML + 1D-CNN，统一指标（含 FPR、延迟）
- **（三）** 集成学习、SMOTE、时代感知分层特征创新

## 参考

- [KDD Cup 99](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
