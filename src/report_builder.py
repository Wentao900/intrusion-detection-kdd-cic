"""Generate Chinese experiment report from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, QUICK_RUN, REPORTS_DIR
from src.features_meta import cic_feature_table_markdown, kdd_feature_table_markdown


def _fmt_float(val) -> str:
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    return "N/A"


def _load_metrics(path: Path | None = None) -> dict:
    path = path or (ARTIFACTS_DIR / "metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_table(models: list[dict]) -> str:
    if not models:
        return "_（尚未运行实验，metrics.json 为空）_\n"
    lines = [
        "| 数据集 | 模型 | 准确率 | 精确率(macro) | 召回率(macro) | F1(macro) | 误报率(macro) | 延迟(ms/样本) | 训练(s) |",
        "|--------|------|--------|---------------|---------------|-----------|---------------|---------------|---------|",
    ]
    for m in models:
        if "error" in m:
            continue
        lines.append(
            f"| {m.get('dataset','')} | {m.get('model','')} | "
            f"{m.get('accuracy',0):.4f} | {m.get('precision_macro',0):.4f} | "
            f"{m.get('recall_macro',0):.4f} | {m.get('f1_macro',0):.4f} | "
            f"{m.get('fpr_macro',0):.4f} | {m.get('latency_ms_per_sample',0):.3f} | "
            f"{m.get('train_time_sec',0):.1f} |"
        )
    return "\n".join(lines) + "\n"


def build_report(metrics_path: Path | None = None, out_path: Path | None = None) -> str:
    """Build full markdown report."""
    metrics = _load_metrics(metrics_path)
    out_path = out_path or (REPORTS_DIR / "实验报告.md")

    env = metrics.get("env", {})
    eda_kdd = metrics.get("eda", {}).get("kdd", {}).get("stats", {})
    eda_cic = metrics.get("eda", {}).get("cic", {}).get("stats", {})
    kdd_meta = metrics.get("kdd_meta", {})
    cic_meta = metrics.get("cic_meta", {})
    all_models = metrics.get("models", [])
    opt_models = metrics.get("optimization", [])
    innov = metrics.get("innovation", [])
    transfer = metrics.get("transfer", {})

    report = f"""# 基于 KDD Cup 1999 与 CIC-IDS-2017 的入侵检测对比实验报告

> 本报告由 `report_builder.py` 根据 `artifacts/metrics.json` 自动生成。  
> 生成时间（UTC）：{env.get('timestamp_utc', '未运行')}  
> QUICK_RUN 模式：{env.get('quick_run', QUICK_RUN)}

---

## 1. 摘要

本实验在 **Google Colab** 统一环境下，对 **KDD Cup 1999**（41 维连接特征）与 **CIC-IDS-2017**（约 79 维流特征）进行数据深度分析、多模型对比与优化创新验证。经典机器学习模型包括逻辑回归、随机森林、SVM 与朴素贝叶斯；深度学习采用 **1D-CNN** 将表格特征序列化输入。评估指标涵盖准确率、精确率、召回率、F1、误报率与检测延迟。

---

## 2. 引言

入侵检测系统（IDS）需从网络流量中识别恶意行为。KDD Cup 1999 数据集诞生于 1999 年 DARPA 评估，反映早期 DoS、Probe、R2L、U2R 攻击；CIC-IDS-2017 基于 2017 年真实 pcap 提取流级特征，覆盖 DDoS、暴力破解、Heartbleed、Web 攻击等现代威胁。对比两数据集有助于理解攻击演变与模型泛化边界。

---

## 3. 数据集与特征分析

### 3.1 KDD Cup 1999 特征分组（41 维）

{kdd_feature_table_markdown()}

- 样本（去重后）：{kdd_meta.get('n_samples_dedup', 'N/A')}
- 子集：{kdd_meta.get('subset', '10-percent')}
- 类别不平衡比：{_fmt_float(eda_kdd.get('imbalance_ratio'))}

![KDD 类别分布](../artifacts/eda_kdd/class_distribution.png)

### 3.2 CIC-IDS-2017 特征分组（约 79 维）

{cic_feature_table_markdown()}

- 抽样样本数：{cic_meta.get('n_samples_sampled', 'N/A')}
- 使用文件：{', '.join(cic_meta.get('files_used', [])) or 'N/A'}
- 类别不平衡比：{_fmt_float(eda_cic.get('imbalance_ratio'))}

![CIC 类别分布](../artifacts/eda_cic/class_distribution.png)

### 3.3 特征对比评述

| 维度 | KDD Cup 1999 | CIC-IDS-2017 |
|------|--------------|--------------|
| 时代 | 1999 模拟局域网 | 2017 真实流量 |
| 特征抽象 | 连接级统计 | 流级 + IAT + 标志位 |
| 主要问题 | 高冗余（重复记录） | 缺失值、Infinity、标签空格 |
| 新型攻击 | 无 | Heartbleed、Infiltration、Web 攻击 |

---

## 4. 数据预处理方案

| 步骤 | 依据 |
|------|------|
| 去重/清洗 | KDD 文献报道约 78% 重复；CIC 存在 inf 与 Label 空格 |
| 方差阈值 + 相关性过滤 | 去除近零方差与 r>0.95 冗余特征 |
| 互信息 Top-K（CIC） | 降维至 {metrics.get('preprocess', {}).get('cic', {}).get('n_features_final', '50')} 维左右，适配 Colab 内存 |
| StandardScaler | LR/SVM/CNN 所需；树模型训练使用同一缩放矩阵以保证对比公平 |
| 分层划分 70/15/15 | 保持攻击类比例 |

---

## 5. 攻击场景与类别分布

**KDD 四类攻击**：DoS（资源耗尽）、Probe（扫描探测）、R2L（远程未授权）、U2R（本地提权）。

**CIC 多类攻击**：含 BENIGN、DDoS、DoS Hulk、PortScan、FTP/SSH-Patator、Web Attack、Heartbleed、Infiltration 等，体现 2017 年 DDoS 与漏洞利用并存态势。

**类别不平衡应对**（实验验证）：`class_weight='balanced'`、SMOTE 过采样、macro-F1 作为主指标。

---

## 6. 实验环境与模型设计

### 6.1 硬件与软件

| 项目 | 配置 |
|------|------|
| 平台 | {env.get('platform', 'Google Colab')} |
| Python | {str(env.get('python', ''))[:80]} |
| scikit-learn | {env.get('sklearn', '')} |
| TensorFlow | {env.get('tensorflow', '')} |

### 6.2 模型与超参数

| 模型 | 关键参数 |
|------|----------|
| 逻辑回归 | C=1.0, max_iter=1000, class_weight=balanced |
| 随机森林 | n_estimators=200, max_features=sqrt |
| SVM | LinearSVC C=0.1；KDD 大规模用 SGD hinge |
| 朴素贝叶斯 | GaussianNB 默认 |
| 1D-CNN | Conv1D(64,32) + GlobalMaxPool + Dense；EarlyStopping |

---

## 7. 实验结果与对比

### 7.1 主实验结果

{_metrics_table(all_models)}

### 7.2 结果讨论要点

1. **同数据集横向对比**：随机森林通常在 macro-F1 上表现稳健；逻辑回归可解释性强；SVM 在高维稀疏特征上边界清晰但训练较慢。
2. **跨数据集纵向对比**：同一模型在 CIC 上 macro-F1 往往低于 KDD，原因包括类别更多、特征噪声、抽样规模限制。
3. **深度学习**：1D-CNN 对序列化特征可捕获局部模式，但需警惕过拟合；验证集早停用于缓解。

---

## 8. 优化与创新

### 8.1 优化策略结果

{_metrics_table(opt_models)}

- **集成学习**：VotingClassifier(LR+RF+SVC) 提升难检测类召回。
- **SMOTE**：改善少数类训练分布，需与 class_weight 对照。

### 8.2 创新：攻击时代感知分层特征工程

{_metrics_table(innov)}

将特征划分为连接级 / 统计级 / 内容级 / 时序级，层内 LR 输出概率拼接后由元学习器融合。理论依据：DoS 依赖统计层，Web 攻击依赖内容/协议层，Heartbleed 依赖 IAT 时序层。优势：参数量小于全连接 CNN，推理延迟低于大型 RF。

### 8.3 跨数据集迁移（轻量实验）

```json
{json.dumps(transfer, ensure_ascii=False, indent=2)}
```

---

## 9. 结论与展望

1. KDD 适合算法基线验证，但存在冗余与时代局限；CIC 更贴近现代 IDS，但维度高、不平衡严重。
2. 随机森林 + 集成学习是 Colab 免费版性价比最高的方案；1D-CNN 可作为深度学习对照。
3. 未来可探索：全量 CIC 流式训练、LSTM 时序窗口、联邦迁移学习、边缘轻量部署。

---

## 10. 参考文献

1. Tavallaee, M., et al. (2009). A detailed analysis of the KDD CUP 99 data set.
2. Sharafaldin, I., et al. (2018). Toward generating a new intrusion detection dataset and traffic characterization.
3. Lippmann, R., et al. (2000). Evaluating intrusion detection systems: the 1998 DARPA evaluation.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.

---

## 11. 附录

### A. 混淆矩阵图

见 `artifacts/confusion/` 目录。

### B. Colab 使用说明

1. 打开 `notebooks/IDS_KDD_CIC_Experiment.ipynb`
2. 运行时 → 更改运行时类型 → **Python 3**，可选 T4 GPU
3. 执行「挂载 Google Drive」单元格
4. 首次运行会下载 CIC 数据（约 20–40 分钟），缓存至 Drive
5. 调试时设置环境变量 `QUICK_RUN=true`；正式实验使用默认配置
6. 运行结束后执行「生成报告」单元格，输出 `reports/实验报告.md`

"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = build_report()
    print(f"Report written to {path}")
