"""Exploratory data analysis plots for KDD and CIC datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.features_meta import KDD_FEATURE_GROUPS, CIC_FEATURE_GROUPS
from src.preprocess import imbalance_ratio

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_class_distribution(
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    """Bar plot of class counts (log scale)."""
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数 (log)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_group_pie(
    groups: dict[str, list],
    title: str,
    out_path: Path,
) -> None:
    """Pie chart of feature group proportions."""
    sizes = [len(v) for v in groups.values()]
    labels = list(groups.keys())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
    top_k: int = 15,
) -> dict[str, float]:
    """Top-K mutual information bar chart."""
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    idx = np.argsort(mi)[::-1][:top_k]
    top_names = [feature_names[i] for i in idx]
    top_mi = mi[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_mi, y=[str(n)[:40] for n in top_names], ax=ax, orient="h")
    ax.set_title(title)
    ax.set_xlabel("Mutual Information")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {top_names[i]: float(top_mi[i]) for i in range(len(top_names))}


def plot_correlation_heatmap(
    X: pd.DataFrame,
    title: str,
    out_path: Path,
    max_features: int = 25,
) -> None:
    """Correlation heatmap for top-variance features."""
    if X.shape[1] > max_features:
        variances = X.var().sort_values(ascending=False)
        cols = variances.head(max_features).index.tolist()
        X = X[cols]
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_eda_kdd(
    df: pd.DataFrame,
    X_processed: np.ndarray | None = None,
    y_processed: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Generate all KDD EDA artifacts."""
    out_dir = ARTIFACTS_DIR / "eda_kdd"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        df["label_category"],
        "KDD Cup 1999 攻击类别分布",
        out_dir / "class_distribution.png",
    )
    plot_feature_group_pie(
        KDD_FEATURE_GROUPS,
        "KDD 41维特征分组占比",
        out_dir / "feature_groups.png",
    )

    stats = {
        "imbalance_ratio": imbalance_ratio(
            pd.Categorical(df["label_category"]).codes
        ),
        "class_distribution": df["label_category"].value_counts().to_dict(),
    }
    return {"output_dir": str(out_dir), "stats": stats}


def run_eda_cic(df: pd.DataFrame) -> dict:
    """Generate all CIC EDA artifacts."""
    out_dir = ARTIFACTS_DIR / "eda_cic"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        df["Label"],
        "CIC-IDS-2017 攻击类别分布",
        out_dir / "class_distribution.png",
    )
    plot_feature_group_pie(
        CIC_FEATURE_GROUPS,
        "CIC 特征分组占比（近似）",
        out_dir / "feature_groups.png",
    )

    numeric = df.select_dtypes(include=[np.number])
    miss = df.isna().mean().sort_values(ascending=False).head(10).to_dict()

    stats = {
        "imbalance_ratio": imbalance_ratio(
            pd.Categorical(df["Label"]).codes
        ),
        "class_distribution": df["Label"].value_counts().to_dict(),
        "top_missing_rates": {k: float(v) for k, v in miss.items()},
    }
    return {"output_dir": str(out_dir), "stats": stats}


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    title: str,
    out_path: Path,
    normalize: bool = True,
) -> None:
    """Save confusion matrix heatmap."""
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_arr = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("真实")
    ax.set_xlabel("预测")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
