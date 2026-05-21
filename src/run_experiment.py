"""Main experiment orchestrator — run locally or on Colab."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, QUICK_RUN
from src.data_cic import load_cic_chunked
from src.data_kdd import load_kdd_pipeline
from src.eda import plot_confusion_matrix, plot_correlation_heatmap, plot_mutual_information, run_eda_cic, run_eda_kdd
from src.models_dl import train_and_evaluate_cnn
from src.models_ml import ML_MODEL_NAMES, train_and_evaluate_ml
from src.optimize import (
    cross_dataset_transfer_experiment,
    train_ensemble,
    train_hierarchical_stack,
    train_with_smote,
)
from src.preprocess import preprocess_dataset


def _env_info() -> dict:
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = "not installed"
    try:
        import sklearn
        sk_version = sklearn.__version__
    except ImportError:
        sk_version = "unknown"
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "sklearn": sk_version,
        "tensorflow": tf_version,
        "quick_run": QUICK_RUN,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_all(skip_cic_download: bool = False) -> dict:
    """Execute full experiment pipeline."""
    results: dict = {
        "env": _env_info(),
        "eda": {},
        "preprocess": {},
        "models": [],
        "optimization": [],
        "innovation": [],
        "transfer": {},
    }

    # --- KDD ---
    print("Loading KDD...")
    kdd_df, kdd_meta = load_kdd_pipeline()
    results["eda"]["kdd"] = run_eda_kdd(kdd_df)
    results["kdd_meta"] = kdd_meta

    kdd_pp = preprocess_dataset(kdd_df, "label_category", "kdd", scale=True)
    results["preprocess"]["kdd"] = kdd_pp.meta

    # MI / correlation on preprocessed train
    eda_kdd_dir = ARTIFACTS_DIR / "eda_kdd"
    plot_mutual_information(
        kdd_pp.X_train, kdd_pp.y_train, kdd_pp.feature_names,
        "KDD Top-15 互信息特征", eda_kdd_dir / "mutual_information.png",
    )
    X_df = __import__("pandas").DataFrame(kdd_pp.X_train, columns=kdd_pp.feature_names)
    plot_correlation_heatmap(X_df, "KDD 特征相关性", eda_kdd_dir / "correlation.png")

    for model_name in ML_MODEL_NAMES:
        print(f"KDD ML: {model_name}")
        m = train_and_evaluate_ml(model_name, kdd_pp, "kdd")
        results["models"].append(m)
        _save_cm_plot(m, "kdd", model_name)

    print("KDD CNN...")
    cnn_kdd = train_and_evaluate_cnn(kdd_pp, "kdd")
    results["models"].append(cnn_kdd)
    _save_cm_plot(cnn_kdd, "kdd", "cnn_1d")

    # --- CIC ---
    print("Loading CIC...")
    try:
        cic_df, cic_meta = load_cic_chunked(use_cache=True)
    except FileNotFoundError:
        if skip_cic_download:
            print("CIC skipped (no cache, skip_cic_download=True)")
            cic_df, cic_meta = None, {}
        else:
            raise
    else:
        results["eda"]["cic"] = run_eda_cic(cic_df)
        results["cic_meta"] = cic_meta

        cic_pp = preprocess_dataset(cic_df, "Label", "cic", scale=True)
        results["preprocess"]["cic"] = cic_pp.meta

        eda_cic_dir = ARTIFACTS_DIR / "eda_cic"
        plot_mutual_information(
            cic_pp.X_train, cic_pp.y_train, cic_pp.feature_names,
            "CIC Top-15 互信息特征", eda_cic_dir / "mutual_information.png",
        )
        X_cic = __import__("pandas").DataFrame(cic_pp.X_train, columns=cic_pp.feature_names)
        plot_correlation_heatmap(X_cic, "CIC 特征相关性", eda_cic_dir / "correlation.png")

        for model_name in ML_MODEL_NAMES:
            print(f"CIC ML: {model_name}")
            m = train_and_evaluate_ml(model_name, cic_pp, "cic")
            results["models"].append(m)
            _save_cm_plot(m, "cic", model_name)

        print("CIC CNN...")
        cnn_cic = train_and_evaluate_cnn(cic_pp, "cic")
        results["models"].append(cnn_cic)
        _save_cm_plot(cnn_cic, "cic", "cnn_1d")

        # Optimization
        print("Ensemble KDD...")
        results["optimization"].append(train_ensemble(kdd_pp, "kdd"))
        print("Ensemble CIC...")
        results["optimization"].append(train_ensemble(cic_pp, "cic"))
        print("SMOTE CIC...")
        results["optimization"].append(train_with_smote(cic_pp, "cic"))

        # Innovation
        print("Hierarchical stack KDD...")
        results["innovation"].append(train_hierarchical_stack(kdd_pp, "kdd"))
        print("Hierarchical stack CIC...")
        results["innovation"].append(train_hierarchical_stack(cic_pp, "cic"))

        results["transfer"] = cross_dataset_transfer_experiment(kdd_pp, cic_pp)

    # Save metrics
    out_path = ARTIFACTS_DIR / "metrics.json"
    _serialize_results(results, out_path)
    print(f"Saved metrics to {out_path}")
    return results


def _save_cm_plot(metrics: dict, dataset: str, model: str) -> None:
    if "confusion_matrix" not in metrics:
        return
    report = metrics.get("classification_report", {})
    labels = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    out = ARTIFACTS_DIR / "confusion" / f"{dataset}_{model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"], labels,
        f"{dataset} - {model}", out,
    )


def _serialize_results(results: dict, path: Path) -> None:
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    import numpy as np
    path.write_text(
        json.dumps(_clean(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cic-download", action="store_true")
    args = parser.parse_args()
    run_all(skip_cic_download=args.skip_cic_download)
