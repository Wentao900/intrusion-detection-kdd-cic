"""Evaluation metrics including FPR and detection latency."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import LATENCY_SAMPLE_SIZE


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, FPR, per-class report."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    fpr = _macro_fpr(cm)

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "fpr_macro": float(fpr),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _macro_fpr(cm: np.ndarray) -> float:
    """Macro-averaged false positive rate across classes (one-vs-rest)."""
    fprs = []
    n = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = n - cm[i, :].sum() - cm[:, i].sum() + tp
        denom = fp + tn
        fprs.append(fp / denom if denom > 0 else 0.0)
    return float(np.mean(fprs))


def measure_latency_ms(
    model: Any,
    X: np.ndarray,
    n_samples: int | None = None,
) -> float:
    """Average inference latency in ms per sample."""
    n_samples = min(n_samples or LATENCY_SAMPLE_SIZE, len(X))
    X_sub = X[:n_samples]
    # Warmup
    _ = model.predict(X_sub[: min(100, n_samples)])

    t0 = time.perf_counter()
    _ = model.predict(X_sub)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / n_samples
