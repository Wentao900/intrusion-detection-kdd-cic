"""Classic ML models: LR, RF, SVM, GaussianNB."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from src.config import RF_N_ESTIMATORS, RANDOM_STATE
from src.metrics import compute_metrics, measure_latency_ms
from src.preprocess import PreprocessResult


def get_model(name: str, n_classes: int, dataset: str) -> Any:
    """Factory for sklearn classifiers."""
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial" if n_classes > 2 else "auto",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced",
            max_features="sqrt" if dataset == "cic" else "sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "svm":
        if dataset == "kdd" and n_classes > 2:
            return SGDClassifier(
                loss="hinge",
                class_weight="balanced",
                alpha=1e-4,
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        return LinearSVC(C=0.1, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    if name == "naive_bayes":
        return GaussianNB()
    raise ValueError(f"Unknown model: {name}")


ML_MODEL_NAMES = ["logistic_regression", "random_forest", "svm", "naive_bayes"]


def train_and_evaluate_ml(
    model_name: str,
    data: PreprocessResult,
    dataset: str,
) -> dict[str, Any]:
    """Train ML model and return metrics dict."""
    n_classes = len(data.label_encoder.classes_)
    model = get_model(model_name, n_classes, dataset)

    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(data.X_test)
    metrics = compute_metrics(
        data.y_test, y_pred, label_names=list(data.label_encoder.classes_)
    )
    metrics["latency_ms_per_sample"] = measure_latency_ms(model, data.X_test)
    metrics["train_time_sec"] = train_time
    metrics["model"] = model_name
    metrics["dataset"] = dataset
    metrics["model_type"] = "ml"

    # Validation F1 for model selection logging
    y_val_pred = model.predict(data.X_val)
    from sklearn.metrics import f1_score
    metrics["val_f1_macro"] = float(
        f1_score(data.y_val, y_val_pred, average="macro", zero_division=0)
    )
    return metrics
