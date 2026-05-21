"""Optimization: ensemble, SMOTE, era-aware hierarchical stacking, cross-dataset transfer."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from src.config import RANDOM_STATE
from src.features_meta import CROSS_DATASET_ALIGNED, ERA_AWARE_LAYERS_CIC, ERA_AWARE_LAYERS_KDD
from src.metrics import compute_metrics, measure_latency_ms
from src.models_ml import get_model
from src.preprocess import PreprocessResult


def build_voting_ensemble(dataset: str, n_classes: int) -> VotingClassifier:
    """Soft-voting ensemble of LR + RF + LinearSVC."""
    estimators = [
        ("lr", get_model("logistic_regression", n_classes, dataset)),
        ("rf", get_model("random_forest", n_classes, dataset)),
        ("svc", get_model("svm", n_classes, dataset)),
    ]
    return VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)


def train_ensemble(
    data: PreprocessResult,
    dataset: str,
) -> dict[str, Any]:
    """Train voting ensemble and evaluate."""
    n_classes = len(data.label_encoder.classes_)
    model = build_voting_ensemble(dataset, n_classes)

    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(data.X_test)
    metrics = compute_metrics(
        data.y_test, y_pred, label_names=list(data.label_encoder.classes_)
    )
    metrics["latency_ms_per_sample"] = measure_latency_ms(model, data.X_test)
    metrics["train_time_sec"] = train_time
    metrics["model"] = "voting_ensemble"
    metrics["dataset"] = dataset
    metrics["model_type"] = "ensemble"
    metrics["val_f1_macro"] = float(
        f1_score(data.y_val, model.predict(data.X_val), average="macro", zero_division=0)
    )
    return metrics


def train_with_smote(
    data: PreprocessResult,
    dataset: str,
    base_model: str = "random_forest",
) -> dict[str, Any]:
    """Compare RF with SMOTE oversampling on training set only."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return {"error": "imbalanced-learn not installed", "dataset": dataset}

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    try:
        X_res, y_res = smote.fit_resample(data.X_train, data.y_train)
    except ValueError as e:
        return {"error": str(e), "dataset": dataset, "model": f"smote_{base_model}"}

    n_classes = len(data.label_encoder.classes_)
    model = get_model(base_model, n_classes, dataset)
    t0 = time.perf_counter()
    model.fit(X_res, y_res)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(data.X_test)
    metrics = compute_metrics(
        data.y_test, y_pred, label_names=list(data.label_encoder.classes_)
    )
    metrics["latency_ms_per_sample"] = measure_latency_ms(model, data.X_test)
    metrics["train_time_sec"] = train_time
    metrics["model"] = f"smote_{base_model}"
    metrics["dataset"] = dataset
    metrics["model_type"] = "optimized"
    return metrics


class HierarchicalStackingClassifier:
    """Era-aware hierarchical stacking: per-layer LR + meta LR."""

    def __init__(self, layer_feature_indices: dict[str, list[int]], n_classes: int):
        self.layer_indices = layer_feature_indices
        self.n_classes = n_classes
        self.layer_models: dict[str, LogisticRegression] = {}
        self.meta_model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalStackingClassifier":
        meta_features = []
        for layer, idx in self.layer_indices.items():
            if not idx:
                continue
            lr = LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=RANDOM_STATE
            )
            lr.fit(X[:, idx], y)
            self.layer_models[layer] = lr
            proba = lr.predict_proba(X[:, idx])
            meta_features.append(proba)
        meta_X = np.hstack(meta_features)
        self.meta_model.fit(meta_X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = []
        for layer, lr in self.layer_models.items():
            idx = self.layer_indices[layer]
            proba = lr.predict_proba(X[:, idx])
            meta_features.append(proba)
        meta_X = np.hstack(meta_features)
        return self.meta_model.predict(meta_X)


def _layer_indices(feature_names: list[str], layers: dict[str, list[str]]) -> dict[str, list[int]]:
    """Map layer feature names to column indices (fuzzy match for CIC spaces)."""
    indices: dict[str, list[int]] = {}
    normalized = {f.strip().lower(): i for i, f in enumerate(feature_names)}
    for layer, feats in layers.items():
        idx_list = []
        for f in feats:
            key = f.strip().lower()
            if key in normalized:
                idx_list.append(normalized[key])
            else:
                for nk, i in normalized.items():
                    if key in nk or nk in key:
                        idx_list.append(i)
                        break
        indices[layer] = sorted(set(idx_list))
    return indices


def train_hierarchical_stack(
    data: PreprocessResult,
    dataset: str,
) -> dict[str, Any]:
    """Train era-aware hierarchical stacking model."""
    layers = ERA_AWARE_LAYERS_KDD if dataset == "kdd" else ERA_AWARE_LAYERS_CIC
    layer_idx = _layer_indices(data.feature_names, layers)
    # Filter empty layers
    layer_idx = {k: v for k, v in layer_idx.items() if v}

    model = HierarchicalStackingClassifier(layer_idx, len(data.label_encoder.classes_))
    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(data.X_test)
    metrics = compute_metrics(
        data.y_test, y_pred, label_names=list(data.label_encoder.classes_)
    )
    metrics["latency_ms_per_sample"] = measure_latency_ms(model, data.X_test)
    metrics["train_time_sec"] = train_time
    metrics["model"] = "hierarchical_stack"
    metrics["dataset"] = dataset
    metrics["model_type"] = "innovation"
    metrics["layers_used"] = list(layer_idx.keys())
    return metrics


def cross_dataset_transfer_experiment(
    kdd_data: PreprocessResult,
    cic_data: PreprocessResult,
) -> dict[str, Any]:
    """
    Lightweight transfer: train RF on KDD-aligned feature semantics.
    Reports proxy-feature subset performance on CIC (illustrative).
    """
    # Use first N overlapping semantic dimensions available in both processed sets
    n_align = min(6, kdd_data.X_train.shape[1], cic_data.X_train.shape[1])
    X_kdd = kdd_data.X_train[:, :n_align]
    X_cic_test = cic_data.X_test[:, :n_align]

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_kdd, kdd_data.y_train)
    y_pred = model.predict(X_cic_test)
    # Note: label spaces differ; this is a demonstration of feature-space shift
    metrics = {
        "experiment": "cross_dataset_proxy",
        "note": "Label spaces differ between KDD and CIC; metrics are illustrative only.",
        "n_aligned_features": n_align,
        "cic_test_accuracy_vs_true": float((y_pred == cic_data.y_test).mean()),
    }
    return metrics
