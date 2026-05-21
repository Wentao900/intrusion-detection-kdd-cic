"""Unified preprocessing: clean, encode, feature selection, scale, split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import (
    CIC_MI_TOP_K,
    CORR_THRESHOLD,
    RANDOM_STATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    VARIANCE_THRESHOLD,
)
from src.data_kdd import get_kdd_feature_columns
from src.data_cic import get_cic_feature_columns


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler | None
    meta: dict


def _encode_kdd_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical KDD columns."""
    X = df[feature_cols].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(X[cat_cols].astype(str))
        enc_names = list(ohe.get_feature_names_out(cat_cols))
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_out = pd.concat(
            [X_num.reset_index(drop=True), pd.DataFrame(encoded, columns=enc_names)],
            axis=1,
        )
        return X_out, list(X_num.columns) + enc_names
    X_out = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_out, feature_cols


def _encode_cic_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Numeric-only CIC matrix with median imputation."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    X = X.fillna(medians).fillna(0)
    return X


def _remove_correlated(X: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    return X[kept], to_drop


def _select_top_mi(X: pd.DataFrame, y: np.ndarray, k: int) -> tuple[pd.DataFrame, list[str]]:
    if X.shape[1] <= k:
        return X, list(X.columns)
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    idx = np.argsort(mi)[::-1][:k]
    cols = [X.columns[i] for i in idx]
    return X[cols], cols


def preprocess_dataset(
    df: pd.DataFrame,
    label_col: str,
    dataset_name: str,
    mi_top_k: int | None = None,
    scale: bool = True,
) -> PreprocessResult:
    """Full preprocessing pipeline for KDD or CIC."""
    mi_top_k = mi_top_k or CIC_MI_TOP_K

    if dataset_name == "kdd":
        feature_cols = get_kdd_feature_columns(df)
        X_df, all_names = _encode_kdd_features(df, feature_cols)
        y_raw = df[label_col].astype(str)
    else:
        feature_cols = get_cic_feature_columns(df)
        X_df = _encode_cic_features(df, feature_cols)
        all_names = list(X_df.columns)
        y_raw = df[label_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Variance filter
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        vt.fit(X_df)
        mask = vt.get_support()
        X_df = X_df.loc[:, mask]
    except ValueError:
        pass

    # Correlation filter
    X_df, dropped_corr = _remove_correlated(X_df)

    # MI feature selection for high-dim (mainly CIC)
    if dataset_name == "cic" and X_df.shape[1] > mi_top_k:
        X_df, selected = _select_top_mi(X_df, y, mi_top_k)
    else:
        selected = list(X_df.columns)

    X = X_df.values.astype(np.float32)

    # Stratified splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), stratify=y, random_state=RANDOM_STATE
    )
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adj), stratify=y_temp, random_state=RANDOM_STATE
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    meta = {
        "dataset": dataset_name,
        "n_features_final": X_train.shape[1],
        "feature_names": selected,
        "dropped_correlated": dropped_corr,
        "classes": list(le.classes_),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }

    return PreprocessResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=selected,
        label_encoder=le,
        scaler=scaler,
        meta=meta,
    )


def imbalance_ratio(y: np.ndarray) -> float:
    """Max class count / min class count."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if len(counts) < 2:
        return 1.0
    return float(counts.max() / counts.min())
