"""KDD Cup 1999 data loading and label mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

from src.config import CACHE_DIR, KDD_REMOVE_DUPLICATES, KDD_SUBSET_10_PERCENT, RANDOM_STATE
from src.features_meta import KDD_ATTACK_TO_CATEGORY, KDD_CATEGORIES


def load_kdd_raw(use_cache: bool = True) -> pd.DataFrame:
    """Load KDD Cup 1999 via sklearn (10% subset by default)."""
    cache_path = CACHE_DIR / "kdd_raw.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    bunch = fetch_kddcup99(
        subset=None,
        percent10=KDD_SUBSET_10_PERCENT,
        as_frame=True,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    df = bunch.frame.copy()
    df.columns = [str(c).replace(".", "") for c in df.columns]
    target_col = None
    for cand in ("target", "labels", "label"):
        if cand in df.columns:
            target_col = cand
            break
    if target_col:
        df = df.rename(columns={target_col: "label_raw"})
    else:
        df["label_raw"] = bunch.target
    # Decode bytes to str if needed
    if df["label_raw"].dtype == object:
        df["label_raw"] = df["label_raw"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def map_kdd_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine-grained attack names to 4-class + normal."""
    out = df.copy()
    raw = out["label_raw"].astype(str).str.strip().str.lower()
    out["attack_name"] = raw.str.rstrip(".")
    def _to_category(name: str) -> str:
        if name == "normal":
            return "normal"
        return KDD_ATTACK_TO_CATEGORY.get(name, KDD_ATTACK_TO_CATEGORY.get(name + ".", "R2L"))

    out["label_category"] = out["attack_name"].map(_to_category)
    out.loc[out["attack_name"] == "normal", "label_category"] = "normal"
    out["label_binary"] = (out["label_category"] != "normal").astype(int)
    return out


def clean_kdd_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows (known ~78% redundancy in KDD)."""
    if not KDD_REMOVE_DUPLICATES:
        return df
    feature_cols = [c for c in df.columns if c not in ("label_raw", "attack_name", "label_category", "label_binary")]
    before = len(df)
    deduped = df.drop_duplicates(subset=feature_cols)
    deduped = deduped.reset_index(drop=True)
    deduped.attrs["duplicates_removed"] = before - len(deduped)
    return deduped


def get_kdd_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric/categorical feature column names."""
    exclude = {"label_raw", "attack_name", "label_category", "label_binary"}
    return [c for c in df.columns if c not in exclude]


def load_kdd_pipeline(use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full KDD load: raw -> labels -> dedup."""
    cache_path = CACHE_DIR / "kdd_processed.parquet"
    meta_path = CACHE_DIR / "kdd_meta.json"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        meta = {}
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
        return df, meta

    df = load_kdd_raw(use_cache=use_cache)
    df = map_kdd_labels(df)
    n_before = len(df)
    df = clean_kdd_duplicates(df)
    n_after = len(df)

    meta = {
        "n_samples_raw": n_before,
        "n_samples_dedup": n_after,
        "n_features": len(get_kdd_feature_columns(df)),
        "class_distribution": df["label_category"].value_counts().to_dict(),
        "subset": "10-percent" if KDD_SUBSET_10_PERCENT else "full",
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    import json
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return df, meta
