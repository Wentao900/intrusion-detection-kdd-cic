"""CIC-IDS-2017 chunked download, load, and stratified sampling."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    CIC_BASE_URL,
    CIC_CHUNK_SIZE,
    CIC_CSV_FILES,
    CIC_MINIMAL_FILES,
    CIC_MISSING_COL_THRESHOLD,
    CIC_PER_CLASS_CAP,
    CIC_SAMPLE_SIZE,
    RANDOM_STATE,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and labels."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    return df


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, drop empty cols, normalize labels."""
    chunk = _normalize_columns(chunk)
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(axis=1, how="all")
    if "Label" not in chunk.columns:
        for c in chunk.columns:
            if c.lower() == "label":
                chunk = chunk.rename(columns={c: "Label"})
                break
    return chunk


def download_cic_files(
    data_dir: Path | None = None,
    files: list[str] | None = None,
    use_minimal_on_fail: bool = True,
) -> list[Path]:
    """Download CIC CSV files from AWS mirror."""
    data_dir = data_dir or (CACHE_DIR / "cic_raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    files = files or CIC_CSV_FILES
    downloaded: list[Path] = []

    for fname in files:
        dest = data_dir / fname
        if dest.exists() and dest.stat().st_size > 1000:
            downloaded.append(dest)
            continue
        url = f"{CIC_BASE_URL}/{fname}"
        try:
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, dest)
            downloaded.append(dest)
        except Exception as e:
            print(f"Failed {fname}: {e}")

    if not downloaded and use_minimal_on_fail:
        return download_cic_files(data_dir, CIC_MINIMAL_FILES, use_minimal_on_fail=False)
    return downloaded


def _stratified_sample_indices(labels: pd.Series, target_size: int, per_class_cap: int) -> np.ndarray:
    """Stratified sampling with per-class cap."""
    rng = np.random.default_rng(RANDOM_STATE)
    indices: list[int] = []
    classes = labels.value_counts()
    n_classes = len(classes)
    base_per_class = max(target_size // max(n_classes, 1), 100)

    for cls, count in classes.items():
        cls_idx = labels[labels == cls].index.to_numpy()
        cap = min(per_class_cap, base_per_class, len(cls_idx))
        if count < cap:
            cap = len(cls_idx)
        chosen = rng.choice(cls_idx, size=cap, replace=False)
        indices.extend(chosen.tolist())

    indices = np.array(indices)
    if len(indices) > target_size:
        indices = rng.choice(indices, size=target_size, replace=False)
    return indices


def load_cic_chunked(
    csv_paths: list[Path] | None = None,
    sample_size: int | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load CIC with chunked read and stratified sampling."""
    sample_size = sample_size or CIC_SAMPLE_SIZE
    cache_path = CACHE_DIR / f"cic_sample_{sample_size}.parquet"
    meta_path = CACHE_DIR / "cic_meta.json"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return df, meta

    data_dir = CACHE_DIR / "cic_raw"
    if csv_paths is None:
        csv_paths = download_cic_files(data_dir)
    if not csv_paths:
        raise FileNotFoundError("No CIC CSV files available. Run download on Colab with network.")

    collected: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            for chunk in pd.read_csv(path, chunksize=CIC_CHUNK_SIZE, low_memory=False):
                chunk = _clean_chunk(chunk)
                if "Label" in chunk.columns:
                    collected.append(chunk)
                if sum(len(c) for c in collected) > sample_size * 3:
                    break
        except Exception as e:
            print(f"Skip {path.name}: {e}")

    if not collected:
        raise RuntimeError("Failed to read any CIC data chunks.")

    full = pd.concat(collected, ignore_index=True)
    full = _normalize_columns(full)

    # Drop high-missing columns
    miss_rate = full.isna().mean()
    drop_cols = miss_rate[miss_rate > CIC_MISSING_COL_THRESHOLD].index.tolist()
    drop_cols = [c for c in drop_cols if c != "Label"]
    full = full.drop(columns=drop_cols, errors="ignore")

    # Numeric downcast
    for col in full.select_dtypes(include=[np.number]).columns:
        full[col] = pd.to_numeric(full[col], downcast="float")

    if "Label" not in full.columns:
        raise ValueError("CIC data missing Label column")

    idx = _stratified_sample_indices(full["Label"], sample_size, CIC_PER_CLASS_CAP)
    sampled = full.iloc[idx].reset_index(drop=True)

    meta = {
        "n_samples_sampled": len(sampled),
        "n_features": len([c for c in sampled.columns if c != "Label"]),
        "files_used": [p.name for p in csv_paths],
        "class_distribution": sampled["Label"].value_counts().to_dict(),
        "dropped_high_missing_cols": drop_cols,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return sampled, meta


def get_cic_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Label"]
