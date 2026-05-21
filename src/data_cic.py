"""CIC-IDS-2017 download (UNB zip), extract, chunked load, and stratified sampling."""

from __future__ import annotations

import json
import shutil
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    CIC_CHUNK_SIZE,
    CIC_EXTRACT_DIR_NAME,
    CIC_MISSING_COL_THRESHOLD,
    CIC_PER_CLASS_CAP,
    CIC_SAMPLE_SIZE,
    CIC_ZIP_FILENAME,
    CIC_ZIP_URLS,
    RANDOM_STATE,
)

# Browser-like UA — some mirrors reject default urllib UA
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CIC-IDS-2017-research/1.0)",
}


def _download_url(url: str, dest: Path, timeout: int = 600) -> None:
    """Download file with urllib and progress logging."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_DOWNLOAD_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * chunk_size) < chunk_size:
                    pct = 100 * downloaded / total
                    print(f"  ... {downloaded // (1024*1024)} MB ({pct:.0f}%)")


def download_cic_zip(data_dir: Path | None = None) -> Path:
    """
    Download MachineLearningCSV.zip from UNB CIC mirror.

    The old AWS S3 per-file URLs return 404; the official distribution is a
    single zip archive (~224 MB).
    """
    data_dir = data_dir or (CACHE_DIR / "cic_raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / CIC_ZIP_FILENAME

    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        print(f"Using cached zip: {zip_path}")
        return zip_path

    last_err: Exception | None = None
    for url in CIC_ZIP_URLS:
        try:
            print(f"Downloading CIC zip from {url} ...")
            _download_url(url, zip_path)
            if zip_path.stat().st_size > 1_000_000:
                print(f"Saved {zip_path} ({zip_path.stat().st_size // (1024*1024)} MB)")
                return zip_path
        except Exception as e:
            last_err = e
            print(f"Failed: {e}")
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)

    raise FileNotFoundError(
        "Could not download CIC-IDS-2017 MachineLearningCSV.zip. "
        f"Tried: {CIC_ZIP_URLS}. Last error: {last_err}. "
        "Manual fix: download the zip from https://www.unb.ca/cic/datasets/ids-2017.html "
        f"and place it at {zip_path}"
    ) from last_err


def extract_cic_zip(zip_path: Path, data_dir: Path | None = None) -> Path:
    """Extract zip and return directory containing CSV files."""
    data_dir = data_dir or zip_path.parent
    extract_root = data_dir / CIC_EXTRACT_DIR_NAME
    marker = extract_root / ".extracted_ok"

    if marker.exists():
        csvs = list(extract_root.rglob("*.csv"))
        if csvs:
            return extract_root

    print(f"Extracting {zip_path.name} ...")
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    csvs = list(extract_root.rglob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found after extracting {zip_path}")

    marker.write_text(json.dumps([str(p) for p in csvs[:20]], indent=2))
    print(f"Extracted {len(csvs)} CSV file(s) under {extract_root}")
    return extract_root


def discover_cic_csv_files(extract_root: Path) -> list[Path]:
    """Find all day CSVs inside extracted tree (skip tiny/merged summary files)."""
    csvs = sorted(extract_root.rglob("*.csv"))
    # Prefer per-day ISCX files; skip very small files (< 1 MB)
    day_csvs = [p for p in csvs if p.stat().st_size > 1_000_000]
    if not day_csvs:
        day_csvs = csvs
    return day_csvs


def download_cic_files(
    data_dir: Path | None = None,
    files: list[str] | None = None,
    use_minimal_on_fail: bool = True,
) -> list[Path]:
    """
    Download and extract CIC-IDS-2017, return list of CSV paths.

    ``files`` is ignored (kept for API compat) — all CSVs in the zip are used.
    """
    data_dir = data_dir or (CACHE_DIR / "cic_raw")
    zip_path = download_cic_zip(data_dir)
    extract_root = extract_cic_zip(zip_path, data_dir)
    discovered = discover_cic_csv_files(extract_root)

    if files:
        # Optional filter by basename (case-insensitive)
        want = {f.lower() for f in files}
        filtered = [p for p in discovered if p.name.lower() in want]
        if filtered:
            discovered = filtered

    if not discovered:
        raise FileNotFoundError(f"No CSV files under {extract_root}")

    print("CIC CSV files:", [p.name for p in discovered])
    return discovered


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

    if csv_paths is None:
        csv_paths = download_cic_files()
    if not csv_paths:
        raise FileNotFoundError(
            "No CIC CSV files available. Download MachineLearningCSV.zip on Colab with network."
        )

    collected: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            print(f"Reading {path.name} ...")
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

    miss_rate = full.isna().mean()
    drop_cols = miss_rate[miss_rate > CIC_MISSING_COL_THRESHOLD].index.tolist()
    drop_cols = [c for c in drop_cols if c != "Label"]
    full = full.drop(columns=drop_cols, errors="ignore")

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
        "download_source": "UNB MachineLearningCSV.zip",
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return sampled, meta


def get_cic_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Label"]
