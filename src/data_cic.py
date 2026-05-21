"""CIC-IDS-2017: multi-source load (HF shards / Drive / Kaggle / zip / bundled)."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BUNDLED_DIR,
    CACHE_DIR,
    CIC_BUNDLED_SAMPLE,
    CIC_CHUNK_SIZE,
    CIC_DOWNLOAD_MODE,
    CIC_DRIVE_CANDIDATE_PATHS,
    CIC_EXTRACT_DIR_NAME,
    CIC_HF_MINIMAL_FILES,
    CIC_HF_ONE_FILE,
    CIC_HF_REPO,
    CIC_HF_SUBDIR,
    CIC_KAGGLE_DATASET,
    CIC_MISSING_COL_THRESHOLD,
    CIC_PER_CLASS_CAP,
    CIC_SAMPLE_SIZE,
    CIC_ZIP_FILENAME,
    CIC_ZIP_URLS,
    QUICK_RUN,
    RANDOM_STATE,
)

_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CIC-IDS-2017-research/1.0)",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    return df


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
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
    rng = np.random.default_rng(RANDOM_STATE)
    indices: list[int] = []
    classes = labels.value_counts()
    base_per_class = max(target_size // max(len(classes), 1), 100)

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


def _ensure_bundled_fallback() -> Path:
    """Create tiny fallback parquet if missing (no network)."""
    if CIC_BUNDLED_SAMPLE.exists():
        return CIC_BUNDLED_SAMPLE
    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    n, n_feat = 8000, 40
    labels = ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "FTP-Patator", "Web Attack"]
    df = pd.DataFrame(
        rng.standard_normal((n, n_feat)).astype(np.float32),
        columns=[f"feat_{i}" for i in range(n_feat)],
    )
    df["Label"] = rng.choice(labels, size=n, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    df.to_parquet(CIC_BUNDLED_SAMPLE, index=False)
    return CIC_BUNDLED_SAMPLE


def load_cic_from_bundled() -> tuple[pd.DataFrame, dict]:
    path = _ensure_bundled_fallback()
    df = pd.read_parquet(path)
    df = _normalize_columns(df)
    meta = {
        "download_source": "bundled_fallback",
        "warning": "非官方 CIC 数据，仅用于跑通流程；报告请注明限制",
        "n_samples": len(df),
        "class_distribution": df["Label"].value_counts().to_dict(),
    }
    return df, meta


def load_cic_from_drive() -> tuple[pd.DataFrame, dict] | None:
    """Load user-uploaded file from mounted Google Drive."""
    roots = [
        Path("/content/drive"),
        Path.home() / "Google Drive",
        Path.home() / "Library/CloudStorage",
    ]
    for root in roots:
        if not root.exists():
            continue
        for rel in CIC_DRIVE_CANDIDATE_PATHS:
            for candidate in root.rglob(Path(rel).name):
                if not candidate.exists():
                    continue
                print(f"Found Drive file: {candidate}")
                if candidate.suffix == ".parquet":
                    df = _normalize_columns(pd.read_parquet(candidate))
                    return df, {"download_source": "google_drive", "path": str(candidate)}
                if candidate.suffix == ".zip":
                    extract_root = CACHE_DIR / "cic_drive_zip"
                    shutil.rmtree(extract_root, ignore_errors=True)
                    extract_root.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(candidate, "r") as zf:
                        zf.extractall(extract_root)
                    csvs = discover_cic_csv_files(extract_root)
                    return _load_from_paths(csvs, {"download_source": "google_drive_zip"})
    return None


def download_cic_hf_shards(file_list: list[str]) -> list[Path]:
    """Download selected day parquet shards from Hugging Face (~15–22 MB each)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub") from None

    out_dir = CACHE_DIR / "cic_hf"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for fname in file_list:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 100_000:
            paths.append(dest)
            continue
        print(f"HF download: {fname} (~15–22 MB) ...")
        cached = hf_hub_download(
            repo_id=CIC_HF_REPO,
            filename=f"{CIC_HF_SUBDIR}/{fname}",
            repo_type="dataset",
            local_dir=str(out_dir / "hf_cache"),
        )
        cached_path = Path(cached)
        if cached_path.resolve() != dest.resolve():
            shutil.copy2(cached_path, dest)
        paths.append(dest)
        print(f"  OK {dest.name} ({dest.stat().st_size // (1024*1024)} MB)")

    return paths


def _parquet_paths_to_dataframe(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        print(f"Reading {p.name} ...")
        df = pd.read_parquet(p)
        df = _clean_chunk(df)
        if "Label" in df.columns:
            parts.append(df)
    if not parts:
        raise RuntimeError("No valid parquet shards")
    return pd.concat(parts, ignore_index=True)


def download_cic_kaggle(data_dir: Path | None = None) -> list[Path]:
    """Download via Kaggle API (token on Colab only — not local)."""
    data_dir = data_dir or (CACHE_DIR / "cic_kaggle")
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.run(
            [
                "kaggle", "datasets", "download", "-d", CIC_KAGGLE_DATASET,
                "-p", str(data_dir), "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed: {e}. "
            "Upload kaggle.json to Colab: "
            "https://www.kaggle.com/settings → API → Create New Token"
        ) from e
    csvs = discover_cic_csv_files(data_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSV in {data_dir} after Kaggle unzip")
    return csvs


def _download_url(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_DOWN_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def download_cic_zip(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or (CACHE_DIR / "cic_raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / CIC_ZIP_FILENAME
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        return zip_path
    last_err = None
    for url in CIC_ZIP_URLS:
        try:
            print(f"Downloading zip from {url} ...")
            _download_url(url, zip_path)
            return zip_path
        except Exception as e:
            last_err = e
            zip_path.unlink(missing_ok=True)
    raise FileNotFoundError(f"UNB zip failed: {last_err}") from last_err


def extract_cic_zip(zip_path: Path) -> Path:
    extract_root = zip_path.parent / CIC_EXTRACT_DIR_NAME
    marker = extract_root / ".extracted_ok"
    if marker.exists() and list(extract_root.rglob("*.csv")):
        return extract_root
    print(f"Extracting {zip_path.name} ...")
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    marker.write_text("ok")
    return extract_root


def discover_cic_csv_files(root: Path) -> list[Path]:
    csvs = sorted(root.rglob("*.csv"))
    day = [p for p in csvs if p.stat().st_size > 1_000_000]
    return day or csvs


def _load_from_paths(
    paths: list[Path],
    base_meta: dict,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    sample_size = sample_size or CIC_SAMPLE_SIZE
    collected: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix == ".parquet":
            df = _clean_chunk(pd.read_parquet(path))
            if "Label" in df.columns:
                collected.append(df)
            continue
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
        raise RuntimeError("Failed to read CIC files")

    full = pd.concat(collected, ignore_index=True)
    full = _normalize_columns(full)
    miss_rate = full.isna().mean()
    drop_cols = [c for c in miss_rate[miss_rate > CIC_MISSING_COL_THRESHOLD].index if c != "Label"]
    full = full.drop(columns=drop_cols, errors="ignore")

    for col in full.select_dtypes(include=[np.number]).columns:
        full[col] = pd.to_numeric(full[col], downcast="float")

    idx = _stratified_sample_indices(full["Label"], sample_size, CIC_PER_CLASS_CAP)
    sampled = full.iloc[idx].reset_index(drop=True)
    meta = {
        **base_meta,
        "n_samples_sampled": len(sampled),
        "n_features": len([c for c in sampled.columns if c != "Label"]),
        "files_used": [p.name for p in paths],
        "class_distribution": sampled["Label"].value_counts().to_dict(),
        "dropped_high_missing_cols": drop_cols,
    }
    return sampled, meta


def _resolve_mode() -> str:
    mode = CIC_DOWNLOAD_MODE.lower()
    if mode != "auto":
        return mode
    # auto: Colab-friendly order
    if QUICK_RUN:
        return "hf_one"
    return "hf_minimal"


def acquire_cic_raw(sample_size: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Acquire CIC data using configured strategy.

    Recommended for Colab: CIC_DOWNLOAD_MODE=hf_minimal (default auto)
    Downloads 3 day shards ~55 MB total from Hugging Face.
    """
    mode = _resolve_mode()
    errors: list[str] = []

    if mode in ("drive", "auto"):
        try:
            got = load_cic_from_drive()
            if got:
                df, meta = got
                if "Label" in df.columns and len(df) > (sample_size or CIC_SAMPLE_SIZE):
                    idx = _stratified_sample_indices(
                        df["Label"], sample_size or CIC_SAMPLE_SIZE, CIC_PER_CLASS_CAP
                    )
                    df = df.iloc[idx].reset_index(drop=True)
                meta["n_samples_sampled"] = len(df)
                meta["class_distribution"] = df["Label"].value_counts().to_dict()
                return df, meta
        except Exception as e:
            errors.append(f"drive: {e}")

    if mode in ("hf_minimal", "hf_one", "auto"):
        try:
            files = [CIC_HF_ONE_FILE] if mode == "hf_one" else CIC_HF_MINIMAL_FILES
            paths = download_cic_hf_shards(files)
            full = _parquet_paths_to_dataframe(paths)
            idx = _stratified_sample_indices(full["Label"], sample_size or CIC_SAMPLE_SIZE, CIC_PER_CLASS_CAP)
            sampled = full.iloc[idx].reset_index(drop=True)
            return sampled, {
                "download_source": "huggingface_shards",
                "hf_repo": CIC_HF_REPO,
                "files_used": [p.name for p in paths],
                "n_samples_sampled": len(sampled),
                "class_distribution": sampled["Label"].value_counts().to_dict(),
            }
        except Exception as e:
            errors.append(f"hf: {e}")

    if mode in ("kaggle", "auto"):
        try:
            paths = download_cic_kaggle()
            return _load_from_paths(paths, {"download_source": "kaggle"}, sample_size)
        except Exception as e:
            errors.append(f"kaggle: {e}")

    if mode in ("zip", "auto"):
        try:
            zip_path = download_cic_zip()
            root = extract_cic_zip(zip_path)
            paths = discover_cic_csv_files(root)
            return _load_from_paths(paths, {"download_source": "UNB_zip"}, sample_size)
        except Exception as e:
            errors.append(f"zip: {e}")

    if mode in ("bundled", "auto"):
        print("WARNING: 使用仓库内置 fallback 样本（非官方 CIC），报告需注明。")
        df, meta = load_cic_from_bundled()
        if len(df) > (sample_size or CIC_SAMPLE_SIZE):
            idx = _stratified_sample_indices(df["Label"], sample_size or CIC_SAMPLE_SIZE, CIC_PER_CLASS_CAP)
            df = df.iloc[idx].reset_index(drop=True)
        meta["n_samples_sampled"] = len(df)
        return df, meta

    raise RuntimeError(
        "All CIC download methods failed:\n" + "\n".join(errors) +
        "\n\n建议: 在 Colab 设置 os.environ['CIC_DOWNLOAD_MODE']='hf_minimal' 并重试；"
        "或将 cic_sample.parquet 放到 Drive 的 MyDrive/CIC-IDS-2017/"
    )


def download_cic_files(data_dir: Path | None = None, **kwargs) -> list[Path]:
    """API compat: returns paths used (may be HF parquet)."""
    mode = _resolve_mode()
    if mode in ("hf_minimal", "hf_one", "auto"):
        files = [CIC_HF_ONE_FILE] if mode == "hf_one" or QUICK_RUN else CIC_HF_MINIMAL_FILES
        return download_cic_hf_shards(files)
    zip_path = download_cic_zip(data_dir)
    return discover_cic_csv_files(extract_cic_zip(zip_path))


def load_cic_chunked(
    csv_paths: list[Path] | None = None,
    sample_size: int | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load CIC with caching."""
    sample_size = sample_size or CIC_SAMPLE_SIZE
    cache_path = CACHE_DIR / f"cic_sample_{sample_size}_{_resolve_mode()}.parquet"
    meta_path = CACHE_DIR / f"cic_meta_{sample_size}_{_resolve_mode()}.json"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        print(f"Loaded CIC cache: {cache_path}")
        return df, meta

    if csv_paths:
        df, meta = _load_from_paths(csv_paths, {"download_source": "manual_paths"}, sample_size)
    else:
        df, meta = acquire_cic_raw(sample_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Cached CIC to {cache_path}")
    return df, meta


def get_cic_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Label"]
