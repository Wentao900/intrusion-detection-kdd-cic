"""Global experiment configuration for Colab IDS project."""
from __future__ import annotations

import os
from pathlib import Path

# Reproducibility
RANDOM_STATE = 42

# Quick debug mode: 1/5 sample for fast Colab iteration
QUICK_RUN = os.environ.get("QUICK_RUN", "false").lower() in ("1", "true", "yes")

# Paths (Colab: override PROJECT_ROOT via env or Drive mount)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = DATA_DIR / "cache"
BUNDLED_DIR = DATA_DIR / "bundled"

for d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, CACHE_DIR, BUNDLED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# KDD
KDD_SUBSET_10_PERCENT = True
KDD_REMOVE_DUPLICATES = True

# CIC-IDS-2017
CIC_SAMPLE_SIZE = 50_000 if QUICK_RUN else 400_000
CIC_PER_CLASS_CAP = 25_000 if QUICK_RUN else 80_000
CIC_CHUNK_SIZE = 100_000
CIC_MI_TOP_K = 30 if QUICK_RUN else 50
CIC_MISSING_COL_THRESHOLD = 0.30

# CIC download strategy (Colab 推荐 hf_minimal，无需本机、无需 224MB zip)
# auto | hf_minimal | hf_one | zip | kaggle | drive | bundled
CIC_DOWNLOAD_MODE = os.environ.get("CIC_DOWNLOAD_MODE", "auto")

# Hugging Face: per-day parquet (~15–22 MB each), not full zip
CIC_HF_REPO = "bvsam/cic-ids-2017"
CIC_HF_SUBDIR = "machine_learning"
# Smallest day files first (bytes from HF API, ~17–22 MB each)
CIC_HF_MINIMAL_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
]
CIC_HF_ONE_FILE = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"

# Google Drive: 手动上传 parquet/zip 后的路径（Colab 挂载 Drive 后）
CIC_DRIVE_CANDIDATE_PATHS = [
    "MyDrive/CIC-IDS-2017/cic_sample.parquet",
    "MyDrive/CIC-IDS-2017/MachineLearningCSV.zip",
    "MyDrive/cic_sample.parquet",
]

# Kaggle dataset slug (download on Colab with API token)
CIC_KAGGLE_DATASET = "sweety18/cicids2017-full-modified-all-8-files"

# UNB zip fallback (~224 MB)
CIC_ZIP_URLS = [
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
]
CIC_ZIP_FILENAME = "MachineLearningCSV.zip"
CIC_EXTRACT_DIR_NAME = "MachineLearningCSV"

# Repo-bundled fallback (~2 MB synthetic structure) — 仅当所有下载失败
CIC_BUNDLED_SAMPLE = BUNDLED_DIR / "cic_fallback_sample.parquet"

# Train/val/test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature selection
VARIANCE_THRESHOLD = 0.01
CORR_THRESHOLD = 0.95

# Models
RF_N_ESTIMATORS = 100 if QUICK_RUN else 200
CNN_EPOCHS = 5 if QUICK_RUN else 20
CNN_BATCH_SIZE = 256
CNN_PATIENCE = 3

# Metrics
LATENCY_SAMPLE_SIZE = 5_000 if QUICK_RUN else 10_000

CIC_CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

CIC_BASE_URL = ""
