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

for d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, CACHE_DIR):
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

# CIC CSV filenames (UNB ISCX naming)
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

# Minimum subset if full download fails
CIC_MINIMAL_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# CIC download mirror (AWS Open Data - common Colab mirror)
CIC_BASE_URL = (
    "https://cse-cic-ids2017.s3.us-east-2.amazonaws.com/MachineLearningCSV"
)
