#!/usr/bin/env python3
"""Create tiny CIC-shaped fallback parquet (last-resort offline demo)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BUNDLED_DIR, CIC_BUNDLED_SAMPLE, RANDOM_STATE

LABELS = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack",
]


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 8000
    n_feat = 40
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = rng.choice(LABELS, size=n, p=[0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])

    cols = [f"feat_{i}" for i in range(n_feat)]
    df = pd.DataFrame(X, columns=cols)
    df["Label"] = y

    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CIC_BUNDLED_SAMPLE, index=False)
    print(f"Wrote {CIC_BUNDLED_SAMPLE} ({CIC_BUNDLED_SAMPLE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
