"""Matplotlib font setup for Colab/Linux (CJK) with English fallback."""

from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache

logger = logging.getLogger(__name__)

# English fallbacks for plot text
_LABELS = {
    "class": ("类别", "Class"),
    "count_log": ("样本数 (log)", "Count (log scale)"),
    "true": ("真实", "True label"),
    "pred": ("预测", "Predicted"),
    "kdd_class_title": ("KDD Cup 1999 攻击类别分布", "KDD Cup 1999 — Class Distribution"),
    "kdd_pie_title": ("KDD 41维特征分组占比", "KDD — Feature Group Share"),
    "cic_class_title": ("CIC-IDS-2017 攻击类别分布", "CIC-IDS-2017 — Class Distribution"),
    "cic_pie_title": ("CIC 特征分组占比", "CIC — Feature Group Share"),
}

_CJK_FONTS = [
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Arial Unicode MS",
]


@lru_cache(maxsize=1)
def cjk_font_available() -> bool:
    """Return True if a CJK-capable font is registered."""
    try:
        from matplotlib import font_manager

        names = {f.name for f in font_manager.fontManager.ttflist}
        return any(f in names for f in _CJK_FONTS)
    except Exception:
        return False


def _install_noto_colab() -> bool:
    """Install Noto CJK on Debian/Colab (no-op elsewhere)."""
    if os.environ.get("CICIDS_SKIP_FONT_INSTALL", "").lower() in ("1", "true", "yes"):
        return False
    if not os.path.exists("/usr/bin/apt-get"):
        return False
    try:
        import subprocess

        subprocess.run(
            ["apt-get", "-qq", "update"],
            check=False,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            ["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"],
            check=False,
            capture_output=True,
            timeout=180,
        )
        from matplotlib import font_manager

        font_manager._load_fontmanager(try_read_cache=False)
        return cjk_font_available()
    except Exception as e:
        logger.debug("Noto install skipped: %s", e)
        return False


@lru_cache(maxsize=1)
def setup_plot_font(force_english: bool = False) -> bool:
    """
    Configure matplotlib for Chinese labels on Colab.

    Returns True if CJK font is active, False if using English fallback.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if force_english:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return False

    if not cjk_font_available():
        if _install_noto_colab():
            cjk_font_available.cache_clear()

    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _CJK_FONTS if f in names), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        warnings.filterwarnings(
            "ignore",
            message="Glyph.*missing from font",
            category=UserWarning,
        )
        return True

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


def plot_text(key: str) -> str:
    """Return Chinese or English label depending on font availability."""
    zh, en = _LABELS[key]
    if setup_plot_font():
        return zh
    return en
