#!/usr/bin/env python3
"""Static logic verification — no third-party imports required."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def check_syntax(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))


def check_import_graph() -> None:
    """Ensure module filenames match expected pipeline."""
    expected = {
        "config.py",
        "features_meta.py",
        "data_kdd.py",
        "data_cic.py",
        "preprocess.py",
        "metrics.py",
        "models_ml.py",
        "models_dl.py",
        "optimize.py",
        "eda.py",
        "run_experiment.py",
        "report_builder.py",
    }
    found = {p.name for p in SRC.glob("*.py")}
    missing = expected - found
    extra = found - expected - {"__init__.py"}
    if missing:
        raise SystemExit(f"Missing modules: {missing}")
    if extra:
        print(f"Note: extra modules: {extra}")


def main() -> int:
    errors = []
    for py in sorted(SRC.glob("*.py")):
        try:
            check_syntax(py)
            print(f"OK  {py.name}")
        except SyntaxError as e:
            errors.append(f"{py.name}: {e}")

    try:
        check_import_graph()
        print("OK  import graph")
    except SystemExit as e:
        errors.append(str(e))

    nb = ROOT / "notebooks" / "IDS_KDD_CIC_Experiment.ipynb"
    if nb.exists():
        import json
        json.loads(nb.read_text(encoding="utf-8"))
        print("OK  notebook JSON")

    if errors:
        for err in errors:
            print(f"FAIL {err}", file=sys.stderr)
        return 1
    print("All static checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
