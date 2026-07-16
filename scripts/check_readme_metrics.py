#!/usr/bin/env python3
"""
Usage:
    python scripts/check_readme_metrics.py

Asserts that every headline AUROC/AUPRC in the README results table matches the
committed backing artifact under ``outputs/`` (rounded to 4 decimals). Fails with
a clear diff if the README and the artifacts have drifted apart. Also importable
as ``check_readme_metrics()`` for the test suite.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# README row label -> outputs/<dir> whose test metrics must back it.
EXPECTED = {
    "Stacking, meta-MLP": "meta_mlp",
    "Stacking, meta-LR": "stack_B_wLGBM",
    "XGBoost, weighted (w=5)": "xgb_weighted_w5",
    "Balanced Random Forest (s=1.0)": "rf_balanced_s1.0",
    "LightGBM": "lightgbm",
    "MLP": "mlp",
    "Logistic Regression, weighted": "lr_weighted",
    "Isolation Forest": "isolation_forest",
}


def _test_metrics(outputs_dir: Path) -> dict:
    data = json.loads((outputs_dir / "training_results.json").read_text())
    metrics = data.get("metrics", data)
    return metrics["test"]


def check_readme_metrics(readme: Path | None = None, outputs_root: Path | None = None) -> list[str]:
    readme = readme or (ROOT / "README.md")
    outputs_root = outputs_root or (ROOT / "outputs")
    text = readme.read_text()
    problems = []
    for label, subdir in EXPECTED.items():
        d = outputs_root / subdir
        if not (d / "training_results.json").exists():
            problems.append(f"{label}: missing artifact {d}/training_results.json (run the pipeline)")
            continue
        t = _test_metrics(d)
        for key in ("auroc", "auprc"):
            want = f"{t[key]:.4f}"
            if want not in text:
                problems.append(f"{label}: README is missing {key.upper()}={want} from outputs/{subdir}")
    return problems


def main() -> int:
    problems = check_readme_metrics()
    if problems:
        print("README metrics do NOT match committed artifacts:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: all {len(EXPECTED)} README result rows match their outputs/ artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
