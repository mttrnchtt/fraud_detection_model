#!/usr/bin/env python3
"""
Usage:
    python scripts/select_layer0.py [-c configs/stack.yaml]

Ranks every trained base model by **validation** Average Precision (AUPRC) and
writes ``reports/layer0.txt``. This is the honest basis for choosing which
learners go into a layer-0 option: selection is done on validation, never on the
test split. Each model is loaded from ``models/<name>/model.joblib`` and scored
on ``X_val`` via the same score extraction used at stacking time.
"""

import argparse
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from fd.data_prep.utils import load_config
from fd.stack_helpers.utils import load_npy_data, load_npz_data, run_layer0

MODELS_ROOT = Path("models")


def is_base_model(name: str) -> bool:
    return not (name.startswith("stack_") or name.startswith("meta_"))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    X_val = load_npz_data(paths["train_features"])  # stack.yaml train_features == validation split
    y_val = load_npy_data(paths["train_labels"]).astype(int)

    rows = []
    for d in sorted(MODELS_ROOT.iterdir() if MODELS_ROOT.exists() else []):
        if not d.is_dir() or not is_base_model(d.name):
            continue
        model_path = d / "model.joblib"
        if not model_path.exists():
            continue
        scores = run_layer0({d.name: str(model_path)}, X_val)[:, 0]
        rows.append((d.name, float(average_precision_score(y_val, scores)),
                     float(roc_auc_score(y_val, scores))))

    if not rows:
        raise SystemExit("No base models under models/. Train the base learners first.")

    rows.sort(key=lambda r: r[1], reverse=True)
    report = Path(paths.get("report", "reports/meta_model.txt")).parent / "layer0.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        f.write("Base-learner rankings by Average Precision (VALIDATION set).\n")
        f.write("Selection basis for layer-0 options, never ranked on test.\n\n")
        for i, (name, ap, auroc) in enumerate(rows, 1):
            line = f"{i:2d}. {name:22s} AUPRC={ap:.4f}  AUROC={auroc:.4f}\n"
            f.write(line)
            print(line, end="")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/stack.yaml")
    args = parser.parse_args()
    main(args.config)
