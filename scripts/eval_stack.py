#!/usr/bin/env python3
"""
Usage:
    python scripts/eval_stack.py [-c configs/stack.yaml]

Loads the frozen stacking meta-models persisted by ``scripts/train_stack.py`` and
scores them on the held-out TEST window **exactly once**. Writes a ranked report
to ``reports/meta_model.txt`` and folds the test metrics back into each
``outputs/stack_<option>/training_results.json``.

Run ``scripts/train_stack.py`` first (it trains on validation and persists the
meta-models). This is the only step that reads the test split.
"""

import argparse
import json
from pathlib import Path

from fd.data_prep.utils import load_config
from fd.stack_helpers.meta import build_meta_features, load_meta, score_metrics, stack_dir
from fd.stack_helpers.utils import OPTIONS, load_npy_data, load_npz_data

MODELS_ROOT = "models"
OUTPUTS_ROOT = "outputs"


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]

    X_test = load_npz_data(paths["test_features"])
    y_test = load_npy_data(paths["test_labels"]).astype(int)

    results = []
    for name in OPTIONS:
        if not (stack_dir(MODELS_ROOT, name) / "model.joblib").exists():
            print(f"[skip] {name}: no persisted meta-model (run train_stack.py first)")
            continue
        meta, option_cfg = load_meta(MODELS_ROOT, name)
        X_meta_test = build_meta_features(option_cfg, X_test)
        test_metrics = score_metrics(y_test, meta.predict_proba(X_meta_test)[:, 1])
        results.append((name, list(option_cfg.keys()), test_metrics))

        # Fold test metrics back into the per-option artifact.
        out_path = Path(OUTPUTS_ROOT) / f"stack_{name}" / "training_results.json"
        record = json.loads(out_path.read_text()) if out_path.exists() else {"option": name}
        record.setdefault("metrics", {})["test"] = test_metrics
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))

    if not results:
        raise SystemExit("No persisted stack models found. Run: python scripts/train_stack.py")

    results.sort(key=lambda r: r[2]["auprc"], reverse=True)

    report_path = Path(cfg["paths"].get("report", "reports/meta_model.txt"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as r:
        r.write("Stacking meta-model (LogisticRegression), TEST metrics, scored once.\n")
        r.write("Layer-0 members selected by validation AUPRC (see reports/layer0.txt).\n\n")
        for name, members, m in results:
            block = (
                f"===== META MODEL LOGREG {name} =====\n"
                f"Models: {members}\n"
                f"AUPRC: {m['auprc']:.6f}\n"
                f"AUROC: {m['auroc']:.6f}\n\n"
            )
            print(block, end="")
            r.write(block)
        best = results[0]
        r.write(f"Best by test AUPRC: {best[0]} (AUPRC={best[2]['auprc']:.6f}, AUROC={best[2]['auroc']:.6f})\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/stack.yaml")
    args = parser.parse_args()
    main(args.config)
