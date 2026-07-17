#!/usr/bin/env python3
"""
Usage:
    python scripts/train_stack.py [-c configs/stack.yaml]

Trains the stacking meta-model (LogisticRegression) for every layer-0 option on
the VALIDATION window and persists each fitted meta-model to
``models/stack_<option>/model.joblib`` together with per-option validation
metrics in ``outputs/stack_<option>/training_results.json``.

This script never touches the test split. Selection between options is done on
validation. scripts/eval_stack.py scores the frozen meta-models on test exactly
once for the reported numbers.
"""

import argparse
from pathlib import Path

from fd.data_prep.utils import load_config
from fd.stack_helpers.meta import (
    build_meta_features,
    save_meta,
    score_metrics,
    train_meta_logreg,
    write_json,
)
from fd.stack_helpers.utils import (
    OPTIONS,
    assert_meta_source_is_holdout,
    load_npy_data,
    load_npz_data,
)

MODELS_ROOT = "models"
OUTPUTS_ROOT = "outputs"


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]

    # Anti-leakage: the meta-model learns from layer-0 scores on the held-out
    # validation window, which the base learners never saw.
    assert_meta_source_is_holdout(paths["train_features"])

    X_val = load_npz_data(paths["train_features"])
    y_val = load_npy_data(paths["train_labels"]).astype(int)
    meta_cfg = cfg.get("meta", {})

    ranking = []
    for name, option_cfg in OPTIONS.items():
        X_meta_val = build_meta_features(option_cfg, X_val)
        meta = train_meta_logreg(X_meta_val, y_val, meta_cfg)

        val_metrics = score_metrics(y_val, meta.predict_proba(X_meta_val)[:, 1])
        model_path = save_meta(MODELS_ROOT, name, meta, option_cfg)
        write_json(
            Path(OUTPUTS_ROOT) / f"stack_{name}" / "training_results.json",
            {
                "option": name,
                "members": list(option_cfg.keys()),
                "meta_source": paths["train_features"],
                "metrics": {"val": val_metrics},
                "note": "Validation metrics only; test is scored once by eval_stack.py.",
            },
        )
        ranking.append((name, val_metrics["auprc"], val_metrics["auroc"]))
        print(f"[{name}] val AUPRC={val_metrics['auprc']:.6f} AUROC={val_metrics['auroc']:.6f} "
              f"-> {model_path}")

    ranking.sort(key=lambda r: r[1], reverse=True)
    best = ranking[0][0]

    report_dir = Path(paths.get("report", "reports/meta_model.txt")).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "stack_selection.txt", "w") as f:
        f.write("Stacking option ranking by VALIDATION AUPRC. Selection basis, never test.\n\n")
        for name, auprc, auroc in ranking:
            f.write(f"{name:10s} val AUPRC={auprc:.6f} val AUROC={auroc:.6f}\n")
        f.write(f"\nSelected on validation: {best}\n")

    print("\nValidation ranking by AUPRC:")
    for name, auprc, auroc in ranking:
        print(f"  {name:10s} AUPRC={auprc:.6f} AUROC={auroc:.6f}")
    print(f"\nBest option on validation: {best}")
    print("Run: python scripts/eval_stack.py  (scores the frozen meta-models on test)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/stack.yaml")
    args = parser.parse_args()
    main(args.config)
