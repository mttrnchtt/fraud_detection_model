#!/usr/bin/env python3
"""
Usage:
    python scripts/eval_stack.py [-c configs/stack.yaml]

Loads the frozen stacking meta-models persisted by ``scripts/train_stack.py`` and
reports their TEST metrics. The option is chosen by validation AUPRC in
train_stack.py, never here. For the selected option it also calibrates the scores
and picks a decision threshold on validation, then reports the operating point on
test (precision, recall, F1, precision@k, and Amount-weighted savings).

Results are written to ``reports/meta_model.txt`` and folded back into each
``outputs/stack_<option>/training_results.json``. Run ``scripts/train_stack.py`` first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from fd.calibration import apply_calibrator, fit_calibrator
from fd.cost_metrics import precision_at_k, savings_score, select_threshold
from fd.data_prep.utils import load_config
from fd.stack_helpers.meta import build_meta_features, load_meta, score_metrics, stack_dir
from fd.stack_helpers.utils import OPTIONS, load_npy_data, load_npz_data

MODELS_ROOT = "models"
OUTPUTS_ROOT = "outputs"


def load_test_amount(processed_dir, X_test):
    """Recover the raw transaction Amount for test rows by inverting the saved scaler."""
    man = json.loads((Path(processed_dir) / "manifest.json").read_text())
    idx = man["features"].index("Amount")
    s = np.load(Path(processed_dir) / "scaler.npz")
    return np.expm1(X_test[:, idx] * float(s["sigma_amount"]) + float(s["mu_amount"]))


def analyze_operating_point(eval_cfg, meta, option_cfg, X_val, y_val, X_test, y_test, amount):
    """Calibrate on validation, pick a threshold on validation, report test metrics at it."""
    proba_val = meta.predict_proba(build_meta_features(option_cfg, X_val))[:, 1]
    proba_test = meta.predict_proba(build_meta_features(option_cfg, X_test))[:, 1]

    method = eval_cfg.get("calibration", "isotonic")
    calibrator = fit_calibrator(proba_val, y_val, method=method)
    cal_val = apply_calibrator(calibrator, proba_val)
    cal_test = apply_calibrator(calibrator, proba_test)

    thr_method = eval_cfg.get("threshold_method", "max_f1")
    thr = select_threshold(
        y_val, cal_val, method=thr_method,
        target_precision=eval_cfg.get("target_precision", 0.9),
        alert_rate=eval_cfg.get("alert_rate", 0.01),
    )
    y_pred = (cal_test >= thr).astype(int)

    k = eval_cfg.get("precision_at_k", 0.005)
    review_cost = float(eval_cfg.get("review_cost", 3.0))
    return {
        "calibration": method,
        "threshold_method": thr_method,
        "threshold": float(thr),
        "brier_raw": float(brier_score_loss(y_test, proba_test)),
        "brier_calibrated": float(brier_score_loss(y_test, cal_test)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision_at_k": {"k": k, "value": precision_at_k(y_test, cal_test, k)},
        "savings": {"review_cost": review_cost,
                    "value": savings_score(y_test, y_pred, amount, review_cost)},
        "flagged": int(y_pred.sum()),
        "n_test": int(len(y_test)),
    }


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

        # Fold test metrics back into the per-option artifact.
        out_path = Path(OUTPUTS_ROOT) / f"stack_{name}" / "training_results.json"
        record = json.loads(out_path.read_text()) if out_path.exists() else {"option": name}
        record.setdefault("metrics", {})["test"] = test_metrics
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))

        val_auprc = record.get("metrics", {}).get("val", {}).get("auprc")
        results.append((name, list(option_cfg.keys()), test_metrics, val_auprc))

    if not results:
        raise SystemExit("No persisted stack models found. Run: python scripts/train_stack.py")

    # Order and select by VALIDATION AUPRC (from train_stack.py), never by test.
    results.sort(key=lambda r: (r[3] if r[3] is not None else float("-inf")), reverse=True)
    selected = results[0]

    X_val = load_npz_data(paths["train_features"])
    y_val = load_npy_data(paths["train_labels"]).astype(int)
    amount = load_test_amount(paths.get("processed_dir", "data_processed"), X_test)
    meta, option_cfg = load_meta(MODELS_ROOT, selected[0])
    op = analyze_operating_point(cfg.get("evaluation", {}), meta, option_cfg,
                                 X_val, y_val, X_test, y_test, amount)

    sel_path = Path(OUTPUTS_ROOT) / f"stack_{selected[0]}" / "training_results.json"
    rec = json.loads(sel_path.read_text())
    rec.setdefault("metrics", {})["operating_point"] = op
    sel_path.write_text(json.dumps(rec, indent=2))

    report_path = Path(cfg["paths"].get("report", "reports/meta_model.txt"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as r:
        r.write("Stacking meta-model (LogisticRegression). Option chosen by validation AUPRC\n")
        r.write("(train_stack.py). Test metrics below are reported after selection.\n\n")
        for name, members, m, val_auprc in results:
            v = f"{val_auprc:.6f}" if val_auprc is not None else "n/a"
            mark = "  <- selected on validation" if name == selected[0] else ""
            block = (
                f"===== META MODEL LOGREG {name} ====={mark}\n"
                f"Models: {members}\n"
                f"val AUPRC:  {v}\n"
                f"test AUPRC: {m['auprc']:.6f}\n"
                f"test AUROC: {m['auroc']:.6f}\n\n"
            )
            print(block, end="")
            r.write(block)
        r.write(f"Selected on validation: {selected[0]} "
                f"(test AUPRC={selected[2]['auprc']:.6f}, AUROC={selected[2]['auroc']:.6f})\n\n")

        op_lines = (
            "Operating point for the selected option. Calibration and threshold are fit\n"
            "on validation, metrics are on test.\n"
            f"  calibration={op['calibration']} threshold_method={op['threshold_method']} "
            f"threshold={op['threshold']:.4f}\n"
            f"  test precision={op['precision']:.4f} recall={op['recall']:.4f} "
            f"f1={op['f1']:.4f} flagged={op['flagged']}/{op['n_test']}\n"
            f"  Brier raw={op['brier_raw']:.5f} calibrated={op['brier_calibrated']:.5f}\n"
            f"  precision@{op['precision_at_k']['k']}={op['precision_at_k']['value']:.4f}\n"
            f"  savings(review_cost={op['savings']['review_cost']})={op['savings']['value']:.4f}\n"
        )
        print(op_lines, end="")
        r.write(op_lines)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/stack.yaml")
    args = parser.parse_args()
    main(args.config)
