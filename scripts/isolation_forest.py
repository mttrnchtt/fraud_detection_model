#!/usr/bin/env python3
"""
Train the Isolation Forest base model.

Usage:
    python scripts/isolation_forest.py [-c configs/isolation_forest.yaml] [--tune] [--thresholds ...]
"""

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score, roc_auc_score

from fd.common import evaluate_with_thresholds
from fd.mlp_helpers.utils import load_all_data

SKLEARN_KEYS = ("n_estimators", "contamination", "max_samples", "max_features",
                "bootstrap", "n_jobs", "random_state")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_isolation_forest(config, X_train):
    params = {k: v for k, v in config["model"].items() if k in SKLEARN_KEYS}
    model = IsolationForest(**params)
    model.fit(X_train)  # unsupervised, labels are ignored
    return model


def anomaly_scores(model, X):
    # score_samples is higher for inliers, so negate it: higher means more anomalous.
    return -model.score_samples(X)


def tune(config, X_train, X_val, y_val):
    grid = config["tuning"]["param_grid"]
    best_auprc, best_model, best_config = -1.0, None, None
    for params in ParameterSampler(grid, n_iter=config["tuning"]["n_iter"],
                                   random_state=config["experiment"]["seed"]):
        cfg = copy.deepcopy(config)
        cfg["model"].update(params)
        model = train_isolation_forest(cfg, X_train)
        auprc = average_precision_score(y_val, anomaly_scores(model, X_val))
        print(f"  {params}: val AUPRC={auprc:.4f}")
        if auprc > best_auprc:
            best_auprc, best_model, best_config = auprc, model, cfg
    print(f"Best val AUPRC={best_auprc:.4f}")
    return best_model, best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/isolation_forest.yaml")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    args = parser.parse_args()
    config = load_config(args.config)

    X_train, _, X_val, y_val, X_test, y_test = load_all_data(config)
    model, config = (tune(config, X_train, X_val, y_val) if args.tune
                     else (train_isolation_forest(config, X_train), config))

    test_scores = anomaly_scores(model, X_test)
    test_auroc = roc_auc_score(y_test, test_scores)
    test_auprc = average_precision_score(y_test, test_scores)
    print(f"Test AUROC={test_auroc:.4f} AUPRC={test_auprc:.4f}")

    # Map scores to [0, 1] with a range fit on validation (never on test) so the
    # thresholded metrics below leak no test statistic.
    val_scores = anomaly_scores(model, X_val)
    lo, hi = float(val_scores.min()), float(val_scores.max())
    test_proba = np.clip((test_scores - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
    thresholded = evaluate_with_thresholds(y_test, test_proba, args.thresholds)

    paths = config["paths"]
    for d in (paths["checkpoint_dir"], paths["output_dir"], config["logging"]["predictions_path"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{paths['checkpoint_dir']}/model.joblib")
    with open(f"{paths['output_dir']}/training_results.json", "w") as f:
        json.dump({"config": config,
                   "metrics": {"test": {"auroc": test_auroc, "auprc": test_auprc,
                                        "threshold_metrics": thresholded}}}, f, indent=4)
    np.save(f"{config['logging']['predictions_path']}/test_scores.npy", test_scores)
    print(f"Saved model and results under {paths['checkpoint_dir']} and {paths['output_dir']}")


if __name__ == "__main__":
    main()
