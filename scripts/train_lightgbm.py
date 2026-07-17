#!/usr/bin/env python3
"""
Train the LightGBM base model.

Usage:
    python scripts/train_lightgbm.py [-c configs/lightgbm.yaml] [--tune] [--thresholds ...]
"""

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score, roc_auc_score

from fd.common import evaluate_with_thresholds
from fd.data_prep.utils import load_config
from fd.mlp_helpers.utils import load_all_data


def train_lightgbm(config, X_train, y_train):
    """Fit LightGBM on the training split for a fixed number of rounds.

    No early stopping, because it would tune the round count on the validation window
    whose scores later feed the *_wLGBM stacks.
    """
    params = {k: v for k, v in config["model"].items() if k != "n_estimators"}
    train_data = lgb.Dataset(X_train, label=y_train)
    return lgb.train(
        params,
        train_data,
        num_boost_round=int(config["model"].get("n_estimators", 450)),
    )


def tune(config, X_train, y_train, X_val, y_val):
    """Random search over hyperparameters, ranked by validation AUPRC."""
    grid = config["tuning"]["param_grid"]
    best_auprc, best_model, best_config = -1.0, None, None
    for params in ParameterSampler(grid, n_iter=config["tuning"]["n_iter"],
                                   random_state=config["experiment"]["seed"]):
        cfg = copy.deepcopy(config)
        cfg["model"].update(params)
        model = train_lightgbm(cfg, X_train, y_train)
        auprc = average_precision_score(y_val, model.predict(X_val))
        print(f"  {params}: val AUPRC={auprc:.4f}")
        if auprc > best_auprc:
            best_auprc, best_model, best_config = auprc, model, cfg
    print(f"Best val AUPRC={best_auprc:.4f}")
    return best_model, best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/lightgbm.yaml")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    args = parser.parse_args()
    config = load_config(args.config)

    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(config)
    model, config = (tune(config, X_train, y_train, X_val, y_val) if args.tune
                     else (train_lightgbm(config, X_train, y_train), config))

    test_proba = model.predict(X_test)
    test_auroc = roc_auc_score(y_test, test_proba)
    test_auprc = average_precision_score(y_test, test_proba)
    print(f"Test AUROC={test_auroc:.4f} AUPRC={test_auprc:.4f}")
    thresholded = evaluate_with_thresholds(y_test, test_proba, args.thresholds)

    paths = config["paths"]
    for d in (paths["checkpoint_dir"], paths["output_dir"], config["logging"]["predictions_path"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{paths['checkpoint_dir']}/model.joblib")
    with open(f"{paths['output_dir']}/training_results.json", "w") as f:
        json.dump({"config": config,
                   "metrics": {"test": {"auroc": test_auroc, "auprc": test_auprc,
                                        "threshold_metrics": thresholded}}}, f, indent=4)
    np.save(f"{config['logging']['predictions_path']}/test_proba.npy", test_proba)
    print(f"Saved model and results under {paths['checkpoint_dir']} and {paths['output_dir']}")


if __name__ == "__main__":
    main()
