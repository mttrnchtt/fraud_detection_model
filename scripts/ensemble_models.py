#!/usr/bin/env python3
"""
Train the layer-0 tree and linear base models and save them for stacking.

Models trained:
- baseline Logistic Regression, Random Forest, XGBoost
- Balanced Random Forest over several sampling strategies
- weighted XGBoost over several positive-class weights
- class-weighted Logistic Regression

Usage:
    python scripts/ensemble_models.py [-c configs/ensemble.yaml]
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fd.common import compute_metrics as base_metrics
from fd.data_prep.utils import load_config


def load_processed_data(processed_dir):
    d = Path(processed_dir)
    return (
        np.load(d / "X_train.npz")["data"], np.load(d / "X_val.npz")["data"],
        np.load(d / "X_test.npz")["data"],
        np.load(d / "y_train.npy"), np.load(d / "y_val.npy"), np.load(d / "y_test.npy"),
    )


def compute_metrics(y_true, y_pred, y_proba):
    """Shared metrics plus balanced accuracy, which this sweep reports."""
    m = base_metrics(y_true, y_pred, y_proba)
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    return m


def base_specs(seed):
    """One (display name, output dir, estimator) per model in the sweep."""
    specs = [
        ("Baseline Logistic Regression", "lr_baseline",
         LogisticRegression(C=0.1, random_state=seed, max_iter=1000)),
        ("Baseline Random Forest", "rf_baseline",
         RandomForestClassifier(max_depth=5, n_estimators=50, random_state=seed, n_jobs=-1)),
        ("Baseline XGBoost", "xgb_baseline",
         xgboost.XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=50,
                               random_state=seed, n_jobs=-1, eval_metric="logloss")),
    ]
    for s in (0.01, 0.05, 0.1, 0.5, 1.0):
        specs.append((f"Balanced RF (s={s})", f"rf_balanced_s{s}",
                      BalancedRandomForestClassifier(max_depth=5, n_estimators=50,
                                                     sampling_strategy=s, random_state=seed, n_jobs=-1)))
    for w in (1, 5, 10, 50, 100):
        specs.append((f"Weighted XGBoost (w={w})", f"xgb_weighted_w{w}",
                      xgboost.XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=50,
                                            scale_pos_weight=w, random_state=seed, n_jobs=-1,
                                            eval_metric="logloss")))
    specs.append(("Weighted Logistic Regression", "lr_weighted",
                  LogisticRegression(C=0.1, random_state=seed, max_iter=1000, class_weight="balanced")))
    return specs


def train_and_evaluate(classifier, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit the classifier (standard-scaled) and score it on train, val, and test."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
    start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start

    predictions, results = {}, {}
    for split, X_split, y_split in (
        ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)
    ):
        start = time.time()
        y_proba = pipe.predict_proba(X_split)[:, 1]
        pred_time = time.time() - start
        y_pred = (y_proba >= 0.5).astype(int)
        predictions[f"{split}_pred"] = y_pred
        predictions[f"{split}_proba"] = y_proba
        m = compute_metrics(y_split, y_pred, y_proba)
        m["pred_time"] = pred_time
        results[split] = m
    results["train_time"] = train_time
    return pipe, results, predictions


def save_model_results(name, pipe, results, predictions, model_dir, output_dir):
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    pred_dir = output_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_dir / "model.joblib")
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for key, arr in predictions.items():
        np.save(pred_dir / f"{key}.npy", arr)
    print(f"  saved {name} to {model_dir} and {output_dir}")


def summary_row(name, results):
    t = results["test"]
    return {
        "Model": name,
        "Fit time (s)": f"{results['train_time']:.3f}",
        "Score time (s)": f"{t['pred_time']:.3f}",
        "AUPRC": f"{t['auprc']:.3f}",
        "AUC ROC": f"{t['auroc']:.3f}",
        "Balanced Accuracy": f"{t['balanced_accuracy']:.3f}",
    }


def write_report(report_path, rows, y_train, y_val, y_test):
    df = pd.DataFrame(rows).sort_values("AUPRC", ascending=False, ignore_index=True)
    best = df.iloc[0]
    with open(report_path, "w") as r:
        r.write("Layer-0 base models for imbalanced fraud detection.\n")
        r.write("Ranked by AUPRC, the headline metric at 0.17 percent positives.\n\n")
        r.write(f"Train: {len(y_train)} rows, {int(y_train.sum())} frauds\n")
        r.write(f"Val:   {len(y_val)} rows, {int(y_val.sum())} frauds\n")
        r.write(f"Test:  {len(y_test)} rows, {int(y_test.sum())} frauds\n\n")
        r.write(df.to_string(index=False))
        r.write(f"\n\nBest model by test AUPRC: {best['Model']} "
                f"(AUPRC={best['AUPRC']}, AUC ROC={best['AUC ROC']})\n")
    print(f"Report written to {report_path}")


def main(config_path):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    seed = cfg["experiment"]["seed"]

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(paths["processed_dir"])
    print(f"Train {X_train.shape[0]} rows, {int(y_train.sum())} frauds ({100 * y_train.mean():.2f}%)")

    rows = []
    for name, dirname, clf in base_specs(seed):
        print(f"Training {name} ...")
        pipe, results, predictions = train_and_evaluate(
            clf, X_train, y_train, X_val, y_val, X_test, y_test
        )
        save_model_results(name, pipe, results, predictions,
                           f"{paths['models_dir']}/{dirname}", f"{paths['outputs_dir']}/{dirname}")
        rows.append(summary_row(name, results))

    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_report(reports_dir / "ensemble_report.txt", rows, y_train, y_val, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/ensemble.yaml")
    args = parser.parse_args()
    main(args.config)
