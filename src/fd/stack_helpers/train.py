import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .model import create_mlp_model, get_model_info
from .utils import (
    assert_meta_source_is_holdout,
    load_all_data,
    run_layer0,
    set_seed,
    split_meta_data,
)
from .eval import compute_metrics


def compute_sample_weights(y, pos_weight):
    w = np.ones(len(y), dtype=np.float32)
    w[y == 1] = pos_weight
    return w


def train_mlp_model(config, option_cfg, tune=False):
    """Train the meta-MLP on stacked layer-0 scores.

    Meta-features come from scoring the held-out validation window with the base
    learners, which never saw it. With tune=True the meta-train is split again to
    pick hyperparameters. Returns (model, results, predictions).
    """
    set_seed(config["experiment"]["seed"])
    assert_meta_source_is_holdout(config["paths"]["train_features"])

    X_source, y_source, X_test_raw, y_test = load_all_data(config)
    X_meta_full = run_layer0(option_cfg, X_source)
    X_meta_test = run_layer0(option_cfg, X_test_raw)

    if tune:
        X_meta_train, y_train, X_meta_val, y_val = split_meta_data(X_meta_full, y_source, ratio=0.67)
    else:
        X_meta_train, y_train = X_meta_full, y_source
        X_meta_val = np.empty((0, X_meta_full.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=np.float32)

    # Base scores mix [0, 1] probabilities with the Isolation Forest's unbounded
    # anomaly score, so put them on a common scale (fit on meta-train) first.
    scaler = MinMaxScaler().fit(X_meta_train)
    X_meta_train = scaler.transform(X_meta_train)
    if X_meta_val.shape[0] > 0:
        X_meta_val = scaler.transform(X_meta_val)
    X_meta_test = scaler.transform(X_meta_test)

    model = create_mlp_model(config)

    n_pos = int(y_train.sum())
    pos_weight = float((len(y_train) - n_pos) / max(n_pos, 1))
    model.fit(X_meta_train, y_train, sample_weight=compute_sample_weights(y_train, pos_weight))

    train_proba = model.predict_proba(X_meta_train)[:, 1]
    test_proba = model.predict_proba(X_meta_test)[:, 1]
    if tune and X_meta_val.shape[0] > 0:
        val_pred = model.predict(X_meta_val)
        val_proba = model.predict_proba(X_meta_val)[:, 1]
    else:
        val_pred = np.empty((0,), dtype=np.float32)
        val_proba = np.empty((0,), dtype=np.float32)

    results = {
        "model_info": get_model_info(model),
        "training_config": {
            "pos_weight": pos_weight,
            "validation_split": "manual" if tune else None,
        },
        "metrics": {
            "train": compute_metrics(y_train, model.predict(X_meta_train), train_proba),
            "val": compute_metrics(y_val, val_pred, val_proba) if tune else None,
            "test": compute_metrics(y_test, model.predict(X_meta_test), test_proba),
        },
        "data_info": {
            "train_samples": int(X_meta_train.shape[0]),
            "val_samples": int(X_meta_val.shape[0]),
            "test_samples": int(X_meta_test.shape[0]),
            "features": int(X_meta_train.shape[1]),
            "class_balance_train": {
                "positive": n_pos,
                "negative": int(len(y_train) - n_pos),
                "ratio": pos_weight,
            },
        },
    }
    predictions = {
        "train_pred": model.predict(X_meta_train),
        "train_proba": train_proba,
        "val_pred": val_pred,
        "val_proba": val_proba,
        "test_pred": model.predict(X_meta_test),
        "test_proba": test_proba,
    }
    return model, results, predictions


def save_training_results(model, results, config, predictions=None):
    """Persist the model, its metrics, and optionally predictions and a config copy."""
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, checkpoint_dir / "model.joblib")
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if config["logging"]["save_predictions"] and predictions is not None:
        pred_dir = Path(config["logging"]["predictions_path"])
        pred_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in predictions.items():
            np.save(pred_dir / f"{name}.npy", arr)

    if config["logging"]["save_config_copy"]:
        import yaml

        with open(output_dir / "config_copy.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Saved model to {checkpoint_dir}/model.joblib and results to {output_dir}/training_results.json")
