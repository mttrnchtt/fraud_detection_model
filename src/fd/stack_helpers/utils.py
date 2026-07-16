"""Stacking layer-0 helpers.

Anti-leakage contract. The meta-model is trained on layer-0 scores of a window
the base models never saw during training, which is the validation split. The
test split is scored exactly once at the very end, for the reported numbers only.
Which base learners go into a layer-0 option is decided by validation AUPRC, never
by test (see reports/layer0.txt and scripts/select_layer0.py).

assert_meta_source_is_holdout guards this at runtime, so a mis-pointed config such
as train_features: X_train.npz fails loudly instead of leaking quietly.
"""

import os

import numpy as np
import joblib

from fd.common import set_seed  # re-exported for backwards compatibility

__all__ = [
    "set_seed",
    "assert_meta_source_is_holdout",
    "load_npz_data",
    "load_npy_data",
    "load_all_data",
    "split_meta_data",
    "run_layer0",
    "OPTIONS",
]


def assert_meta_source_is_holdout(path: str) -> None:
    """Fail fast if the meta-model would be trained on the base-learner train split.

    The meta-model must learn from layer-0 scores on a held-out window (the
    validation split), not the split the base learners were fit on. We refuse the
    obvious leakage case where ``train_features`` points at ``X_train.*``.
    """
    base = os.path.basename(str(path)).lower()
    if base.startswith("x_train"):
        raise ValueError(
            f"Refusing to train the meta-model on {path!r}: stacking requires a "
            "held-out source (e.g. data_processed/X_val.npz). Point "
            "paths.train_features at the validation split, not the layer-0 "
            "training split."
        )


def load_npz_data(path: str) -> np.ndarray:
    """
    Loads data from a .npz file.
    """
    return np.load(path)['data'].astype(np.float32)


def load_npy_data(path: str) -> np.ndarray:
    """
    Loads data from a .npy file.
    """
    return np.load(path).astype(np.float32)


def load_all_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads train and test data from the config.
    """
    X_train = load_npz_data(config["paths"]["train_features"])
    y_train = load_npy_data(config["paths"]["train_labels"])
    X_test = load_npz_data(config["paths"]["test_features"])
    y_test = load_npy_data(config["paths"]["test_labels"])
    return X_train, y_train, X_test, y_test


def split_meta_data(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.67
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split meta data into (train, val) while preserving order.

    ratio: fraction of samples to use for meta-train.
    """
    N = len(X)
    split = int(N * ratio)

    X_meta_train = X[:split]
    y_meta_train = y[:split]

    X_meta_val = X[split:]
    y_meta_val = y[split:]

    return X_meta_train, y_meta_train, X_meta_val, y_meta_val


def _run_layer0_model(model_path: str, X: np.ndarray) -> np.ndarray:
    """
    Load a trained layer-0 model and return a 1D score for X.
    - For classifiers: P(y=1)
    - For anomaly models (IsolationForest): anomaly score
    """
    model = joblib.load(model_path)

    if hasattr(model, "predict_proba"):
        # Standard probabilistic layer-0 model
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "score_samples"):
        # Isolation Forest: score_samples -> higher = more normal,
        # so we negate to get higher = more anomalous.
        return -model.score_samples(X)
    elif hasattr(model, "predict"):
        # LightGBM Booster or similar that returns raw scores/probs directly
        return model.predict(X)
    else:
        raise ValueError(f"Model at {model_path} has neither predict_proba nor score_samples")


def run_layer0(layer0_config: dict[str, str], X: np.ndarray) -> np.ndarray:
    """
    Stack predictions of all layer-0 models on X -> (n_samples, n_models).
    layer0_config maps model_name -> path_to_model.joblib
    """
    preds = []
    for name, model_path in layer0_config.items():
        y_proba = _run_layer0_model(model_path, X)
        preds.append(y_proba)
    return np.column_stack(preds)


# Layer-0 model sets. Members are chosen by VALIDATION AUPRC (see
# scripts/select_layer0.py -> reports/layer0.txt), never by test. Option A is the
# broad set (diverse XGB weights); Option B is the compact set; the *_wLGBM
# variants add the LightGBM booster.
layer0_option_A = {
    'mlp': 'models/mlp/model.joblib',
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',
    'xgb_w10': 'models/xgb_weighted_w10/model.joblib',
    'xgb_w50': 'models/xgb_weighted_w50/model.joblib',
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',
    'lr_weighted': 'models/lr_weighted/model.joblib',
    'isolation_forest': 'models/isolation_forest/model.joblib',
}

layer0_option_B = {
    'mlp': 'models/mlp/model.joblib',
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',
    'lr_weighted': 'models/lr_weighted/model.joblib',
    'isolation_forest': 'models/isolation_forest/model.joblib',
}

layer0_option_A_wLGBM = {
    'mlp': 'models/mlp/model.joblib',
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',
    'xgb_w10': 'models/xgb_weighted_w10/model.joblib',
    'xgb_w50': 'models/xgb_weighted_w50/model.joblib',
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',
    'lgbm': 'models/lightgbm/model.joblib',
    'lr_weighted': 'models/lr_weighted/model.joblib',
    'isolation_forest': 'models/isolation_forest/model.joblib',
}

layer0_option_B_wLGBM = {
    'mlp': 'models/mlp/model.joblib',
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',
    'lgbm': 'models/lightgbm/model.joblib',
    'lr_weighted': 'models/lr_weighted/model.joblib',
    'isolation_forest': 'models/isolation_forest/model.joblib',
}

# Name -> layer-0 set, so scripts can select an option by string.
OPTIONS = {
    "A": layer0_option_A,
    "B": layer0_option_B,
    "A_wLGBM": layer0_option_A_wLGBM,
    "B_wLGBM": layer0_option_B_wLGBM,
}