import numpy as np
import sklearn
import joblib


def set_seed(seed: int):
    """
    Sets the seed for reproducibility.
    """
    np.random.seed(seed)
    sklearn.utils.check_random_state(seed)


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


layer0_option_A = {
    'mlp': 'models/mlp/model.joblib',                      # Rank 1: 0.777
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',      # Rank 2: 0.771
    'xgb_w10': 'models/xgb_weighted_w10/model.joblib',    # Rank 3: 0.769
    'xgb_w50': 'models/xgb_weighted_w50/model.joblib',    # Rank 4: 0.765
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',      # Rank 5: 0.763
    'lr_weighted': 'models/lr_weighted/model.joblib',     # Rank 12: 0.690
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
    'mlp': 'models/mlp/model.joblib',                      # Rank 1: 0.777
    'xgb_w5': 'models/xgb_weighted_w5/model.joblib',      # Rank 2: 0.771
    'xgb_w10': 'models/xgb_weighted_w10/model.joblib',    # Rank 3: 0.769
    'xgb_w50': 'models/xgb_weighted_w50/model.joblib',    # Rank 4: 0.765
    'rf_s1': 'models/rf_balanced_s1.0/model.joblib',      # Rank 5: 0.763
    'lgbm': 'models/lightgbm/model.joblib',
    'lr_weighted': 'models/lr_weighted/model.joblib',     # Rank 12: 0.690
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