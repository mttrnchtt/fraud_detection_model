# set_seed(seed) - ensures reproducibility (simplest, good starting point)
# load_npz_data(path) - loads one .npz file
# load_npy_data(path) - loads one .npy file
# load_all_data(config) - loads all train/val/test data

import numpy as np
import sklearn

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

def load_all_data(config: dict) -> tuple[np.ndarray, ...]:
    """
    Loads all data from the config.
    """
    X_train = load_npz_data(config["paths"]["train_features"])
    y_train = load_npy_data(config["paths"]["train_labels"])
    X_val = load_npz_data(config["paths"]["val_features"])
    y_val = load_npy_data(config["paths"]["val_labels"])
    X_test = load_npz_data(config["paths"]["test_features"])
    y_test = load_npy_data(config["paths"]["test_labels"])
    return X_train, y_train, X_val, y_val, X_test, y_test