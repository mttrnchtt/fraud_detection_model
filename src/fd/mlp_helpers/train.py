import numpy as np

from fd.common import save_training_results  # re-exported for scripts that import it here
from .model import create_mlp_model, get_model_info
from .utils import load_all_data, set_seed
from .eval import compute_metrics

__all__ = ["compute_sample_weights", "train_mlp_model", "save_training_results"]


def compute_sample_weights(y, pos_weight):
    w = np.ones(len(y))
    w[y == 1] = pos_weight
    return w


def train_mlp_model(config):
    """Train the base MLP on the train split and score train, val, and test.

    Class imbalance is handled with per-sample weights at fit() time, since
    MLPClassifier has no class_weight. Returns (model, results, predictions).
    """
    set_seed(config["experiment"]["seed"])
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(config)

    model = create_mlp_model(config)
    pos_weight = config["training"]["pos_weight"]
    model.fit(X_train, y_train, sample_weight=compute_sample_weights(y_train, pos_weight))

    def proba(X):
        return model.predict_proba(X)[:, 1]

    results = {
        "model_info": get_model_info(model),
        "training_config": {"pos_weight": pos_weight, "validation_split": "manual"},
        "metrics": {
            "train": compute_metrics(y_train, model.predict(X_train), proba(X_train)),
            "val": compute_metrics(y_val, model.predict(X_val), proba(X_val)),
            "test": compute_metrics(y_test, model.predict(X_test), proba(X_test)),
        },
        "data_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "features": X_train.shape[1],
            "class_balance_train": {
                "positive": int(y_train.sum()),
                "negative": int(len(y_train) - y_train.sum()),
                "ratio": float(len(y_train) - y_train.sum()) / float(y_train.sum()),
            },
        },
    }
    predictions = {
        "train_pred": model.predict(X_train), "train_proba": proba(X_train),
        "val_pred": model.predict(X_val), "val_proba": proba(X_val),
        "test_pred": model.predict(X_test), "test_proba": proba(X_test),
    }
    return model, results, predictions
