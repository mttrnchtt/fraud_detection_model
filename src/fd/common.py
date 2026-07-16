"""Shared building blocks for the base-MLP and stacking helper packages: seeding,
metric computation, and MLP construction. Both packages import from here so they
stay in sync.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


def set_seed(seed: int) -> None:
    """Set the process-wide NumPy RNG seed.

    Estimators additionally take ``random_state`` explicitly, so this plus fixed
    ``random_state`` makes the pipeline deterministic end to end.
    """
    np.random.seed(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Binary-classification metrics. AUPRC is the headline for rare positives."""
    return {
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_with_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds=None) -> dict:
    """Metrics at several decision thresholds (0.5 is rarely right at 0.17% positives)."""
    if thresholds is None:
        thresholds = [0.1, 0.25, 0.5]
    return {
        f"threshold_{t}": compute_metrics(y_true, (y_proba >= t).astype(int), y_proba)
        for t in thresholds
    }


def print_training_summary(results: Dict) -> None:
    """Human-readable summary of a training_results dict (val may be None)."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    info = results["model_info"]
    print("Model: MLPClassifier")
    print(f"Architecture: {info['hidden_layer_sizes']}")
    print(f"Activation: {info['activation']}")
    print(f"Solver: {info['solver']}")
    print(f"Learning Rate: {info['learning_rate_init']}")

    data = results["data_info"]
    print("\nData:")
    print(f"  Training samples: {data['train_samples']:,}")
    print(f"  Validation samples: {data['val_samples']:,}")
    print(f"  Test samples: {data['test_samples']:,}")
    print(f"  Features: {data['features']}")
    print(f"  Class ratio (neg:pos): {data['class_balance_train']['ratio']:.1f}:1")

    print("\nMetrics:")
    for split in ("train", "val", "test"):
        metrics = results["metrics"][split]
        print(f"  {split.upper()}:")
        if metrics is None:
            print("    (Not available)")
            continue
        for k in ("auroc", "auprc", "f1", "precision", "recall", "accuracy"):
            print(f"    {k.upper():5s} {metrics[k]:.4f}")
    print("=" * 60)


def create_mlp_model(config: dict) -> MLPClassifier:
    """Build an MLPClassifier from a config.

    Only the config keys MLPClassifier actually honors are read (see the config
    comments). Class imbalance is handled via sample weights at fit() time, not
    here, because MLPClassifier has no ``class_weight``.
    """
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]

    early_stopping_enabled = training_config["early_stopping"]["monitor"] is not None
    patience = training_config["early_stopping"]["patience"] if early_stopping_enabled else 10

    return MLPClassifier(
        hidden_layer_sizes=tuple(model_config["hidden_dims"]),
        activation=model_config["activation"],
        solver=optimizer_config["name"],
        learning_rate_init=optimizer_config["lr"],
        alpha=optimizer_config["weight_decay"],
        max_iter=training_config["epochs"],
        batch_size=training_config["batch_size"],
        shuffle=False,                 # preserve chronological order
        early_stopping=False,          # we hold out our own validation split
        validation_fraction=0,
        n_iter_no_change=patience,
        learning_rate="adaptive",
        random_state=config["experiment"]["seed"],
        verbose=False,
        warm_start=False,
    )


def get_model_info(model: MLPClassifier) -> dict:
    """Architecture summary for logging."""
    return {
        "hidden_layer_sizes": model.hidden_layer_sizes,
        "activation": model.activation,
        "solver": model.solver,
        "learning_rate_init": model.learning_rate_init,
        "alpha": model.alpha,
        "max_iter": model.max_iter,
        "n_layers_": getattr(model, "n_layers_", None),
        "n_outputs_": getattr(model, "n_outputs_", None),
        "classes_": model.classes_.tolist() if hasattr(model, "classes_") else None,
    }
