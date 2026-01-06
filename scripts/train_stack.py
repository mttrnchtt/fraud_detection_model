"""
Usage:
    python -m scripts.train_stack
Description:
    Trains a stacking meta-model (LogisticRegression) on layer-0 predictions
    and evaluates on the held-out test set for three layer-0 option sets (A/B/C).
    Writes a simple text report with AUPRC and AUROC per option.
"""

import yaml
from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

# Project data & layer-0 model option dictionaries
from src.fd.stack_helpers.utils import (
    load_all_data, run_layer0,
    layer0_option_A, layer0_option_B, layer0_option_A_wLGBM, layer0_option_B_wLGBM
)


# -----------------------
# Config / IO
# -----------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback defaults if config not found
        return {
            "paths": {
                "report": "outputs/stack/stack_report.txt"
            },
            "meta": {
                "solver": "liblinear",
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": 42
            }
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)


# -----------------------
# Meta training & eval
# -----------------------
def train_meta_logreg(X_meta_train: np.ndarray, y_train: np.ndarray, meta_cfg: dict) -> LogisticRegression:
    """
    Train a LogisticRegression meta-learner on stacked layer-0 probabilities.
    """
    solver = meta_cfg.get("solver", "liblinear")
    max_iter = int(meta_cfg.get("max_iter", 5000))
    class_weight = meta_cfg.get("class_weight", "balanced")
    random_state = meta_cfg.get("random_state", 42)

    # Create pipeline with scaling
    meta = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])
    
    meta.fit(X_meta_train, y_train)
    return meta


def evaluate_option(option_cfg: dict[str, str]) -> tuple[str, float, float]:
    """
    For a given option (A/B/C layer-0 set):
        - Build meta-train features from X_train
        - Train LR meta
        - Build meta-test features from X_test
        - Compute AUPRC & AUROC
    Returns a tuple (pretty_models_string, auprc, auroc)
    """
    X_train, Y_train, X_test, Y_test = load_all_data(CONFIG)
    X_meta_train = run_layer0(option_cfg, X_train)
    X_meta_test  = run_layer0(option_cfg, X_test)

    meta = train_meta_logreg(X_meta_train, Y_train, CONFIG.get("meta", {}))
    yhat = meta.predict_proba(X_meta_test)[:, 1]

    auprc = float(average_precision_score(Y_test, yhat))
    auroc = float(roc_auc_score(Y_test, yhat))

    models_list_str = str(list(option_cfg.keys()))
    return models_list_str, auprc, auroc


# -----------------------
# Main
# -----------------------
def main():
    report_path = Path(CONFIG["paths"]["report"])
    ensure_parent_dir(report_path)

    options = [
        # ("Option A", layer0_option_A),
        # ("Option B", layer0_option_B),
        ("Option_A_wLGBM", layer0_option_A_wLGBM),
        ("Option_B_wLGBM", layer0_option_B_wLGBM),
    ]

    with open(report_path, "w") as r:
        for opt_name, opt_cfg in options:
            models_str, auprc, auroc = evaluate_option(opt_cfg)

            line = (
                f"===== META MODEL LOGREG {opt_name} =====\n"
                f"Models: {models_str}\n"
                f"AUPRC: {auprc:.6f}\n"
                f"AUROC: {auroc:.6f}\n\n"
            )
            print(line, end="")
            r.write(line)


if __name__ == "__main__":
    CONFIG = load_config("configs/stack.yaml")
    main()
