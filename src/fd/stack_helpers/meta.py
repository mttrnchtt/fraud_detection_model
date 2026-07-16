"""Layer-1 meta-model construction, IO, and scoring for the stacking ensemble.

Shared by ``scripts/train_stack.py`` (fits the meta-model on the *validation*
window and persists it) and ``scripts/eval_stack.py`` (loads the frozen
meta-model and scores the *test* window exactly once).
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .utils import run_layer0


def build_meta_features(option_cfg: dict[str, str], X: np.ndarray) -> np.ndarray:
    """Stack layer-0 scores for X into an (n_samples, n_models) meta-feature matrix."""
    return run_layer0(option_cfg, X)


def train_meta_logreg(X_meta: np.ndarray, y: np.ndarray, meta_cfg: dict | None = None) -> Pipeline:
    """Fit a MinMax-scaled LogisticRegression meta-learner on stacked layer-0 scores.

    The scaler matters: layer-0 scores mix [0, 1] probabilities with the
    Isolation Forest's unbounded anomaly score, so they must be put on a common
    scale before the linear combiner.
    """
    meta_cfg = meta_cfg or {}
    clf = LogisticRegression(
        solver=meta_cfg.get("solver", "liblinear"),
        max_iter=int(meta_cfg.get("max_iter", 5000)),
        class_weight=meta_cfg.get("class_weight", "balanced"),
        random_state=int(meta_cfg.get("random_state", 42)),
    )
    meta = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])
    meta.fit(X_meta, y)
    return meta


def score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUPRC (headline for rare positives) and AUROC for a probability/score vector."""
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
    }


def stack_dir(models_root: str, option_name: str) -> Path:
    return Path(models_root) / f"stack_{option_name}"


def save_meta(models_root: str, option_name: str, meta: Pipeline, option_cfg: dict[str, str]) -> Path:
    """Persist a fitted meta-model plus the layer-0 members it was built from."""
    d = stack_dir(models_root, option_name)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": meta, "members": list(option_cfg.keys()), "option_cfg": option_cfg}, d / "model.joblib")
    return d / "model.joblib"


def load_meta(models_root: str, option_name: str) -> tuple[Pipeline, dict[str, str]]:
    """Load a persisted meta-model and its layer-0 option mapping."""
    payload = joblib.load(stack_dir(models_root, option_name) / "model.joblib")
    return payload["meta"], payload["option_cfg"]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
