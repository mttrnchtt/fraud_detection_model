"""Probability calibration for fraud scores.

Fit on the validation window, apply to test. The mapping is monotonic, so it changes
the numbers a threshold compares against but not the ranking of predictions.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_calibrator(proba, y, method="isotonic"):
    """Fit a calibrator on scores and labels.

    method is "isotonic" (non-parametric monotone) or "platt" (sigmoid).
    Returns a (kind, model) pair for apply_calibrator.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(proba, y)
        return ("isotonic", model)
    if method == "platt":
        model = LogisticRegression(solver="lbfgs")
        model.fit(proba.reshape(-1, 1), y)
        return ("platt", model)
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(calibrator, proba):
    """Apply a fitted calibrator to scores."""
    kind, model = calibrator
    proba = np.asarray(proba, dtype=float)
    if kind == "isotonic":
        return model.predict(proba)
    return model.predict_proba(proba.reshape(-1, 1))[:, 1]
