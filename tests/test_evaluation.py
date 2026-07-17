"""Unit tests for the operating-point evaluation layer: calibration, threshold
selection, precision@k, and the Amount-weighted savings metric. All synthetic.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fd.calibration import apply_calibrator, fit_calibrator
from fd.cost_metrics import precision_at_k, savings_score, select_threshold


def _scored(n=2000, seed=0):
    """Separable-ish scores: positives score higher on average."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    proba = rng.random(n) * 0.4 + 0.3 * y  # positives shifted up
    return y, proba


# --------------------------- calibration ---------------------------

@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_is_monotonic_and_bounded(method):
    y, proba = _scored()
    cal = fit_calibrator(proba, y, method=method)
    out = apply_calibrator(cal, proba)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # maps scores monotonically: a rising input grid gives non-decreasing output
    grid = np.linspace(proba.min(), proba.max(), 50)
    mapped = apply_calibrator(cal, grid)
    assert np.all(np.diff(mapped) >= -1e-9)


def test_platt_calibration_preserves_auroc_exactly():
    # a strictly monotonic sigmoid does not create ties, so AUROC is unchanged
    y, proba = _scored()
    cal = fit_calibrator(proba, y, method="platt")
    out = apply_calibrator(cal, proba)
    assert roc_auc_score(y, out) == pytest.approx(roc_auc_score(y, proba), abs=1e-9)


def test_calibration_rejects_unknown_method():
    y, proba = _scored()
    with pytest.raises(ValueError):
        fit_calibrator(proba, y, method="nope")


# --------------------------- threshold selection ---------------------------

def test_max_f1_threshold_separates_clean_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = select_threshold(y, proba, method="max_f1")
    assert 0.3 < thr <= 0.7
    pred = (proba >= thr).astype(int)
    assert (pred == y).all()


def test_target_precision_threshold_reaches_precision():
    y, proba = _scored(seed=1)
    thr = select_threshold(y, proba, method="target_precision", target_precision=0.8)
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    flagged = int(pred.sum())
    assert flagged == 0 or tp / flagged >= 0.8 - 1e-9


def test_alert_budget_flags_expected_fraction():
    y, proba = _scored(seed=2)
    thr = select_threshold(y, proba, method="alert_budget", alert_rate=0.02)
    flagged = int((proba >= thr).sum())
    assert abs(flagged - 0.02 * len(proba)) <= 0.02 * len(proba)  # within one budget


def test_unknown_threshold_method_raises():
    y, proba = _scored()
    with pytest.raises(ValueError):
        select_threshold(y, proba, method="nope")


# --------------------------- precision@k ---------------------------

def test_precision_at_k_counts_and_fractions():
    y = np.array([1, 0, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, proba, 2) == pytest.approx(0.5)   # top 2: 0.9(1), 0.8(0)
    assert precision_at_k(y, proba, 3) == pytest.approx(2 / 3)  # top 3 has 2 positives
    assert precision_at_k(y, proba, 0.4) == pytest.approx(0.5)  # fraction -> top 2


# --------------------------- savings ---------------------------

def test_savings_perfect_catch_no_review_cost():
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 10.0, 50.0, 5.0])
    pred = y.copy()  # catch all fraud, flag nothing else
    assert savings_score(y, pred, amount, review_cost=0.0) == pytest.approx(1.0)


def test_savings_flag_nothing_is_zero():
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 10.0, 50.0, 5.0])
    pred = np.zeros_like(y)
    assert savings_score(y, pred, amount, review_cost=3.0) == pytest.approx(0.0)


def test_savings_penalizes_review_cost():
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 10.0, 50.0, 5.0])
    pred = np.ones_like(y)  # catch all fraud (150) but pay review on all 4 rows
    # baseline loss = 150, model loss = 0 missed + 3*4 review = 12 -> savings = (150-12)/150
    assert savings_score(y, pred, amount, review_cost=3.0) == pytest.approx((150 - 12) / 150)


def test_savings_no_fraud_returns_zero():
    y = np.array([0, 0, 0])
    amount = np.array([1.0, 2.0, 3.0])
    assert savings_score(y, np.ones_like(y), amount, review_cost=1.0) == 0.0
