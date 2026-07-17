"""Threshold selection and operating-point metrics for fraud scoring.

A 0.5 threshold is useless at 0.17 percent fraud, so the threshold is picked on
validation and the test window is scored at it.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(y_true, proba, method="max_f1", target_precision=0.9, alert_rate=0.01):
    """Pick a decision threshold on a scored split.

    method:
      "max_f1"           threshold that maximises F1
      "target_precision" lowest threshold reaching precision >= target_precision
      "alert_budget"     threshold that flags an alert_rate fraction of rows
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    if method == "alert_budget":
        k = max(1, int(round(alert_rate * len(proba))))
        return float(np.sort(proba)[::-1][k - 1])

    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]  # drop the trailing point with no threshold

    if method == "max_f1":
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        return float(thr[int(np.argmax(f1))])

    if method == "target_precision":
        ok = np.where(prec >= target_precision)[0]
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])

    raise ValueError(f"unknown threshold method: {method}")


def precision_at_k(y_true, proba, k):
    """Precision among the top-k highest-scored rows. k is a count, or a fraction in (0, 1]."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(proba)
    kk = int(round(k * n)) if 0 < k <= 1 else int(k)
    kk = max(1, min(kk, n))
    top = np.argsort(proba)[::-1][:kk]
    return float(y_true[top].sum() / kk)


def savings_score(y_true, y_pred, amount, review_cost):
    """Fraction of fraud loss avoided, net of review cost.

    Missing a fraud loses its amount. Flagging a transaction costs review_cost whether
    or not it is fraud. The baseline flags nothing and loses the full fraud amount.
    Returns (baseline_loss - model_loss) / baseline_loss.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amount = np.asarray(amount, dtype=float)

    baseline_loss = amount[y_true == 1].sum()
    if baseline_loss <= 0:
        return 0.0
    missed = (y_true == 1) & (y_pred == 0)
    model_loss = amount[missed].sum() + review_cost * (y_pred == 1).sum()
    return float((baseline_loss - model_loss) / baseline_loss)
