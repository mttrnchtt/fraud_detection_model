"""Unit tests for the feature pipeline, the anti-leakage invariants, and the
README<->artifact consistency check. All data-dependent tests use synthetic
frames so they run without the Kaggle CSV.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fd.data_prep.data import chronological_split
from fd.data_prep.features import Scaler, fit_scaler
from fd.stack_helpers.utils import OPTIONS, assert_meta_source_is_holdout
from fd.common import compute_metrics

ROOT = Path(__file__).resolve().parent.parent


def _synthetic_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Time": rng.permutation(np.arange(n) * 30.0),  # shuffled on purpose
            "Amount": rng.exponential(50.0, size=n),
            **{f"V{i}": rng.normal(size=n) for i in range(1, 29)},
            "Class": (rng.random(n) < 0.02).astype(int),
        }
    )


# --------------------------- chronological split ---------------------------

def test_chronological_split_is_time_ordered_and_disjoint():
    df = _synthetic_df(1000)
    tr, va, te = chronological_split(df, val_size=0.15, test_size=0.15)
    assert len(tr) + len(va) + len(te) == len(df)
    # strictly time-ordered windows: max(train) <= min(val) <= max(val) <= min(test)
    assert tr["Time"].max() <= va["Time"].min()
    assert va["Time"].max() <= te["Time"].min()


def test_chronological_split_sizes():
    df = _synthetic_df(1000)
    tr, va, te = chronological_split(df, val_size=0.15, test_size=0.15)
    assert len(te) == 150 and len(va) == 150 and len(tr) == 700


def test_chronological_split_rejects_degenerate_sizes():
    df = _synthetic_df(100)
    with pytest.raises(ValueError):
        chronological_split(df, val_size=0.6, test_size=0.6)


# --------------------------- scaler / features ---------------------------

def test_scaler_fits_on_train_only_and_outputs_four_columns():
    df = _synthetic_df(1000)
    tr, va, _ = chronological_split(df, 0.15, 0.15)
    scaler = fit_scaler(tr)
    assert isinstance(scaler, Scaler)
    out = scaler.get_normalized_features(tr)
    assert list(out.columns) == ["time_days_z", "tod_sin", "tod_cos", "Amount"]
    # time_days_z has ~zero mean on the train split it was fit on
    assert abs(float(out["time_days_z"].mean())) < 1e-6


def test_time_of_day_encoding_is_on_unit_circle():
    df = _synthetic_df(500)
    scaler = fit_scaler(df)
    out = scaler.transform_time_features(df)
    r2 = out["tod_sin"] ** 2 + out["tod_cos"] ** 2
    assert np.allclose(r2, 1.0, atol=1e-6)


def test_scaler_handles_zero_sigma_subset():
    df = pd.DataFrame({"Time": [10.0, 10.0, 10.0], "Amount": [5.0, 5.0, 5.0]})
    scaler = fit_scaler(df)  # sigma==0 -> guarded to 1.0, must not divide by zero
    out = scaler.get_normalized_features(df)
    assert np.isfinite(out.to_numpy()).all()


# --------------------------- anti-leakage guard ---------------------------

@pytest.mark.parametrize("bad", ["data_processed/X_train.npz", "X_TRAIN.npz", "/a/b/x_train.NPZ"])
def test_meta_source_guard_rejects_train_split(bad):
    with pytest.raises(ValueError):
        assert_meta_source_is_holdout(bad)


@pytest.mark.parametrize("ok", ["data_processed/X_val.npz", "X_val.npz", "oof_scores.npz"])
def test_meta_source_guard_accepts_holdout(ok):
    assert_meta_source_is_holdout(ok)  # must not raise


def test_layer0_options_reference_expected_model_paths():
    for name, opt in OPTIONS.items():
        assert opt, f"option {name} is empty"
        for member, path in opt.items():
            assert path.startswith("models/") and path.endswith("model.joblib")


# --------------------------- metrics ---------------------------

def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, (proba >= 0.5).astype(int), proba)
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0 and m["f1"] == 1.0


def test_compute_metrics_degenerate_predictions_do_not_crash():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4])  # never crosses 0.5
    m = compute_metrics(y, (proba >= 0.5).astype(int), proba)
    assert m["precision"] == 0.0 and m["recall"] == 0.0  # zero_division handled


# --------------------------- import smoke ---------------------------

@pytest.mark.parametrize("script", [
    "prepare_data", "ensamble_models", "train_layer0_mlp", "train_lightgbm",
    "isolation_forest", "select_layer0", "train_stack", "eval_stack",
    "train_stack_mlp", "download_data", "check_readme_metrics", "serve",
])
def test_pipeline_scripts_import_without_side_effects(script):
    path = ROOT / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"_smoke_{script}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top level must be import-safe (main guarded)


# --------------------------- README <-> artifacts ---------------------------

def test_readme_numbers_match_artifacts_when_present():
    from check_readme_metrics import EXPECTED, check_readme_metrics  # noqa: E402

    missing = [d for d in EXPECTED.values()
               if not (ROOT / "outputs" / d / "training_results.json").exists()]
    if missing:
        pytest.skip(f"artifacts not generated yet: {missing} (run `make all`)")
    assert check_readme_metrics() == []
