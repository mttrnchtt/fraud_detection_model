#!/usr/bin/env python3
"""
Inference stub. Loads a frozen stacking option (base models plus the meta-model
persisted by train_stack.py) and returns fraud probabilities for a feature matrix.
It is only a starting point. A real deployment still needs a validation-tuned
decision threshold, probability calibration, an HTTP or gRPC surface, input
validation and auth, batching, and drift monitoring (see the README, "What I
would do next").

Usage (demo, scores the processed test split):
    python scripts/serve.py --option B_wLGBM --n 5
"""

import argparse

import numpy as np

from fd.stack_helpers.meta import build_meta_features, load_meta

MODELS_ROOT = "models"


def score(X: np.ndarray, option: str = "B_wLGBM") -> np.ndarray:
    """Return P(fraud) for each row of X using the frozen stack `option`.

    X must have the same 32 preprocessed features produced by prepare_data.py.
    """
    meta, option_cfg = load_meta(MODELS_ROOT, option)
    X_meta = build_meta_features(option_cfg, X.astype(np.float32))
    return meta.predict_proba(X_meta)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--option", default="B_wLGBM", help="persisted stack option to load")
    parser.add_argument("--features", default="data_processed/X_test.npz",
                        help="npz with a 'data' array of preprocessed features")
    parser.add_argument("--n", type=int, default=5, help="how many rows to score for the demo")
    args = parser.parse_args()

    X = np.load(args.features)["data"][: args.n]
    proba = score(X, args.option)
    for i, p in enumerate(proba):
        print(f"row {i}: P(fraud)={p:.4f}")
    print("\n[stub] Returning raw model probabilities. No calibrated threshold is applied.")


if __name__ == "__main__":
    main()
