# exploratory/

One-off exploratory analyses — **not** part of the reproducible pipeline in
`scripts/`. They are kept for provenance but are intentionally quarantined here:

- They run analysis at import time and pop up matplotlib windows (`plt.show()`).
- They read `data/creditcard.csv` directly and derive their **own** time splits
  (with delay gaps / prequential folds) that are **not** comparable to the
  canonical 70/15/15 split produced by `scripts/prepare_data.py`.
- `unprocessed_model_selection.py` additionally re-searches over the *test*
  window — that is test peeking and is why these are excluded from the
  headline pipeline. Treat their numbers as scratch EDA, not results.

The trustworthy, leakage-controlled results come from `scripts/` (see the top-level
README). Nothing under `scripts/`, `src/`, or `tests/` imports anything here.
