# Credit Card Fraud Detection with a Stacking Ensemble

This project flags fraudulent credit card transactions. It is put together the way a real
deployment has to work. It never trains on transactions from the future, it keeps the true fraud
rate instead of resampling it away, and it picks which model to ship using a validation window so
the test window can be scored a single time at the very end.

The data is the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
284,807 transactions with 492 frauds, which is about 0.17 percent.

## Results

Every number below is on the held out test window, the last 15 percent of transactions in time.
The layer-0 set and the meta-model are chosen on the validation window. The test window is used
only for the final numbers reported here, never for selection. Everything is reproducible with
`make reproduce` and checked against the saved files under `outputs/` by
`scripts/check_readme_metrics.py`.

| Model (test set, scored once)         | AUROC  | AUPRC  |
|---------------------------------------|--------|--------|
| Stacking, meta-MLP (B_wLGBM)          | 0.9886 | 0.7746 |
| Stacking, meta-LR (B_wLGBM)           | 0.9880 | 0.7640 |
| XGBoost, weighted (w=5), best single  | 0.9808 | 0.7657 |
| Balanced Random Forest (s=1.0)        | 0.9826 | 0.7471 |
| LightGBM                              | 0.9713 | 0.7381 |
| MLP                                   | 0.9841 | 0.7234 |
| Logistic Regression, weighted         | 0.9772 | 0.6897 |
| Isolation Forest (unsupervised)       | 0.9371 | 0.0542 |

The MLP combiner comes out a little ahead of the best single model on both scores. The linear combiner scores about the same as the best single model. The gain from stacking is small here and depends on the combiner. AUPRC is the number to watch, because on data this imbalanced accuracy and AUROC stay high even for a useless model. A model that flags nothing at all is still 99.83 percent accurate.

## Turning scores into a decision

AUROC and AUPRC rank transactions but do not set a cutoff. `eval_stack.py` calibrates the selected
stack's scores with isotonic regression on validation, then picks a threshold on validation and
scores the test window at it. Calibration drops the test Brier score from 0.0110 to 0.0004, so the
threshold maps to a real precision and recall.

At the max-F1 threshold the selected stack (B_wLGBM) flags 44 of 42,721 test transactions:

| Metric (test, threshold set on validation) | Value |
|---------------------------------------------|-------|
| Precision                                   | 0.886 |
| Recall                                      | 0.750 |
| F1                                          | 0.812 |
| precision@0.5%                              | 0.192 |
| Savings, review cost 3.0                    | 0.594 |

Savings is the share of fraud dollars avoided net of review cost, using each transaction's Amount.
Missing a fraud loses its amount, reviewing a flagged transaction costs a fixed amount, and the
baseline flags nothing. The threshold, the calibration method, and the review cost are set in
`configs/stack.yaml`.

## Why the split and the sampling matter

Two shortcuts show up in a lot of fraud tutorials, and both make the score look better than it is.

The first is a random train and test split. Shuffle before you split and the model ends up
training on transactions that happened after the ones it is later tested on. That is leakage from
the future and it does not survive contact with production. Here the split is strictly by time.

- Train, the first 70 percent, fits the base models.
- Validation, the next 15 percent, ranks the base models and trains and selects the meta-model.
- Test, the final 15 percent, is scored once for the numbers above.

The second shortcut is resampling the classes. Oversampling the frauds or undersampling everything
else makes training easier but reports a fraud rate the model will never meet in production. Here
the real 0.17 percent is kept everywhere, and the imbalance is handled inside the models with class
weights and balanced sampling.

A third shortcut is easy to miss. Choosing which base models to keep by looking at their test scores
is itself a kind of leakage. Here the base models are ranked by validation AUPRC in
`scripts/select_layer0.py`, which writes `reports/layer0.txt`, and a guard called
`assert_meta_source_is_holdout` refuses to train the meta-model on the base training split.

## How it works

Layer 0 is a set of base models picked so they make different kinds of mistakes.

- MLP, for nonlinear boundaries
- XGBoost, weighted toward the fraud class
- Balanced Random Forest, with balanced subsampling
- LightGBM
- Logistic Regression, a linear baseline worth keeping
- Isolation Forest, an unsupervised anomaly score and the only model that never sees a label

Layer 1 is the meta-model. The base models score the validation window. Those scores are put on a
common scale first, which matters because the tree and linear models output probabilities while the
Isolation Forest outputs an unbounded anomaly score. The scaled scores become the features for a
small meta-model, either a logistic regression or a compact MLP, trained on validation. The frozen
meta-model then scores the test window once. Stacking helps here because the base models disagree
with each other. A combiner on top of near identical models would gain nothing.

## Layout

```
scripts/     one step per file, each run as `python scripts/<name>.py` from the repo root
src/fd/      the importable package: data_prep, mlp_helpers, stack_helpers, common
configs/     one YAML per step
outputs/     saved metrics per model and per stack option, tracked in git
reports/     layer0.txt (validation ranking), meta_model.txt (test), ensemble_report.txt
tests/       pytest suite for the feature pipeline and the leakage guards
exploratory/ older one off analyses, kept for history and left out of the pipeline
```

Model files and the raw data are not committed. The pipeline regenerates them. The small JSON metric
files under `outputs/` are committed as the evidence behind the numbers in this README.

## Running it

```bash
git clone https://github.com/mttrnchtt/fraud_detection_model.git
cd fraud_detection_model

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Reproduce everything with one command.

```bash
make reproduce
```

Or run the steps yourself.

```bash
python scripts/download_data.py     # needs Kaggle credentials, or drop creditcard.csv into data/
python scripts/prepare_data.py      # time ordered train, validation and test splits

python scripts/ensemble_models.py   # logistic regression, random forest and XGBoost variants
python scripts/train_layer0_mlp.py  # the MLP base model
python scripts/train_lightgbm.py    # the LightGBM base model
python scripts/isolation_forest.py  # the Isolation Forest base model

python scripts/select_layer0.py     # rank base models by validation AUPRC
python scripts/train_stack.py       # train the meta-model on validation
python scripts/eval_stack.py        # score the frozen stack on test, once
```

Run the tests with `make test`.

## Limits of this dataset

`V1..V28` are PCA components released for anonymity. There are no card, merchant, or device IDs and
no per-card history, so the methods used in production fraud systems do not apply here: velocity and
per-entity aggregation features, graph models over shared entities, and sequence models over a
card's history. What works here is what the repo already does, a cost-sensitive tree ensemble with
time-ordered validation, calibration, and a cost-based threshold. The stacking gain stays small.

## What I would do next

- Choose the threshold on a slice of validation the meta-model was not trained on. Today the
  calibrator and the threshold are fit on the same validation window the meta-model learns from, so
  the operating point carries a little in-sample optimism.
- Add drift monitoring, because fraud patterns move and a model this tied to time order will decay.
  There is a small `serve.py` that loads the frozen stack and scores new rows. It is only a starting
  point.
- Try CatBoost as an extra base learner and a LightGBM plus CatBoost blend. On this anonymized
  dataset the expected gain is small.
