# Credit Card Fraud Detection — Stacking Ensemble

A fraud detector for credit card transactions built to behave like a production system: it never looks into the future, it keeps the real fraud rate untouched, and it combines several different model families through stacking to squeeze out precision on an extremely rare positive class.

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 of them fraudulent (0.17%).

## Results

Final numbers on the held-out test window (the last 15% of transactions in time):

| Model | AUROC | AUPRC |
|---|---|---|
| **Stacking, meta-LR** | **0.9857** | 0.7848 |
| Stacking, meta-MLP (compact) | 0.9836 | **0.7865** |
| MLP (best single model) | ~0.983 | ~0.777 |
| XGBoost, weighted (w=5) | — | ~0.771 |
| Balanced Random Forest | — | ~0.763 |
| LightGBM | — | ~0.739 |
| Logistic Regression | — | ~0.690 |

The point of the table is not the top number on its own. It is that stacking beats every individual model, and it does so because the base learners make *different* mistakes.

## Why the setup matters

Most fraud tutorials quietly cheat in one of two ways, and both inflate the score:

**Random train/test splits.** If you shuffle transactions before splitting, the model trains on transactions that happened after the ones it is tested on. That is future leakage, and it will not survive contact with production. Here the data is split strictly by time:

- Train (first 70%) — trains the base models
- Validation (next 15%) — tunes base models and trains the meta-model
- Test (final 15%) — touched exactly once, for the numbers above

**Resampling the classes.** Oversampling fraud or undersampling the rest makes training easier but reports a fraud rate the model will never see live. Here the real 0.17% is preserved everywhere, and class imbalance is handled inside the models (class weights, balanced sampling) rather than by rewriting the dataset.

Because positives are so rare, the headline metric is **AUPRC**, not accuracy or AUROC. A model that flags nothing is 99.83% accurate and completely useless.

## How it works

**Layer 0 — base learners.** Six model families, chosen to disagree with each other:

- MLP — nonlinear boundaries
- XGBoost — gradient-boosted trees, weighted toward the fraud class
- Balanced Random Forest — bagging with balanced subsampling
- LightGBM — leaf-wise boosting
- Logistic Regression — a linear baseline worth keeping
- Isolation Forest — unsupervised anomaly score, the only model that never sees a label

**Layer 1 — meta-model.** The base models score the validation and test windows, and those scores become the features for a small meta-model (logistic regression or a compact MLP) that learns how to weigh them. Stacking helps here precisely because the base models are heterogeneous — a linear combiner on top of near-identical models would gain nothing.

## Repository layout

```
scripts/     runnable steps: prepare data, train each base model, build and evaluate the stack
src/fd/      shared code, organized per model (data_prep, rf_helpers, mlp_helpers, stack_helpers)
outputs/     saved training_results.json for every model variant
```

## Running it

```bash
git clone https://github.com/mttrnchtt/fraud_detection_model.git
cd fraud_detection_model

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# 1. prepare the time-ordered splits
python scripts/prepare_data.py

# 2. train the base models (examples)
python scripts/layer0_models.py
python scripts/train_lightgbm.py
python scripts/isolation_forest.py

# 3. build and evaluate the stack
python scripts/train_stack.py
python scripts/eval_stack.py
```

## What I would do next

- Calibrate the final scores (Platt / isotonic) so the threshold maps to a real precision–recall trade-off a fraud team could set by policy.
- Report cost-weighted metrics — a missed fraud and a false alarm are not equally expensive, and the operating point should reflect that.
- Add drift monitoring, since fraud patterns move and a model this dependent on temporal order will decay.
