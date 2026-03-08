# Credit Card Fraud Detection – Stacking Ensemble

This project implements a **time-aware credit card fraud detector** using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The goal is to build a realistic pipeline that:
1.  **Respects temporal order**: No future information leakage.
2.  **Preserves class imbalance**: Maintains the real-world ~0.17% fraud rate.
3.  **Maximizes AUPRC**: Uses heterogeneous models and stacking to push Average Precision as close as possible to 0.8.

---

## 📊 Data & Evaluation Setup

### Dataset
- **Records**: 284,807 transactions
- **Frauds**: 492 (~0.172%)
- **Features**:
    - `Time`: Seconds since the first transaction.
    - `Amount`: Transaction amount.
    - `V1–V28`: Anonymized PCA components.

### Time-Aware Splits
To mimic a production environment, we use strictly chronological splits:

| Split | Description | Usage |
| :--- | :--- | :--- |
| **Train A** | First 70% | Training Layer-0 (Base) Models |
| **Validation B** | Next 15% | Tuning Layer-0 & Training Layer-1 (Meta) Models |
| **Test C** | Final 15% | Final Evaluation |

> **Note:** No oversampling or undersampling is used. The real fraud rate is preserved to ensure realistic performance estimates.

---

## 🧠 Model Architecture

### Layer 0: Base Learners
We combine different "types of intelligence" to maximize diversity:

| Model Type | Description | Variants |
| :--- | :--- | :--- |
| **MLP** | Nonlinear intelligence | `sklearn` MLP (256-128-64), ReLU, Adam |
| **XGBoost** | Tree/Rule intelligence | Weighted ($w \in \{5, 10, 50, 100\}$) |
| **Random Forest** | Bagging intelligence | Balanced ($s \in \{0.5, 1.0\}$), Baseline |
| **LightGBM** | Leaf-wise tree intelligence | Tuned for high precision/recall trade-off |
| **Logistic Regression** | Linear intelligence | Weighted, Baseline |
| **Isolation Forest** | Anomaly intelligence | Unsupervised anomaly scores |

### Layer 1: Stacking (Meta-Models)
Meta-models take the predictions (scores) from Layer-0 models on splits B and C as input features.

- **Meta-MLP**: Small networks (e.g., [16, 8]), trained on B.
- **Meta-Logistic Regression**: Linear combination of base scores.

---

## 📈 Results (Test Window C)

### Top Stacking Ensembles

| Rank | Ensemble Type | Layer-0 Models | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **Meta-MLP (Compact)** | MLP, XGB (w5), RF (s1), LR (w), IF | 0.9836 | **0.7865** |
| 🥈 | **Meta-MLP (Extended)** | + LightGBM, More XGBs | 0.9850 | 0.7850 |
| 🥉 | **Meta-LR** | All Base Models | **0.9857** | 0.7848 |

### Base Model Performance (Reference)

| Model | AUPRC (approx) | Notes |
| :--- | :--- | :--- |
| **MLP** | ~0.777 | Strongest single model |
| **XGBoost (w=5)** | ~0.771 | Strong tree baseline |
| **Balanced RF** | ~0.763 | Good diversity |
| **LightGBM** | ~0.739 | Leaf-wise split diversity |
| **Logistic Regression** | ~0.690 | Linear baseline |

---

## 🚀 Usage

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nurlan-nurlybay/fraud_detection_model.git
cd fraud_detection_model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python -m scripts.preprocess_data
# Outputs data_processed/X_*.npz and y_*.npy
```

### 3. Train Layer-0 Models

```bash
# Train individual base models
python -m scripts.train_layer0_mlp -c configs/mlp.yaml
python -m scripts.train_lightgbm -c configs/lightgbm.yaml
python -m scripts.train_isolation_forest -c configs/isolation_forest.yaml
# ... see scripts/ for other models (RF, XGB, LR)
```

### 4. Train & Evaluate Stacked Model

```bash
python -m scripts.train_stack_mlp \
  -c configs/meta_mlp.yaml \
  --option A \
  --tune \
  --thresholds 0.1 0.25 0.5
```

---

## 📂 Directory Structure

```text
fraud_detection_model/
├── configs/              # YAML configuration files
├── data/                 # Raw dataset (creditcard.csv)
├── data_processed/       # Preprocessed numpy arrays
├── models/               # Saved model artifacts (.joblib)
├── notebooks/            # Exploratory notebooks
├── outputs/              # Training logs and predictions
├── reports/              # Figures and performance reports
├── scripts/              # Training and evaluation scripts
├── src/                  # Shared source code (utils, models)
└── tests/                # Unit tests
```
