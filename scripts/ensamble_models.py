#!/usr/bin/env python3
"""
Train and evaluate ensemble models for fraud detection.

This script trains multiple baseline and ensemble models, saves them,
and generates predictions for stacking in layer 1.

Models trained:
- Baseline Logistic Regression
- Baseline Random Forest
- Baseline XGBoost
- Balanced Random Forest (multiple sampling strategies)
- Weighted XGBoost (multiple weights)
- Weighted Logistic Regression

Usage:
    python -m scripts.ensamble_models
"""

import json
import numpy as np
import pandas as pd
import re
import time
import joblib
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================
# UTILITY FUNCTIONS
# ============================================
def load_processed_data(processed_dir="data_processed"):
    """Load processed data from data_processed directory."""
    data_dir = Path(processed_dir)
    X_train = np.load(data_dir / "X_train.npz")['data']
    X_val = np.load(data_dir / "X_val.npz")['data']
    X_test = np.load(data_dir / "X_test.npz")['data']
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    with open(data_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, manifest


def compute_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive metrics for binary classification."""
    return {
        'auroc': metrics.roc_auc_score(y_true, y_proba),
        'auprc': metrics.average_precision_score(y_true, y_proba),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred)
    }


def train_and_evaluate(classifier, X_train, y_train, X_val, y_val, X_test, y_test, scale=True):
    """
    Train a model and evaluate on all splits.
    
    Returns:
        pipe: Trained pipeline
        results: Dictionary with metrics and timing
        predictions: Dictionary with predictions for all splits
    """
    if scale:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
    else:
        pipe = Pipeline([('clf', classifier)])
    
    # Train
    start_time = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict on all splits
    predictions = {}
    results = {}
    
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        start_time = time.time()
        y_pred_proba = pipe.predict_proba(X_split)[:, 1]
        pred_time = time.time() - start_time
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        predictions[f'{split_name}_pred'] = y_pred
        predictions[f'{split_name}_proba'] = y_pred_proba
        
        split_metrics = compute_metrics(y_split, y_pred, y_pred_proba)
        split_metrics['pred_time'] = pred_time
        results[split_name] = split_metrics
    
    results['train_time'] = train_time
    
    return pipe, results, predictions


def save_model_results(model_name, pipe, results, predictions, model_dir, output_dir):
    """Save model, results, and predictions to disk."""
    # Create directories
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'model.joblib'
    joblib.dump(pipe, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Save predictions
    for key, value in predictions.items():
        np.save(pred_dir / f'{key}.npy', value)
    print(f"  Predictions saved to: {pred_dir}")


def get_model_dir_name(model_name):
    """Convert model display name to directory-friendly name."""
    # Remove special characters and spaces
    name = model_name.lower()
    name = name.replace(' ', '_')
    name = name.replace('(', '')
    name = name.replace(')', '')
    name = name.replace('=', '')
    name = name.replace(',', '')
    # Handle specific cases
    if 'baseline' in name:
        if 'logistic' in name:
            return 'lr_baseline'
        elif 'random' in name:
            return 'rf_baseline'
        elif 'xgboost' in name:
            return 'xgb_baseline'
    elif 'balanced_rf' in name:
        # Extract sampling strategy
        strategy = name.split('s')[1].strip()
        return f'rf_balanced_s{strategy}'
    elif 'weighted_xgboost' in name:
        # Extract weight - use regex to find 'w' followed by digits
        # This avoids splitting on 'w' in "weighted"
        match = re.search(r'w(\d+)', name)
        if match:
            weight = match.group(1)
            return f'xgb_weighted_w{weight}'
        # Fallback if regex doesn't match
        return 'xgb_weighted_unknown'
    elif 'weighted_logistic' in name:
        return 'lr_weighted'
    return name


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, manifest = load_processed_data()
    
    fraud_rate = y_train.mean()
    imbalance_ratio = fraud_rate / (1 - fraud_rate)
    
    print(f"Train set: {X_train.shape[0]} samples, {y_train.sum()} frauds ({100*y_train.mean():.2f}%)")
    print(f"Val set:   {X_val.shape[0]} samples, {y_val.sum()} frauds ({100*y_val.mean():.2f}%)")
    print(f"Test set:  {X_test.shape[0]} samples, {y_test.sum()} frauds ({100*y_test.mean():.2f}%)")
    print()
    
    # Store all models and results
    all_models = {}
    results_list = []
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # BASELINE MODELS
    # ============================================
    print("="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    # Logistic Regression Baseline
    print("\n[1/3] Training Baseline Logistic Regression...")
    lr_baseline = LogisticRegression(C=0.1, random_state=0, max_iter=1000)
    pipe, results, predictions = train_and_evaluate(
        lr_baseline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    model_name = 'Baseline Logistic Regression'
    model_dir_name = get_model_dir_name(model_name)
    save_model_results(
        model_name, pipe, results, predictions,
        f'models/{model_dir_name}',
        f'outputs/{model_dir_name}'
    )
    all_models[model_name] = pipe
    results_list.append({
        'Model': model_name,
        'Fit time (s)': f"{results['train_time']:.3f}",
        'Score time (s)': f"{results['test']['pred_time']:.3f}",
        'AUC ROC': f"{results['test']['auroc']:.3f}",
        'Average Precision': f"{results['test']['auprc']:.3f}",
        'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
    })
    
    # Random Forest Baseline
    print("\n[2/3] Training Baseline Random Forest...")
    rf_baseline = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=0, n_jobs=-1)
    pipe, results, predictions = train_and_evaluate(
        rf_baseline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    model_name = 'Baseline Random Forest'
    model_dir_name = get_model_dir_name(model_name)
    save_model_results(
        model_name, pipe, results, predictions,
        f'models/{model_dir_name}',
        f'outputs/{model_dir_name}'
    )
    all_models[model_name] = pipe
    results_list.append({
        'Model': model_name,
        'Fit time (s)': f"{results['train_time']:.3f}",
        'Score time (s)': f"{results['test']['pred_time']:.3f}",
        'AUC ROC': f"{results['test']['auroc']:.3f}",
        'Average Precision': f"{results['test']['auprc']:.3f}",
        'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
    })
    
    # XGBoost Baseline
    print("\n[3/3] Training Baseline XGBoost...")
    xgb_baseline = xgboost.XGBClassifier(
        learning_rate=0.3, max_depth=6, n_estimators=50,
        random_state=0, n_jobs=-1, eval_metric='logloss'
    )
    pipe, results, predictions = train_and_evaluate(
        xgb_baseline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    model_name = 'Baseline XGBoost'
    model_dir_name = get_model_dir_name(model_name)
    save_model_results(
        model_name, pipe, results, predictions,
        f'models/{model_dir_name}',
        f'outputs/{model_dir_name}'
    )
    all_models[model_name] = pipe
    results_list.append({
        'Model': model_name,
        'Fit time (s)': f"{results['train_time']:.3f}",
        'Score time (s)': f"{results['test']['pred_time']:.3f}",
        'AUC ROC': f"{results['test']['auroc']:.3f}",
        'Average Precision': f"{results['test']['auprc']:.3f}",
        'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
    })
    
    # ============================================
    # BALANCED RANDOM FOREST
    # ============================================
    print("\n" + "="*80)
    print("TRAINING BALANCED RANDOM FOREST MODELS")
    print("="*80)
    sampling_strategies = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for i, sampling_strategy in enumerate(sampling_strategies, 1):
        print(f"\n[{i}/{len(sampling_strategies)}] Training Balanced RF (s={sampling_strategy})...")
        brf = BalancedRandomForestClassifier(
            max_depth=5, n_estimators=50, sampling_strategy=sampling_strategy,
            random_state=0, n_jobs=-1
        )
        pipe, results, predictions = train_and_evaluate(
            brf, X_train, y_train, X_val, y_val, X_test, y_test
        )
        model_name = f'Balanced RF (s={sampling_strategy})'
        model_dir_name = get_model_dir_name(model_name)
        save_model_results(
            model_name, pipe, results, predictions,
            f'models/{model_dir_name}',
            f'outputs/{model_dir_name}'
        )
        all_models[model_name] = pipe
        results_list.append({
            'Model': model_name,
            'Fit time (s)': f"{results['train_time']:.3f}",
            'Score time (s)': f"{results['test']['pred_time']:.3f}",
            'AUC ROC': f"{results['test']['auroc']:.3f}",
            'Average Precision': f"{results['test']['auprc']:.3f}",
            'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
        })
    
    # ============================================
    # WEIGHTED XGBOOST
    # ============================================
    print("\n" + "="*80)
    print("TRAINING WEIGHTED XGBOOST MODELS")
    print("="*80)
    scale_weights = [1, 5, 10, 50, 100]
    
    for i, scale_weight in enumerate(scale_weights, 1):
        print(f"\n[{i}/{len(scale_weights)}] Training Weighted XGBoost (w={scale_weight})...")
        wxgb = xgboost.XGBClassifier(
            learning_rate=0.3, max_depth=6, n_estimators=50,
            scale_pos_weight=scale_weight, random_state=0, n_jobs=-1, eval_metric='logloss'
        )
        pipe, results, predictions = train_and_evaluate(
            wxgb, X_train, y_train, X_val, y_val, X_test, y_test
        )
        model_name = f'Weighted XGBoost (w={scale_weight})'
        model_dir_name = get_model_dir_name(model_name)
        save_model_results(
            model_name, pipe, results, predictions,
            f'models/{model_dir_name}',
            f'outputs/{model_dir_name}'
        )
        all_models[model_name] = pipe
        results_list.append({
            'Model': model_name,
            'Fit time (s)': f"{results['train_time']:.3f}",
            'Score time (s)': f"{results['test']['pred_time']:.3f}",
            'AUC ROC': f"{results['test']['auroc']:.3f}",
            'Average Precision': f"{results['test']['auprc']:.3f}",
            'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
        })
    
    # ============================================
    # CLASS-WEIGHTED LOGISTIC REGRESSION
    # ============================================
    print("\n" + "="*80)
    print("TRAINING WEIGHTED LOGISTIC REGRESSION")
    print("="*80)
    print("\n[1/1] Training Weighted Logistic Regression...")
    lr_weighted = LogisticRegression(C=0.1, random_state=0, max_iter=1000, class_weight='balanced')
    pipe, results, predictions = train_and_evaluate(
        lr_weighted, X_train, y_train, X_val, y_val, X_test, y_test
    )
    model_name = 'Weighted Logistic Regression'
    model_dir_name = get_model_dir_name(model_name)
    save_model_results(
        model_name, pipe, results, predictions,
        f'models/{model_dir_name}',
        f'outputs/{model_dir_name}'
    )
    all_models[model_name] = pipe
    results_list.append({
        'Model': model_name,
        'Fit time (s)': f"{results['train_time']:.3f}",
        'Score time (s)': f"{results['test']['pred_time']:.3f}",
        'AUC ROC': f"{results['test']['auroc']:.3f}",
        'Average Precision': f"{results['test']['auprc']:.3f}",
        'Balanced Accuracy': f"{results['test']['balanced_accuracy']:.3f}"
    })
    
    # ============================================
    # CREATE RESULTS TABLE AND REPORT
    # ============================================
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    results_df = pd.DataFrame(results_list)
    
    # Create report file in reports directory
    report_path = reports_dir / 'ensemble_report.txt'
    report = open(report_path, 'w')
    
    # Header
    report.write("="*100 + "\n")
    report.write("ENSEMBLE METHODS FOR IMBALANCED FRAUD DETECTION - EXPERIMENTAL RESULTS\n")
    report.write("="*100 + "\n\n")
    
    # Dataset info
    report.write("DATASET INFORMATION\n")
    report.write("-"*100 + "\n")
    report.write(f"Train set: {X_train.shape[0]} samples, {y_train.sum()} frauds ({100*y_train.mean():.2f}%)\n")
    report.write(f"Val set:   {X_val.shape[0]} samples, {y_val.sum()} frauds ({100*y_val.mean():.2f}%)\n")
    report.write(f"Test set:  {X_test.shape[0]} samples, {y_test.sum()} frauds ({100*y_test.mean():.2f}%)\n")
    report.write(f"Imbalance ratio: {imbalance_ratio:.4f}\n")
    report.write(f"Inverse imbalance ratio: {1/imbalance_ratio:.2f}\n\n")
    
    # Complete results
    report.write("="*100 + "\n")
    report.write("COMPLETE RESULTS - ALL ENSEMBLE METHODS\n")
    report.write("="*100 + "\n\n")
    report.write(results_df.to_string(index=False))
    report.write("\n\n")
    
    # Baseline comparison
    baseline_df = results_df[results_df['Model'].str.contains('Baseline')].copy()
    report.write("="*100 + "\n")
    report.write("BASELINE MODELS COMPARISON\n")
    report.write("="*100 + "\n\n")
    report.write(baseline_df.to_string(index=False))
    report.write("\n\n")
    
    # Balanced RF comparison
    brf_df = results_df[results_df['Model'].str.contains('Balanced RF')].copy()
    baseline_rf = results_df[results_df['Model'] == 'Baseline Random Forest'].copy()
    brf_comparison = pd.concat([baseline_rf, brf_df], ignore_index=True)
    report.write("="*100 + "\n")
    report.write("RANDOM FOREST: BASELINE VS BALANCED\n")
    report.write("="*100 + "\n\n")
    report.write(brf_comparison.to_string(index=False))
    report.write("\n\n")
    
    # Weighted XGBoost comparison
    wxgb_df = results_df[results_df['Model'].str.contains('Weighted XGBoost')].copy()
    baseline_xgb = results_df[results_df['Model'] == 'Baseline XGBoost'].copy()
    wxgb_comparison = pd.concat([baseline_xgb, wxgb_df], ignore_index=True)
    report.write("="*100 + "\n")
    report.write("XGBOOST: BASELINE VS WEIGHTED\n")
    report.write("="*100 + "\n\n")
    report.write(wxgb_comparison.to_string(index=False))
    report.write("\n\n")
    
    # Logistic Regression comparison
    lr_baseline = results_df[results_df['Model'] == 'Baseline Logistic Regression'].copy()
    lr_weighted = results_df[results_df['Model'] == 'Weighted Logistic Regression'].copy()
    lr_comparison = pd.concat([lr_baseline, lr_weighted], ignore_index=True)
    report.write("="*100 + "\n")
    report.write("LOGISTIC REGRESSION: BASELINE VS WEIGHTED\n")
    report.write("="*100 + "\n\n")
    report.write(lr_comparison.to_string(index=False))
    report.write("\n\n")
    
    # Summary table
    summary_list = []
    
    # Baseline best
    baseline_auc = baseline_df['AUC ROC'].astype(float)
    best_baseline_idx = baseline_auc.idxmax()
    summary_list.append({
        'Category': 'Best Baseline',
        'Model': baseline_df.loc[best_baseline_idx, 'Model'],
        'AUC ROC': baseline_df.loc[best_baseline_idx, 'AUC ROC'],
        'Average Precision': baseline_df.loc[best_baseline_idx, 'Average Precision'],
        'Balanced Accuracy': baseline_df.loc[best_baseline_idx, 'Balanced Accuracy']
    })
    
    # Best Balanced RF
    brf_auc = brf_df['AUC ROC'].astype(float)
    best_brf_idx = brf_auc.idxmax()
    summary_list.append({
        'Category': 'Best Balanced RF',
        'Model': brf_df.loc[best_brf_idx, 'Model'],
        'AUC ROC': brf_df.loc[best_brf_idx, 'AUC ROC'],
        'Average Precision': brf_df.loc[best_brf_idx, 'Average Precision'],
        'Balanced Accuracy': brf_df.loc[best_brf_idx, 'Balanced Accuracy']
    })
    
    # Best Weighted XGBoost
    wxgb_auc = wxgb_df['AUC ROC'].astype(float)
    best_wxgb_idx = wxgb_auc.idxmax()
    summary_list.append({
        'Category': 'Best Weighted XGBoost',
        'Model': wxgb_df.loc[best_wxgb_idx, 'Model'],
        'AUC ROC': wxgb_df.loc[best_wxgb_idx, 'AUC ROC'],
        'Average Precision': wxgb_df.loc[best_wxgb_idx, 'Average Precision'],
        'Balanced Accuracy': wxgb_df.loc[best_wxgb_idx, 'Balanced Accuracy']
    })
    
    # Weighted LR
    summary_list.append({
        'Category': 'Weighted LR',
        'Model': lr_weighted.loc[lr_weighted.index[0], 'Model'],
        'AUC ROC': lr_weighted.loc[lr_weighted.index[0], 'AUC ROC'],
        'Average Precision': lr_weighted.loc[lr_weighted.index[0], 'Average Precision'],
        'Balanced Accuracy': lr_weighted.loc[lr_weighted.index[0], 'Balanced Accuracy']
    })
    
    summary_df = pd.DataFrame(summary_list)
    report.write("="*100 + "\n")
    report.write("SUMMARY: BEST MODELS BY CATEGORY\n")
    report.write("="*100 + "\n\n")
    report.write(summary_df.to_string(index=False))
    report.write("\n\n")
    
    # Overall best model
    all_auc = results_df['AUC ROC'].astype(float)
    best_overall_idx = all_auc.idxmax()
    report.write("="*100 + "\n")
    report.write("OVERALL BEST MODEL (BY AUC ROC)\n")
    report.write("="*100 + "\n\n")
    report.write(f"Model: {results_df.loc[best_overall_idx, 'Model']}\n")
    report.write(f"AUC ROC: {results_df.loc[best_overall_idx, 'AUC ROC']}\n")
    report.write(f"Average Precision: {results_df.loc[best_overall_idx, 'Average Precision']}\n")
    report.write(f"Balanced Accuracy: {results_df.loc[best_overall_idx, 'Balanced Accuracy']}\n")
    report.write(f"Fit time: {results_df.loc[best_overall_idx, 'Fit time (s)']}s\n")
    report.write(f"Score time: {results_df.loc[best_overall_idx, 'Score time (s)']}s\n\n")
    
    report.write("="*100 + "\n")
    report.write("END OF REPORT\n")
    report.write("="*100 + "\n")
    
    report.close()
    
    print(f"\n✓ Ensemble methods report saved to: {report_path}")
    print(f"✓ All models saved to: models/")
    print(f"✓ All outputs saved to: outputs/")
    print(f"\nTotal models trained: {len(all_models)}")
    print(f"Best model: {results_df.loc[best_overall_idx, 'Model']}")
    print(f"Best AUC ROC: {results_df.loc[best_overall_idx, 'AUC ROC']}")


if __name__ == "__main__":
    main()
