#!/usr/bin/env python3
"""
Train Isolation Forest model for credit card fraud detection.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Train Isolation Forest model
3. Evaluate performance
4. Save results and artifacts

Usage:
    python -m scripts.isolation_forest
    -c or --config [path]  # default configs/isolation_forest.yaml
    --tune  # enable tuning 
    --thresholds  # default 0.1 0.25 0.5
"""

import argparse
import sys
import json
import joblib
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fd.mlp_helpers.utils import load_all_data

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute comprehensive metrics for binary classification."""
    return {
        'auroc': float(roc_auc_score(y_true, y_proba)),
        'auprc': float(average_precision_score(y_true, y_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

def evaluate_with_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list = None) -> dict:
    """Evaluate model performance at different probability thresholds."""
    if thresholds is None:
        thresholds = [0.1, 0.25, 0.5]
    
    results = {}
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        results[f'threshold_{threshold}'] = metrics
    
    return results

def train_isolation_forest(config: dict, X_train: np.ndarray, y_train: np.ndarray = None):
    """Train Isolation Forest model."""
    model_params = config['model']
    # Remove non-sklearn params if any
    sklearn_params = {k: v for k, v in model_params.items() if k in [
        'n_estimators', 'contamination', 'max_samples', 'max_features', 'bootstrap', 'n_jobs', 'random_state'
    ]}
    
    model = IsolationForest(**sklearn_params)
    model.fit(X_train) # IF is unsupervised, ignores y_train
    
    return model

def get_scores(model, X):
    """Get anomaly scores. 
    IsolationForest returns anomaly score as decision_function.
    Lower is more abnormal. We want higher = more abnormal (probability-like).
    score_samples returns the opposite of the anomaly score defined in the paper.
    The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
    The measure of normality of an observation given a tree depth is equivalent to the path length of the observation.
    
    sklearn's score_samples: "Opposite of the anomaly score defined in the original paper."
    sklearn's decision_function: "Average anomaly score of X of the base classifiers."
    
    We will use -score_samples to get a score where higher means more anomalous.
    However, for consistency with other models that output [0, 1] probability, we might want to normalize or just use it as a raw score.
    AUPRC/AUROC work with raw scores.
    """
    # score_samples returns negative values for outliers, positive for inliers?
    # No, score_samples returns the opposite of the anomaly score.
    # The lower, the more abnormal.
    # So -score_samples is higher for more abnormal.
    return -model.score_samples(X)

def hyperparameter_tuning(config: dict, X_train, y_train, X_val, y_val):
    """Perform random search for hyperparameters."""
    print("Starting hyperparameter tuning...")
    
    param_grid = config['tuning']['param_grid']
    n_iter = config['tuning']['n_iter']
    
    # Generate parameter combinations
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=config['experiment']['seed']))
    
    best_val_auprc = -1
    best_model = None
    best_config = None
    
    print(f"Testing {len(param_list)} combinations...")
    
    for i, params in enumerate(param_list):
        print(f"[{i+1}/{len(param_list)}] Testing: {params}")
        
        # Create temp config
        temp_config = copy.deepcopy(config)
        temp_config['model'].update(params)
        
        try:
            model = train_isolation_forest(temp_config, X_train)
            
            # Evaluate on validation set
            val_scores = get_scores(model, X_val)
            val_auprc = average_precision_score(y_val, val_scores)
            val_auroc = roc_auc_score(y_val, val_scores)
            
            print(f"  -> Val AUPRC: {val_auprc:.4f}, Val AUROC: {val_auroc:.4f}")
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model = model
                best_config = temp_config
                print(f"  NEW BEST! Val AUPRC: {val_auprc:.4f}")
            else:
                print(f"  Not better (best: {best_val_auprc:.4f})")
                
        except Exception as e:
            print(f"  Failed: {e}")
            
    print("="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    print("="*60)
    
    return best_model, best_config

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest model")
    parser.add_argument('--config', '-c', type=str, default='configs/isolation_forest.yaml', help='Path to config file')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.1, 0.25, 0.5], help='Thresholds for evaluation')
    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(config)
    
    # Train or Tune
    if args.tune:
        model, best_config = hyperparameter_tuning(config, X_train, y_train, X_val, y_val)
        config = best_config # Update config with best params
    else:
        print("Starting Isolation Forest training...")
        model = train_isolation_forest(config, X_train)

    # Evaluate
    print("\nEvaluating on Test Set...")
    test_scores = get_scores(model, X_test)
    
    # Since IF scores are not probabilities [0,1], we need to be careful with thresholds.
    # However, for the sake of the report, we can normalize them or just report AUROC/AUPRC which are threshold-independent.
    # For hard metrics (F1, etc), we need a threshold.
    # We can find a threshold that gives a certain percentile or just use the raw scores if they happen to be in a reasonable range (unlikely).
    # A common approach for IF is to use the contamination parameter to determine the threshold on the training set.
    # But here we are evaluating with specific thresholds provided by user/default.
    # Let's normalize scores to [0, 1] using MinMax based on Test set (or combined) for reporting purposes? 
    # Or better, just report AUROC/AUPRC as primary metrics.
    
    # For the purpose of this specific request "report nicely like other models", I will calculate metrics.
    # But standard thresholds like 0.5 might not make sense for raw anomaly scores.
    # I'll normalize the scores to [0, 1] for the threshold evaluation to make sense.
    
    # Normalize scores for threshold evaluation
    # Note: This is a simplification. In production you'd save the scaler.
    min_score = test_scores.min()
    max_score = test_scores.max()
    test_proba = (test_scores - min_score) / (max_score - min_score)
    
    metrics_dict = evaluate_with_thresholds(y_test, test_proba, args.thresholds)
    
    # Print results
    print(f"\nFinal Test Performance:")
    test_auroc = roc_auc_score(y_test, test_scores)
    test_auprc = average_precision_score(y_test, test_scores)
    print(f"   AUROC: {test_auroc:.4f}")
    print(f"   AUPRC: {test_auprc:.4f}")
    
    for threshold_key, metrics in metrics_dict.items():
        threshold = threshold_key.replace('threshold_', '')
        print(f"\nThreshold {threshold} (Normalized Scores):")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Save results
    print("\nSaving results...")
    ensure_dir(config['paths']['checkpoint_dir'])
    ensure_dir(config['paths']['output_dir'])
    ensure_dir(config['logging']['predictions_path'])
    
    # Save model
    joblib.dump(model, f"{config['paths']['checkpoint_dir']}/model.joblib")
    
    # Save metrics
    results = {
        'config': config,
        'metrics': {
            'test': {
                'auroc': test_auroc,
                'auprc': test_auprc,
                'threshold_metrics': metrics_dict
            }
        }
    }
    
    with open(f"{config['paths']['output_dir']}/training_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save predictions
    np.save(f"{config['logging']['predictions_path']}/test_scores.npy", test_scores)
    np.save(f"{config['logging']['predictions_path']}/test_proba_normalized.npy", test_proba)
    
    print(f"\nOutputs saved to:")
    print(f"   Model: {config['paths']['checkpoint_dir']}/model.joblib")
    print(f"   Results: {config['paths']['output_dir']}/training_results.json")
    print(f"   Predictions: {config['logging']['predictions_path']}")

if __name__ == "__main__":
    main()
