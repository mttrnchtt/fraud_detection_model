#!/usr/bin/env python3
"""
Train LightGBM model for credit card fraud detection.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Train LightGBM model (with optional extensive tuning)
3. Evaluate performance
4. Save results and artifacts

Usage:
    python -m scripts.train_lightgbm
    -c or --config [path]  # default configs/lightgbm.yaml
    --tune  # enable tuning 
    --thresholds  # default 0.1 0.25 0.5
"""

import argparse
import sys
import json
import joblib
import numpy as np
import yaml
import copy
from pathlib import Path
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

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

def train_lightgbm(config: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train LightGBM model."""
    model_params = config['model']
    train_params = config['training']
    
    # Prepare LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Filter params for LGBM
    lgbm_params = {k: v for k, v in model_params.items() if k not in ['n_estimators']} # n_estimators handled by num_boost_round
    
    # Add callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=train_params['early_stopping_rounds']),
        lgb.log_evaluation(period=train_params['verbose_eval'])
    ]
    
    model = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=model_params.get('n_estimators', 1000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    return model

def hyperparameter_tuning(config: dict, X_train, y_train, X_val, y_val):
    """Perform extensive random search for hyperparameters."""
    print("Starting extensive hyperparameter tuning...")
    
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
            # Train with early stopping
            model = train_lightgbm(temp_config, X_train, y_train, X_val, y_val)
            
            # Evaluate on validation set
            val_proba = model.predict(X_val)
            val_auprc = average_precision_score(y_val, val_proba)
            val_auroc = roc_auc_score(y_val, val_proba)
            
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
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument('--config', '-c', type=str, default='configs/lightgbm.yaml', help='Path to config file')
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
        print("Starting LightGBM training...")
        model = train_lightgbm(config, X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nEvaluating on Test Set...")
    test_proba = model.predict(X_test)
    
    metrics_dict = evaluate_with_thresholds(y_test, test_proba, args.thresholds)
    
    # Print results
    print(f"\nFinal Test Performance:")
    test_auroc = roc_auc_score(y_test, test_proba)
    test_auprc = average_precision_score(y_test, test_proba)
    print(f"   AUROC: {test_auroc:.4f}")
    print(f"   AUPRC: {test_auprc:.4f}")
    
    for threshold_key, metrics in metrics_dict.items():
        threshold = threshold_key.replace('threshold_', '')
        print(f"\nThreshold {threshold}:")
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
    np.save(f"{config['logging']['predictions_path']}/test_proba.npy", test_proba)
    
    print(f"\nOutputs saved to:")
    print(f"   Model: {config['paths']['checkpoint_dir']}/model.joblib")
    print(f"   Results: {config['paths']['output_dir']}/training_results.json")
    print(f"   Predictions: {config['logging']['predictions_path']}")

if __name__ == "__main__":
    main()
