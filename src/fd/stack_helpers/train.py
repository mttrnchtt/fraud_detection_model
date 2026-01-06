import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import json

from sklearn.neural_network import MLPClassifier

from .model import create_mlp_model, get_model_info
from .utils import load_all_data, set_seed, run_layer0, split_meta_data
from .eval import compute_metrics


def compute_sample_weights(y_train: np.ndarray, pos_weight: float) -> np.ndarray:
    """
    Compute sample weights to handle class imbalance.
    """
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    sample_weights[y_train == 1] = pos_weight
    return sample_weights


def train_mlp_model(
    config: Dict[str, Any],
    option_cfg: Dict[str, str],
    tune: bool = False
) -> Tuple[MLPClassifier, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Train meta-level MLPClassifier on stacked layer-0 outputs.

    Args:
        config: Configuration dictionary for meta-MLP
        option_cfg: layer0_option_A/B dict mapping model_name -> model_path
        tune: if True, split meta-train into train/val for hyperparam tuning

    Returns:
        (trained_model, training_results, predictions_dict)
    """
    # Seed
    set_seed(config['experiment']['seed'])

    # 1) Load base train/test (raw features + labels)
    X_train_raw, y_train, X_test_raw, y_test = load_all_data(config)

    # 2) Build meta features from layer-0 models
    X_meta_train_full = run_layer0(option_cfg, X_train_raw)
    X_meta_test = run_layer0(option_cfg, X_test_raw)

    # 3) Optionally split meta train into train/val
    if tune:
        X_meta_train, y_train_meta, X_meta_val, y_val_meta = split_meta_data(
            X_meta_train_full,
            y_train,
            ratio=0.67
        )
    else:
        X_meta_train = X_meta_train_full
        y_train_meta = y_train
        X_meta_val = np.empty((0, X_meta_train_full.shape[1]), dtype=np.float32)
        y_val_meta = np.empty((0,), dtype=np.float32)

    # 4) Create meta-MLP model
    model = create_mlp_model(config)

    # 5) Sample weights for class imbalance (on meta-train labels)
    training_config = config['training']
    pos_weight = training_config['pos_weight']
    sample_weights = compute_sample_weights(y_train_meta, pos_weight)

    # 6) Train meta model on stacked features
    model.fit(X_meta_train, y_train_meta, sample_weight=sample_weights)

    # 7) Predict on train / val / test
    train_pred = model.predict(X_meta_train)
    train_proba = model.predict_proba(X_meta_train)[:, 1]

    if tune and X_meta_val.shape[0] > 0:
        val_pred = model.predict(X_meta_val)
        val_proba = model.predict_proba(X_meta_val)[:, 1]
    else:
        val_pred = np.empty((0,), dtype=np.float32)
        val_proba = np.empty((0,), dtype=np.float32)

    test_pred = model.predict(X_meta_test)
    test_proba = model.predict_proba(X_meta_test)[:, 1]

    # 8) Compute metrics
    train_metrics = compute_metrics(y_train_meta, train_pred, train_proba)
    val_metrics = compute_metrics(y_val_meta, val_pred, val_proba) if tune else None
    test_metrics = compute_metrics(y_test, test_pred, test_proba)

    # 9) Compile results
    results = {
        'model_info': get_model_info(model),
        'training_config': {
            'pos_weight': pos_weight,
            'sample_weights_used': True,
            'validation_split': 'manual' if tune else None,
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
        'data_info': {
            'train_samples': int(X_meta_train.shape[0]),
            'val_samples': int(X_meta_val.shape[0]),
            'test_samples': int(X_meta_test.shape[0]),
            'features': int(X_meta_train.shape[1]),
            'class_balance_train': {
                'positive': int(y_train_meta.sum()),
                'negative': int(len(y_train_meta) - y_train_meta.sum()),
                'ratio': float(len(y_train_meta) - y_train_meta.sum()) / float(y_train_meta.sum() + 1e-8),
            },
        },
    }

    # 10) Predictions dict
    predictions = {
        'train_pred': train_pred,
        'train_proba': train_proba,
        'val_pred': val_pred,
        'val_proba': val_proba,
        'test_pred': test_pred,
        'test_proba': test_proba,
    }

    return model, results, predictions


def save_training_results(
    model: MLPClassifier,
    results: Dict[str, Any],
    config: Dict[str, Any],
    predictions: Dict[str, np.ndarray] = None
) -> None:
    """
    Save trained model and results to disk.
    """
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = checkpoint_dir / 'model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Save predictions
    if config['logging']['save_predictions'] and predictions is not None:
        pred_dir = Path(config['logging']['predictions_path'])
        pred_dir.mkdir(parents=True, exist_ok=True)

        np.save(pred_dir / 'train_pred.npy', predictions['train_pred'])
        np.save(pred_dir / 'train_proba.npy', predictions['train_proba'])
        np.save(pred_dir / 'val_pred.npy', predictions['val_pred'])
        np.save(pred_dir / 'val_proba.npy', predictions['val_proba'])
        np.save(pred_dir / 'test_pred.npy', predictions['test_pred'])
        np.save(pred_dir / 'test_proba.npy', predictions['test_proba'])

        print(f"Predictions saved to: {pred_dir}")

    if config['logging']['save_config_copy']:
        import yaml
        config_path = output_dir / 'config_copy.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Config copy saved to: {config_path}")
