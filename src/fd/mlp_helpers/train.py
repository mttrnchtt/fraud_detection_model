import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import json

from sklearn.neural_network import MLPClassifier

from .model import create_mlp_model, get_model_info
from .utils import load_all_data, set_seed
from .eval import compute_metrics


def compute_sample_weights(y_train: np.ndarray, pos_weight: float) -> np.ndarray:
    """
    Compute sample weights to handle class imbalance.
    
    Args:
        y_train: Training labels (0/1)
        pos_weight: Weight for positive class (e.g., 518.7)
        
    Returns:
        Sample weights array
    """
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 1] = pos_weight
    return sample_weights




def train_mlp_model(config: Dict[str, Any]) -> Tuple[MLPClassifier, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Train MLPClassifier model with full configuration support.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_model, training_results, predictions_dict)
    """
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(config)
    
    # Create model
    print("Creating model...")
    model = create_mlp_model(config)
    
    # Compute sample weights for class imbalance
    training_config = config['training']
    pos_weight = training_config['pos_weight']
    sample_weights = compute_sample_weights(y_train, pos_weight)
    
    print(f"Training model with sample weights (pos_weight={pos_weight})...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Class balance - Train: {y_train.sum()} pos, {len(y_train)-y_train.sum()} neg")
    
    # Train model (MLPClassifier handles early stopping internally)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Make predictions
    print("Making predictions...")
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    print("Computing metrics...")
    train_metrics = compute_metrics(y_train, train_pred, train_proba)
    val_metrics = compute_metrics(y_val, val_pred, val_proba)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    
    # Compile results
    results = {
        'model_info': get_model_info(model),
        'training_config': {
            'pos_weight': pos_weight,
            'sample_weights_used': True,
            'validation_split': 'manual',  # We used separate val set
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
        # Note: Predictions are saved separately as .npy files for efficiency
        'data_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'class_balance_train': {
                'positive': int(y_train.sum()),
                'negative': int(len(y_train) - y_train.sum()),
                'ratio': float(len(y_train) - y_train.sum()) / float(y_train.sum())
            }
        }
    }
    
    # Create predictions dict for efficient saving
    predictions = {
        'train_pred': train_pred,
        'train_proba': train_proba,
        'val_pred': val_pred,
        'val_proba': val_proba,
        'test_pred': test_pred,
        'test_proba': test_proba,
    }
    
    return model, results, predictions


def save_training_results(model: MLPClassifier, results: Dict[str, Any], config: Dict[str, Any], 
                         predictions: Dict[str, np.ndarray] = None) -> None:
    """
    Save trained model and results to disk.
    
    Args:
        model: Trained MLPClassifier
        results: Training results dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    
    # Create directories
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
    
    # Save predictions if requested
    if config['logging']['save_predictions'] and predictions is not None:
        pred_dir = Path(config['logging']['predictions_path'])
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions as numpy arrays for easy loading
        np.save(pred_dir / 'train_pred.npy', predictions['train_pred'])
        np.save(pred_dir / 'train_proba.npy', predictions['train_proba'])
        np.save(pred_dir / 'val_pred.npy', predictions['val_pred'])
        np.save(pred_dir / 'val_proba.npy', predictions['val_proba'])
        np.save(pred_dir / 'test_pred.npy', predictions['test_pred'])
        np.save(pred_dir / 'test_proba.npy', predictions['test_proba'])
        
        print(f"Predictions saved to: {pred_dir}")
    
    # Save config copy if requested
    if config['logging']['save_config_copy']:
        config_path = output_dir / 'config_copy.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Config copy saved to: {config_path}")


