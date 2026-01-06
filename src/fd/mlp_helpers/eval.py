import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    return {
        'auroc': roc_auc_score(y_true, y_proba),
        'auprc': average_precision_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def print_training_summary(results: Dict) -> None:
    """
    Print a summary of training results.
    
    Args:
        results: Training results dictionary
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Model info
    model_info = results['model_info']
    print(f"Model: MLPClassifier")
    print(f"Architecture: {model_info['hidden_layer_sizes']}")
    print(f"Activation: {model_info['activation']}")
    print(f"Solver: {model_info['solver']}")
    print(f"Learning Rate: {model_info['learning_rate_init']}")
    
    # Data info
    data_info = results['data_info']
    print(f"\nData:")
    print(f"  Training samples: {data_info['train_samples']:,}")
    print(f"  Validation samples: {data_info['val_samples']:,}")
    print(f"  Test samples: {data_info['test_samples']:,}")
    print(f"  Features: {data_info['features']}")
    print(f"  Class ratio (neg:pos): {data_info['class_balance_train']['ratio']:.1f}:1")
    
    # Metrics
    print(f"\nMetrics:")
    for split in ['train', 'val', 'test']:
        metrics = results['metrics'][split]
        print(f"  {split.upper()}:")
        
        if metrics is None:
            print("    (Not available)")
            continue

        print(f"    AUROC: {metrics['auroc']:.4f}")
        print(f"    AUPRC: {metrics['auprc']:.4f}")
        print(f"    F1:    {metrics['f1']:.4f}")
        print(f"    Prec:  {metrics['precision']:.4f}")
        print(f"    Rec:   {metrics['recall']:.4f}")
        print(f"    Acc:   {metrics['accuracy']:.4f}")
    
    print("="*60)


def evaluate_with_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list = None) -> dict:
    """
    Evaluate model performance at different probability thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate (default: [0.1, 0.25, 0.5])
        
    Returns:
        Dictionary with metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.25, 0.5]
    
    results = {}
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        results[f'threshold_{threshold}'] = metrics
    
    return results
