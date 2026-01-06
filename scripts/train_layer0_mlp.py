#!/usr/bin/env python3
"""
Train MLP baseline model for credit card fraud detection.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Train MLP model
3. Evaluate performance
4. Save results and artifacts

Usage:
    python -m scripts.train_layer0_mlp
    -c or --config [path]  # default configs/mlp.yaml
    --tune  # enable tuning 
    --thresholds  # default 0.1 0.2 0.3 0.4 0.5
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fd.mlp_helpers.train import train_mlp_model, save_training_results
from src.fd.mlp_helpers.eval import print_training_summary, evaluate_with_thresholds
from src.fd.mlp_helpers.utils import load_all_data
import copy


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def hyperparameter_tuning(config: dict):
    """
    Perform hyperparameter tuning using validation set.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Tuple of (best_model, best_results, best_predictions, best_config)
    """
    # Define hyperparameter search space
    hidden_dims_options = [
        [128, 64, 32],    # Smaller 3-layer
        [256, 128, 64],   # Known good
        [512, 256, 128],   # Larger 3-layer
        [384, 192, 96]   # In-between
    ]
    
    learning_rates = [0.01, 0.05, 0.1]
    
    best_val_auprc = 0
    best_model = None
    best_results = None
    best_predictions = None
    best_config = None
    
    total_combinations = len(hidden_dims_options) * len(learning_rates)
    current_combination = 0
    
    print(f"🔍 Testing {total_combinations} hyperparameter combinations...")
    print(f"Hidden layer options: {hidden_dims_options}")
    print(f"Learning rates: {learning_rates}")
    print(f"Monitor metric: {config['training']['early_stopping']['monitor']}")
    print()
    
    for hidden_dims in hidden_dims_options:
        for lr in learning_rates:
            current_combination += 1
            
            # Create modified config
            test_config = copy.deepcopy(config)
            test_config['model']['hidden_dims'] = hidden_dims
            test_config['optimizer']['lr'] = lr
            test_config['experiment']['name'] = f"tune_{current_combination}"
            
            print(f"[{current_combination}/{total_combinations}] Testing: hidden_dims={hidden_dims}, lr={lr}")
            
            try:
                # Train model with this configuration
                model, results, predictions = train_mlp_model(test_config)
                
                # Get validation AUPRC (our target metric)
                val_auprc = results['metrics']['val']['auprc']
                val_auroc = results['metrics']['val']['auroc']
                
                print(f"  → Val AUPRC: {val_auprc:.4f}, Val AUROC: {val_auroc:.4f}")
                
                # Check if this is the best configuration
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    best_model = model
                    best_results = results
                    best_predictions = predictions
                    best_config = test_config
                    print(f"  🎯 NEW BEST! Val AUPRC: {val_auprc:.4f}")
                else:
                    print(f"  📉 Not better (best: {best_val_auprc:.4f})")
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                
            print()
    
    print("="*60)
    print("🎯 HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    print(f"Best hidden_dims: {best_config['model']['hidden_dims']}")
    print(f"Best learning rate: {best_config['optimizer']['lr']}")
    print("="*60)
    
    return best_model, best_results, best_predictions, best_config


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train MLP baseline model for fraud detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/mlp.yaml',
        help='Path to configuration file (default: configs/mlp.yaml)'
    )
    
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help='Probability thresholds for evaluation (default: 0.1 0.25 0.5)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (tests different architectures and learning rates)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate config paths exist
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Print experiment info
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Architecture: {config['model']['hidden_dims']}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print(f"Epochs: {config['training']['epochs']}")
    print()
    
    try:
        # Check if hyperparameter tuning is requested
        if args.tune:
            print("🔍 Starting hyperparameter tuning...")
            best_model, best_results, best_predictions, best_config = hyperparameter_tuning(config)
        else:
            # Train single model
            print("🚀 Starting MLP training...")
            best_model, best_results, best_predictions = train_mlp_model(config)
            best_config = config  # Use original config for single model training
        
        # Print training summary
        print_training_summary(best_results)
        
        # Threshold analysis (always run)
        print(f"\n📊 Threshold Analysis (Test Set):")
        print("="*50)
        
        # Load test data for threshold analysis
        X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(config)
        
        # Analyze test set with specified thresholds
        threshold_results = evaluate_with_thresholds(
            y_test, 
            best_predictions['test_proba'], 
            args.thresholds
        )
        
        for threshold_key, metrics in threshold_results.items():
            threshold = threshold_key.replace('threshold_', '')
            print(f"Threshold {threshold}:")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print()
        
        # Save all results
        print("💾 Saving results...")
        # Use best_config (contains the optimal hyperparameters)
        save_config = best_config
        save_training_results(best_model, best_results, save_config, best_predictions)
        
        # Final summary
        test_metrics = best_results['metrics']['test']
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Final Test Performance:")
        print(f"   AUROC: {test_metrics['auroc']:.4f}")
        print(f"   AUPRC: {test_metrics['auprc']:.4f}")
        print(f"   F1:    {test_metrics['f1']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        
        print(f"\n📁 Outputs saved to:")
        print(f"   Model: {config['paths']['checkpoint_dir']}/model.joblib")
        print(f"   Results: {config['paths']['output_dir']}/training_results.json")
        print(f"   Predictions: {config['paths']['output_dir']}/preds/")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
