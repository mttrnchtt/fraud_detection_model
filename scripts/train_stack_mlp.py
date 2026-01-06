#!/usr/bin/env python3
"""
Train MLP meta-model for stacked fraud detection.

This script trains an MLPClassifier on top of layer-0 model outputs
(MLP, XGB, RF, LR, Isolation Forest, etc.).

Pipeline:
1. Load configuration
2. Build meta-features from layer-0 models (stacked predictions)
3. (Optional) Hyperparameter tuning on a split of meta-train (B_train / B_val)
4. Retrain best meta-MLP on full meta-train (B)
5. Evaluate on test set (C)
6. Save results and artifacts

Usage:
    python -m scripts.train_stack_mlp
      -c or --config [path]           # default configs/meta_mlp.yaml
      --option [A|B]                  # which layer-0 set to use (default: A)
      --tune                          # enable hyperparameter tuning
      --thresholds 0.1 0.25 0.5       # thresholds for test analysis
"""

import argparse
import sys
from pathlib import Path
import copy
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fd.stack_helpers.train import train_mlp_model, save_training_results
from src.fd.mlp_helpers.eval import print_training_summary, evaluate_with_thresholds
from src.fd.stack_helpers.utils import load_all_data, layer0_option_A, layer0_option_B, layer0_option_A_wLGBM, layer0_option_B_wLGBM


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def hyperparameter_tuning(config: dict, option_cfg: dict[str, str]):
    """
    Perform hyperparameter tuning for the meta-MLP using a split
    inside the meta-train (B -> B_meta_train + B_meta_val).

    Args:
        config: base configuration dictionary
        option_cfg: layer0_option_A/B dict

    Returns:
        (final_model, final_results, final_predictions, best_config)
        where final_* correspond to a model retrained on full meta-train (B)
        with the best hyperparameters.
    """
    # Small, reasonable search space for meta-MLP
    hidden_dims_options = [
        [16, 8],
        [32, 16],
        [32],
    ]
    learning_rates = [0.001, 0.003, 0.005]

    best_val_auprc = -1.0
    best_config = None

    total_combinations = len(hidden_dims_options) * len(learning_rates)
    current_combination = 0

    print(f"🔍 Testing {total_combinations} hyperparameter combinations for meta-MLP...")
    print(f"Hidden layer options: {hidden_dims_options}")
    print(f"Learning rates: {learning_rates}")
    print()

    for hidden_dims in hidden_dims_options:
        for lr in learning_rates:
            current_combination += 1

            test_config = copy.deepcopy(config)
            test_config["model"]["hidden_dims"] = hidden_dims
            test_config["optimizer"]["lr"] = lr
            test_config["experiment"]["name"] = f"meta_tune_{current_combination}"

            print(f"[{current_combination}/{total_combinations}] "
                  f"Testing meta-MLP: hidden_dims={hidden_dims}, lr={lr}")

            try:
                # tune=True -> train_mlp_model will split meta-train into train/val
                model, results, preds = train_mlp_model(test_config, option_cfg, tune=True)

                if results["metrics"]["val"] is None:
                    raise RuntimeError("Validation metrics missing during tuning.")

                val_auprc = results["metrics"]["val"]["auprc"]
                val_auroc = results["metrics"]["val"]["auroc"]

                print(f"  → Val AUPRC: {val_auprc:.4f}, Val AUROC: {val_auroc:.4f}")

                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    best_config = test_config
                    print(f"  🎯 NEW BEST! Val AUPRC: {val_auprc:.4f}")
                else:
                    print(f"  📉 Not better (best: {best_val_auprc:.4f})")

            except Exception as e:
                print(f"  ❌ Failed: {e}")

            print()

    print("=" * 60)
    print("🎯 META-MLP HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    if best_config is not None:
        print(f"Best hidden_dims: {best_config['model']['hidden_dims']}")
        print(f"Best learning rate: {best_config['optimizer']['lr']}")
    print("=" * 60)

    if best_config is None:
        raise RuntimeError("No successful hyperparameter configuration found for meta-MLP.")

    # 🔁 Retrain best meta-MLP on FULL meta-train (B), no split:
    print("🚀 Retraining best meta-MLP on full meta-train (B)...")
    final_model, final_results, final_preds = train_mlp_model(
        best_config,
        option_cfg,
        tune=False
    )

    return final_model, final_results, final_preds, best_config


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP meta-model (stacking) for fraud detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/stack.yaml",
        help="Path to configuration file for meta-MLP (default: configs/stack.yaml)"
    )

    parser.add_argument(
        "--option",
        type=str,
        choices=["A", "B", "A_wLGBM", "B_wLGBM"],
        default="A",
        help="Which layer-0 model set to use: A or B (default: A)"
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5],
        help="Probability thresholds for evaluation on test set "
             "(default: 0.1 0.25 0.5)"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning for the meta-MLP"
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(args.config)

    # Choose layer-0 option
    if args.option == "A":
        option_cfg = layer0_option_A
    elif args.option == "B":
        option_cfg = layer0_option_B
    elif args.option == "A_wLGBM":
        option_cfg = layer0_option_A_wLGBM
    elif args.option == "B_wLGBM":
        option_cfg = layer0_option_B_wLGBM

    print(f"Meta-MLP Experiment: {config['experiment']['name']}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Initial architecture: {config['model']['hidden_dims']}")
    print(f"Initial learning rate: {config['optimizer']['lr']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Using layer-0 option: {args.option} -> models: {list(option_cfg.keys())}")
    print()

    try:
        # Train meta-MLP (with or without tuning)
        if args.tune:
            print("🔍 Starting meta-MLP hyperparameter tuning...")
            best_model, best_results, best_predictions, best_config = hyperparameter_tuning(
                config,
                option_cfg
            )
        else:
            print("🚀 Starting meta-MLP training (no tuning)...")
            best_model, best_results, best_predictions = train_mlp_model(
                config,
                option_cfg,
                tune=False
            )
            best_config = config

        # Print training summary
        print_training_summary(best_results)

        # Threshold analysis on TEST (C)
        print("\n📊 Threshold Analysis (Test Set):")
        print("=" * 50)

        # Reload y_test for threshold evaluation (raw X is not needed here)
        _, _, X_test_raw, y_test = load_all_data(config)

        threshold_results = evaluate_with_thresholds(
            y_test,
            best_predictions["test_proba"],
            args.thresholds
        )

        for threshold_key, metrics in threshold_results.items():
            threshold = threshold_key.replace("threshold_", "")
            print(f"Threshold {threshold}:")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print()

        # Save results with best_config (has best hyperparams if tuned)
        print("💾 Saving meta-MLP results...")
        save_training_results(best_model, best_results, best_config, best_predictions)

        # Final summary
        test_metrics = best_results["metrics"]["test"]
        print("\n🎉 Meta-MLP training completed successfully!")
        print("📈 Final Test Performance (stacked meta-model):")
        print(f"   AUROC:     {test_metrics['auroc']:.4f}")
        print(f"   AUPRC:     {test_metrics['auprc']:.4f}")
        print(f"   F1:        {test_metrics['f1']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")

        print("\n📁 Outputs saved to:")
        print(f"   Model:    {best_config['paths']['checkpoint_dir']}/model.joblib")
        print(f"   Results:  {best_config['paths']['output_dir']}/training_results.json")
        print(f"   Preds:    {best_config['logging']['predictions_path']}/")

    except Exception as e:
        print(f"❌ Meta-MLP training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
