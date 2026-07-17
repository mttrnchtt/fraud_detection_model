#!/usr/bin/env python3
"""
Train the base MLP.

Usage:
    python scripts/train_layer0_mlp.py [-c configs/mlp.yaml] [--tune] [--thresholds 0.1 0.25 0.5]
"""

import argparse
import copy
import sys
from pathlib import Path

from fd.data_prep.utils import load_config
from fd.mlp_helpers.train import train_mlp_model, save_training_results
from fd.mlp_helpers.eval import print_training_summary, evaluate_with_thresholds
from fd.mlp_helpers.utils import load_all_data


def hyperparameter_tuning(config):
    """Grid over a few architectures and learning rates, keep the best val AUPRC."""
    architectures = [[128, 64, 32], [256, 128, 64], [512, 256, 128], [384, 192, 96]]
    learning_rates = [0.01, 0.05, 0.1]

    best_auprc, best = 0.0, None
    for hidden_dims in architectures:
        for lr in learning_rates:
            cfg = copy.deepcopy(config)
            cfg["model"]["hidden_dims"] = hidden_dims
            cfg["optimizer"]["lr"] = lr
            try:
                model, results, preds = train_mlp_model(cfg)
            except Exception as e:
                print(f"  skipped hidden_dims={hidden_dims}, lr={lr}: {e}")
                continue
            auprc = results["metrics"]["val"]["auprc"]
            print(f"  hidden_dims={hidden_dims}, lr={lr}: val AUPRC={auprc:.4f}")
            if auprc > best_auprc:
                best_auprc, best = auprc, (model, results, preds, cfg)
    if best is None:
        raise RuntimeError("No configuration trained successfully.")
    print(f"Best val AUPRC={best_auprc:.4f} with hidden_dims={best[3]['model']['hidden_dims']}, "
          f"lr={best[3]['optimizer']['lr']}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/mlp.yaml")
    parser.add_argument("--tune", action="store_true", help="grid-search architecture and learning rate")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"Config not found: {args.config}")
    config = load_config(args.config)

    if args.tune:
        model, results, predictions, config = hyperparameter_tuning(config)
    else:
        model, results, predictions = train_mlp_model(config)

    print_training_summary(results)

    _, _, _, _, _, y_test = load_all_data(config)
    print("\nThreshold analysis (test set):")
    for key, m in evaluate_with_thresholds(y_test, predictions["test_proba"], args.thresholds).items():
        t = key.replace("threshold_", "")
        print(f"  {t}: F1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")

    save_training_results(model, results, config, predictions)
    test = results["metrics"]["test"]
    print(f"\nTest AUROC={test['auroc']:.4f} AUPRC={test['auprc']:.4f}")


if __name__ == "__main__":
    main()
