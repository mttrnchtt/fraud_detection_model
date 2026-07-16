#!/usr/bin/env python3
"""
Train the meta-MLP on stacked layer-0 scores.

Usage:
    python scripts/train_stack_mlp.py [-c configs/stack.yaml] [--option A|B|A_wLGBM|B_wLGBM] [--tune]
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

from fd.stack_helpers.train import train_mlp_model, save_training_results
from fd.mlp_helpers.eval import print_training_summary, evaluate_with_thresholds
from fd.stack_helpers.utils import OPTIONS, load_all_data


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def hyperparameter_tuning(config, option_cfg):
    """Grid over small architectures and learning rates on the meta-train val split,
    then retrain the best config on the full meta-train."""
    architectures = [[16, 8], [32, 16], [32]]
    learning_rates = [0.001, 0.003, 0.005]

    best_auprc, best_config = -1.0, None
    for hidden_dims in architectures:
        for lr in learning_rates:
            cfg = copy.deepcopy(config)
            cfg["model"]["hidden_dims"] = hidden_dims
            cfg["optimizer"]["lr"] = lr
            try:
                _, results, _ = train_mlp_model(cfg, option_cfg, tune=True)
            except Exception as e:
                print(f"  skipped hidden_dims={hidden_dims}, lr={lr}: {e}")
                continue
            auprc = results["metrics"]["val"]["auprc"]
            print(f"  hidden_dims={hidden_dims}, lr={lr}: val AUPRC={auprc:.4f}")
            if auprc > best_auprc:
                best_auprc, best_config = auprc, cfg
    if best_config is None:
        raise RuntimeError("No configuration trained successfully.")
    print(f"Best val AUPRC={best_auprc:.4f}, retraining on full meta-train")
    model, results, preds = train_mlp_model(best_config, option_cfg, tune=False)
    return model, results, preds, best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/stack.yaml")
    parser.add_argument("--option", choices=list(OPTIONS), default="A")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"Config not found: {args.config}")
    config = load_config(args.config)
    option_cfg = OPTIONS[args.option]
    print(f"Layer-0 option {args.option}: {list(option_cfg)}")

    if args.tune:
        model, results, predictions, config = hyperparameter_tuning(config, option_cfg)
    else:
        model, results, predictions = train_mlp_model(config, option_cfg, tune=False)

    print_training_summary(results)

    _, _, _, y_test = load_all_data(config)
    print("\nThreshold analysis (test set):")
    for key, m in evaluate_with_thresholds(y_test, predictions["test_proba"], args.thresholds).items():
        t = key.replace("threshold_", "")
        print(f"  {t}: F1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")

    save_training_results(model, results, config, predictions)
    test = results["metrics"]["test"]
    print(f"\nTest AUROC={test['auroc']:.4f} AUPRC={test['auprc']:.4f}")


if __name__ == "__main__":
    main()
