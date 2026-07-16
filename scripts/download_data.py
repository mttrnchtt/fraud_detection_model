#!/usr/bin/env python3
"""
Usage:
    python scripts/download_data.py [-c configs/data_prep.yaml]

Fetches the Kaggle Credit Card Fraud dataset via kagglehub and copies
creditcard.csv to the path configured in ``paths.raw_csv`` (default
``data/creditcard.csv``). Requires Kaggle credentials configured for kagglehub
(``~/.kaggle/kaggle.json`` or the KAGGLE_USERNAME / KAGGLE_KEY env vars).

If the file already exists it is left untouched.
"""

import argparse
import shutil
from pathlib import Path

from fd.data_prep.utils import load_config

DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dest = Path(cfg["paths"]["raw_csv"])
    if dest.exists():
        print(f"{dest} already exists, nothing to download.")
        return

    import kagglehub  # imported lazily so the rest of the pipeline needs no Kaggle creds

    print(f"Downloading {DATASET} via kagglehub ...")
    path = Path(kagglehub.dataset_download(DATASET))
    src = path / CSV_NAME
    if not src.exists():
        matches = list(path.rglob(CSV_NAME))
        if not matches:
            raise SystemExit(f"{CSV_NAME} not found in downloaded dataset at {path}")
        src = matches[0]

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"Copied {src} -> {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/data_prep.yaml")
    args = parser.parse_args()
    main(args.config)
