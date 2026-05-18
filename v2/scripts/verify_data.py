"""
scripts/verify_data.py
----------------------
Dry run: load HAM10000, check splits, print stats.
Run before any training phase to confirm data is wired up correctly.

Usage:
    python scripts/verify_data.py --data_dir D:\ML\skin-lesion-vit\data\raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import CLASS_NAMES, get_dataloaders, load_ham10000
from src.data.transforms import get_train_transform, get_val_transform


def main(data_dir: str) -> None:
    print("=" * 60)
    print("  HAM10000 DATA VERIFICATION")
    print("=" * 60)

    train_df, val_df, test_df = load_ham10000(data_dir, verbose=True)

    # Check every split has all 7 classes
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = set(CLASS_NAMES) - set(df["dx"].unique())
        if missing:
            print(f"WARNING: {name} is missing classes: {missing}")
        else:
            print(f"{name}: all 7 classes present ✓")

    # Smoke-test DataLoaders — one batch each
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        get_train_transform(224), get_val_transform(224),
        batch_size=8, num_workers=0,   # num_workers=0 for Windows fork safety
    )

    print("\nDataLoader batch shapes:")
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        images, labels = next(iter(loader))
        print(f"  {name}: images {tuple(images.shape)}  labels {tuple(labels.shape)}")

    print("\nVerification complete — data is ready for training.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=r"D:\ML\skin-lesion-vit\data\raw",
        help="Path to folder containing HAM10000_metadata.csv",
    )
    args = parser.parse_args()
    main(args.data_dir)