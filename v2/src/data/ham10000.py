"""
src/data/ham10000.py
--------------------
HAM10000 dataset loader.

Changes vs the original skin-lesion-vit repo
---------------------------------------------
- Splitting is done via split_dataset() with group_col='lesion_id' so no
  lesion leaks across train/val/test.
- WeightedRandomSampler weight calculation moved here so it's testable.
- Image path discovery is factored out so the path layout can be changed
  without touching the Dataset class.
- load_ham10000() is the single public entry point: call it, get back
  (train_df, val_df, test_df) that are ready for HAM10000Dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.splits import (
    assert_no_group_leakage,
    report_split_stats,
    split_ham10000,
    split_summary_string,
)

# ── Class registry ────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)


# ── Path discovery ────────────────────────────────────────────────────────────

def _build_image_path_map(data_dir: str | Path) -> dict[str, str]:
    """
    Walk the two HAM10000 image folders and return image_id → absolute path.
    Handles the standard Kaggle layout:
        data_dir/
            HAM10000_images_part_1/
            HAM10000_images_part_2/
    Also falls back gracefully if images are all in one folder.
    """
    data_dir = Path(data_dir)
    candidate_dirs = [
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        data_dir,  # flat layout fallback
    ]

    path_map: dict[str, str] = {}
    for d in candidate_dirs:
        if not d.exists():
            continue
        for fname in d.iterdir():
            if fname.suffix.lower() == ".jpg":
                path_map[fname.stem] = str(fname)

    return path_map


# ── Metadata loader ───────────────────────────────────────────────────────────

def load_ham10000(
    data_dir: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load HAM10000 metadata and return lesion-level-split DataFrames.

    Args:
        data_dir: Path to the folder containing HAM10000_metadata.csv and
            the two image subdirectories.
        val_frac: Target validation fraction.
        test_frac: Target test fraction.
        random_state: Seed passed to split_ham10000().
        verbose: If True, print split summary and class distribution.

    Returns:
        (train_df, val_df, test_df) — each has columns:
            image_id, lesion_id, dx, dx_type, age, sex, localization,
            image_path, label   (integer 0-6)

    Raises:
        FileNotFoundError: if metadata CSV or image dirs are missing.
        AssertionError: if leakage is detected post-split (should never fire).
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Attach image paths
    path_map = _build_image_path_map(data_dir)
    df["image_path"] = df["image_id"].map(path_map)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows have no matching image file — dropping.")
    df = df.dropna(subset=["image_path"]).copy()

    # Integer label
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    unknown_dx = df["label"].isna()
    if unknown_dx.any():
        bad = df.loc[unknown_dx, "dx"].unique().tolist()
        raise ValueError(f"Unknown dx values found (not in CLASS_NAMES): {bad}")
    df["label"] = df["label"].astype(int)

    # HAM10000 has lesion_id natively — use it for grouping
    if "lesion_id" not in df.columns:
        raise ValueError("'lesion_id' column not found in metadata CSV.")

    if verbose:
        print(f"Loaded {len(df)} images  |  {df['lesion_id'].nunique()} unique lesions")
        print(f"Class distribution:\n{df['dx'].value_counts().to_string()}\n")

    train_df, val_df, test_df = split_ham10000(
        df, val_frac=val_frac, test_frac=test_frac, random_state=random_state
    )

    # Leakage guard — should never fire, but cheap insurance
    assert_no_group_leakage(train_df, val_df, test_df, group_col="lesion_id")

    if verbose:
        print(split_summary_string(train_df, val_df, test_df, "dx", "lesion_id"))
        stats = report_split_stats(train_df, val_df, test_df, "dx", "lesion_id")
        print("\nPer-class counts:")
        print(stats.to_string(index=False))
        print()

    return train_df, val_df, test_df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Expects a DataFrame produced by load_ham10000() with at minimum:
        image_path (str), label (int 0-6)

    Args:
        df: Split DataFrame from load_ham10000().
        transform: Albumentations transform pipeline. Receives a numpy
            uint8 HWC array, returns a dict with key 'image'.
    """

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, int(row["label"])


# ── Sampler ───────────────────────────────────────────────────────────────────

def get_weighted_sampler(df: pd.DataFrame, label_col: str = "label") -> WeightedRandomSampler:
    """
    WeightedRandomSampler that up-samples rare classes.

    Weight for each sample = 1 / count_of_its_class.
    Draws len(df) samples with replacement — one full epoch per call.
    """
    counts = df[label_col].value_counts().to_dict()
    weights = [1.0 / counts[lbl] for lbl in df[label_col]]
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for all three splits.
    Training uses WeightedRandomSampler; val/test use sequential order.
    """
    train_ds = HAM10000Dataset(train_df, transform=train_transform)
    val_ds   = HAM10000Dataset(val_df,   transform=val_transform)
    test_ds  = HAM10000Dataset(test_df,  transform=val_transform)

    sampler = get_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader