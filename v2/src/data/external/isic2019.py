"""
src/data/external/isic2019.py
-----------------------------
ISIC 2019 Challenge dataset loader.

Reference: https://challenge2019.isic-archive.com/
Combines: HAM10000 + BCN20000 + MSK datasets, deduplicated.

25,331 dermoscopic images across 8 classes:
  MEL, NV, BCC, AK, BKL, DF, VASC, SCC

The 8 classes map almost 1:1 to HAM10000 (extra SCC; AK == akiec).

Setup (manual download, no registration required):
  1. Download ISIC_2019_Training_Input.zip from
     https://challenge.isic-archive.com/data/#2019
  2. Download ISIC_2019_Training_GroundTruth.csv from the same page.
  3. Extract images into: data/external/ISIC2019/ISIC_2019_Training_Input/
  4. Place CSV at: data/external/ISIC2019/ISIC_2019_Training_GroundTruth.csv

IMPORTANT: HAM10000 is *part of* ISIC 2019. If you evaluate a HAM10000-trained
model on the full ISIC 2019 set, you'll have train/test leakage. Use
filter_out_ham10k=True to exclude HAM10000 images by image_id prefix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.external.taxonomy import (
    HAM10K_TO_IDX,
    ISIC2019_NATIVE_CLASSES,
    ISIC2019_TO_HAM10K,
    ISIC2019_TO_UNIVERSAL,
    UNIVERSAL_TO_IDX,
)


def load_isic2019_metadata(
    data_dir: str | Path,
    filter_out_ham10k: bool = True,
) -> pd.DataFrame:
    """
    Load ISIC 2019 ground truth CSV and prepare for eval.

    Args:
        data_dir:           path to data/external/ISIC2019
        filter_out_ham10k:  drop images whose image_id starts with ISIC_00 (HAM10000)
                            to avoid evaluating a HAM10000-trained model on its own
                            training data.

    Returns DataFrame with columns:
        image_id          : ISIC_XXXXXXX
        image_path        : absolute path to image
        native_label      : MEL/NV/BCC/AK/BKL/DF/VASC/SCC (string)
        ham10k_label      : str (after merging SCC -> akiec)
        ham10k_idx        : int 0..6
        universal_label   : str
        universal_idx     : int 0..2
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "ISIC_2019_Training_GroundTruth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ISIC 2019 ground truth not found at {csv_path}.\n"
            f"See setup instructions in src/data/external/isic2019.py docstring."
        )

    df = pd.read_csv(csv_path)

    # ISIC 2019 CSV is one-hot: one column per class, value 1.0 for true class
    if "image" not in df.columns:
        raise ValueError(
            f"Expected 'image' column in ISIC 2019 CSV. Got: {df.columns.tolist()}"
        )
    df = df.rename(columns={"image": "image_id"})

    # Find the native one-hot label
    onehot_cols = [c for c in ISIC2019_NATIVE_CLASSES if c in df.columns]
    if len(onehot_cols) < 7:
        raise ValueError(
            f"Expected one-hot columns {ISIC2019_NATIVE_CLASSES} in CSV. "
            f"Got only: {onehot_cols}"
        )
    df["native_label"] = df[onehot_cols].idxmax(axis=1)

    # UNK rows (unknown class) in ISIC 2019 have all zeros; drop them
    n_unk = (df[onehot_cols].sum(axis=1) == 0).sum()
    if n_unk:
        print(f"  [ISIC2019] Dropping {n_unk} unlabelled (UNK) rows")
        df = df[df[onehot_cols].sum(axis=1) > 0].reset_index(drop=True)

    # Drop HAM10000 overlap if requested (default & recommended for honest eval)
    if filter_out_ham10k:
        # HAM10000 images use ISIC IDs in the form ISIC_002... and ISIC_003...
        # but the simplest safe heuristic: BCN20000 + MSK use ISIC_006+
        # Use the actual HAM10000 metadata if available — here we use prefix.
        # The cleanest approach is to load HAM10000 metadata and exclude those IDs.
        try:
            from src.data.ham10000 import load_ham10000
            # Try common HAM10000 paths
            for ham_path in ("data/raw", "../skin-lesion-vit/data/raw"):
                try:
                    train, val, test = load_ham10000(ham_path)
                    ham_ids = set(train["image_id"]) | set(val["image_id"]) | set(test["image_id"])
                    before = len(df)
                    df = df[~df["image_id"].isin(ham_ids)].reset_index(drop=True)
                    print(f"  [ISIC2019] Removed HAM10000 overlap: {before} -> {len(df)}")
                    break
                except FileNotFoundError:
                    continue
        except Exception as e:
            print(f"  [ISIC2019] Could not load HAM10000 to exclude overlap: {e}")
            print(f"  [ISIC2019] WARNING: HAM10000-trained eval may be optimistic")

    # Build image paths
    image_dir = data_dir / "ISIC_2019_Training_Input"
    if not image_dir.exists():
        image_dir = data_dir / "images"
    if not image_dir.exists():
        image_dir = data_dir
    df["image_path"] = df["image_id"].apply(lambda x: str(image_dir / f"{x}.jpg"))

    # Drop rows with missing files
    exists = df["image_path"].apply(lambda p: Path(p).exists())
    missing = (~exists).sum()
    if missing:
        print(f"  [ISIC2019] Dropping {missing} rows with missing image files")
    df = df[exists].reset_index(drop=True)

    # Apply taxonomy mapping
    df["ham10k_label"] = df["native_label"].map(ISIC2019_TO_HAM10K)
    df["ham10k_idx"] = df["ham10k_label"].map(HAM10K_TO_IDX)

    df["universal_label"] = df["native_label"].map(ISIC2019_TO_UNIVERSAL)
    df["universal_idx"] = df["universal_label"].map(UNIVERSAL_TO_IDX)

    return df


class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019.

    Args:
        df:         DataFrame from load_isic2019_metadata()
        transform:  Albumentations transform pipeline
        label_mode: 'ham10k' (7-class, default — best fit) or
                    'universal' (3-class)
    """

    def __init__(self, df: pd.DataFrame, transform=None, label_mode: str = "ham10k"):
        assert label_mode in ("ham10k", "universal")
        self.label_mode = label_mode
        self.transform = transform
        self.df = df.reset_index(drop=True)
        self.label_col = "ham10k_idx" if label_mode == "ham10k" else "universal_idx"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(row[self.label_col])