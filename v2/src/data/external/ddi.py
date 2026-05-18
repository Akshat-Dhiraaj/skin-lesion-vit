"""
src/data/external/ddi.py
------------------------
Stanford Diverse Dermatology Images (DDI) dataset loader.

Reference: https://ddi-dataset.github.io/

656 biopsy-proven images across all 6 Fitzpatrick skin tones.
Each image labelled with:
  - DDI_file       : filename
  - malignant      : True/False  (binary clinical label)
  - disease        : specific diagnosis (string)
  - skin_tone      : Fitzpatrick 12 / 34 / 56 (stringified pair)

Setup (manual download required):
  1. Register at https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965
  2. Accept the data use agreement.
  3. Download DDI archive, extract to: data/external/DDI/
  4. Verify the metadata CSV exists at: data/external/DDI/ddi_metadata.csv
  5. Verify images live at: data/external/DDI/images/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.external.taxonomy import (
    HAM10K_TO_IDX,
    UNIVERSAL_TO_IDX,
    bucket_ddi_skintone,
    ddi_to_ham10k,
    ddi_to_universal,
)


def load_ddi_metadata(data_dir: str | Path) -> pd.DataFrame:
    """
    Load DDI metadata CSV and prepare it for evaluation.

    Returns a DataFrame with columns:
        image_path        : absolute path to image
        disease           : raw disease string from DDI
        malignant         : True/False
        universal_label   : malignant / benign / non_neoplastic
        universal_idx     : 0 / 1 / 2 (index into UNIVERSAL_CLASSES)
        ham10k_label      : str or None (only for diseases that map)
        ham10k_idx        : int or -1 if no HAM10000 mapping
        skin_tone_raw     : raw '12'/'34'/'56' string
        fitz_bucket       : 'light' / 'medium' / 'dark'
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "ddi_metadata.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"DDI metadata not found at {csv_path}.\n"
            f"See setup instructions in src/data/external/ddi.py docstring."
        )

    df = pd.read_csv(csv_path)

    # Column names vary between DDI releases — standardise
    rename = {}
    for src, dst in [
        ("DDI_file",  "image_id"),
        ("DDI_FILE",  "image_id"),
        ("filename",  "image_id"),
        ("skin_tone", "skin_tone_raw"),
        ("skin tone", "skin_tone_raw"),
        ("disease",   "disease"),
    ]:
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename)

    if "image_id" not in df.columns:
        raise ValueError(
            f"DDI CSV missing an 'image_id'/'DDI_file' column. Columns: {df.columns.tolist()}"
        )

    # Build absolute image paths
    image_dir = data_dir / "images"
    if not image_dir.exists():
        # Fallback: images sometimes sit directly under data_dir
        image_dir = data_dir
    df["image_path"] = df["image_id"].apply(lambda x: str(image_dir / x))

    # Drop rows whose image files don't actually exist on disk
    exists = df["image_path"].apply(lambda p: Path(p).exists())
    missing = (~exists).sum()
    if missing:
        print(f"  [DDI] Dropping {missing} rows with missing image files")
    df = df[exists].reset_index(drop=True)

    # Apply taxonomy mapping
    df["malignant"] = df["malignant"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["universal_label"] = df.apply(
        lambda r: ddi_to_universal(r["malignant"], r.get("disease", "")), axis=1
    )
    df["universal_idx"] = df["universal_label"].map(UNIVERSAL_TO_IDX)

    df["ham10k_label"] = df.apply(
        lambda r: ddi_to_ham10k(r["malignant"], r.get("disease", "")), axis=1
    )
    df["ham10k_idx"] = df["ham10k_label"].map(lambda x: HAM10K_TO_IDX.get(x, -1))

    # Skin tone bucket
    if "skin_tone_raw" in df.columns:
        df["fitz_bucket"] = df["skin_tone_raw"].apply(bucket_ddi_skintone)
    else:
        df["fitz_bucket"] = None

    return df


class DDIDataset(Dataset):
    """
    PyTorch Dataset for DDI evaluation.

    Args:
        df:        DataFrame from load_ddi_metadata()
        transform: Albumentations transform pipeline (use get_val_transform)
        label_mode: 'universal' (3-class) or 'ham10k' (7-class).
                    'ham10k' filters out rows with no HAM10000 mapping.
    """

    def __init__(self, df: pd.DataFrame, transform=None, label_mode: str = "universal"):
        assert label_mode in ("universal", "ham10k")
        self.label_mode = label_mode
        self.transform = transform

        if label_mode == "ham10k":
            df = df[df["ham10k_idx"] >= 0].reset_index(drop=True)
            self.label_col = "ham10k_idx"
        else:
            self.label_col = "universal_idx"

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(row[self.label_col])