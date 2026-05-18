"""
src/data/external/fitzpatrick17k.py
-----------------------------------
Fitzpatrick17k dataset loader.

Reference: https://github.com/mattgroh/fitzpatrick17k
Paper: Groh et al., "Evaluating Deep Neural Networks Trained on Clinical
       Images in Dermatology with the Fitzpatrick 17k Dataset" (CVPR 2021)

16,577 clinical photographs labelled with:
  - md5hash             : image identifier
  - fitzpatrick         : Fitzpatrick scale 1-6 (raw)
  - fitzpatrick_centaur : alternative annotation
  - label               : 114 specific conditions
  - nine_partition_label
  - three_partition_label : {benign, malignant, non-neoplastic}
  - qc                  : 1 if quality-controlled (use only qc==1)
  - url                 : download URL (some dead)

Setup (manual download required — images NOT in the repo):
  1. Clone: git clone https://github.com/mattgroh/fitzpatrick17k.git
  2. The metadata CSV is at fitzpatrick17k/fitzpatrick17k.csv
  3. Download images using the script in fitzpatrick17k/data/download.py
     OR follow the alternative download instructions in their README.
  4. Place metadata + images in: data/external/Fitzpatrick17k/
     Expected layout:
        data/external/Fitzpatrick17k/fitzpatrick17k.csv
        data/external/Fitzpatrick17k/images/{md5hash}.jpg
  5. Some URLs may be dead — expect to drop ~10-20% of rows.
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
    bucket_fitzpatrick,
    fitz17k_to_ham10k,
    fitz17k_to_universal,
)


def load_fitzpatrick17k_metadata(
    data_dir: str | Path,
    only_qc: bool = True,
    drop_unknown_fitz: bool = True,
) -> pd.DataFrame:
    """
    Load Fitzpatrick17k metadata.

    Args:
        data_dir         : path to data/external/Fitzpatrick17k
        only_qc          : if True, keep only rows where qc==1 (recommended)
        drop_unknown_fitz: if True, drop rows where fitzpatrick is unknown (-1 or NaN)

    Returns DataFrame with columns:
        image_id          : md5hash
        image_path        : absolute path to image
        label             : raw 114-class label
        three_partition   : {benign, malignant, non-neoplastic}
        universal_label   : same, but as a single string
        universal_idx     : 0/1/2
        ham10k_label      : str or None (overlap only)
        ham10k_idx        : int or -1
        fitzpatrick       : int 1-6 (or -1 if unknown)
        fitz_bucket       : light/medium/dark or None
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "fitzpatrick17k.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fitzpatrick17k metadata not found at {csv_path}.\n"
            f"See setup instructions in src/data/external/fitzpatrick17k.py docstring."
        )

    df = pd.read_csv(csv_path)

    # Standardise column names across releases
    if "md5hash" in df.columns and "image_id" not in df.columns:
        df["image_id"] = df["md5hash"]
    if "fitzpatrick_scale" in df.columns and "fitzpatrick" not in df.columns:
        df["fitzpatrick"] = df["fitzpatrick_scale"]

    # Filter QC: in the mattgroh/fitzpatrick17k CSV, the `qc` column is a
    # reviewer flag, not a pass/fail bit. Values:
    #   NaN                 → no issue raised (assumed-OK, KEEP)
    #   "1 Diagnostic"      → reviewer-confirmed good   (KEEP)
    #   "2 Characteristic"  → reviewer-confirmed good   (KEEP)
    #   "5 Potentially"     → potentially mislabelled   (KEEP — ambiguous)
    #   "3 Wrongly labelled"→ DROP
    #   "4 Other"           → DROP (other quality issue)
    if only_qc and "qc" in df.columns:
        before = len(df)
        drop_flags = {"3 Wrongly labelled", "4 Other"}
        qc_str = df["qc"].astype(str).str.strip()
        df = df[~qc_str.isin(drop_flags)].reset_index(drop=True)
        print(f"  [Fitz17k] qc filter (drop wrong+other): {before} -> {len(df)} rows")

    # Drop unknown Fitzpatrick if requested
    if drop_unknown_fitz and "fitzpatrick" in df.columns:
        before = len(df)
        df = df[df["fitzpatrick"].astype(str).isin({"1", "2", "3", "4", "5", "6"})]
        df = df.reset_index(drop=True)
        print(f"  [Fitz17k] unknown-fitz filter: {before} -> {len(df)} rows")

    # Build image paths
    image_dir = data_dir / "images"
    if not image_dir.exists():
        image_dir = data_dir
    df["image_path"] = df["image_id"].apply(lambda x: str(image_dir / f"{x}.jpg"))

    # Drop rows with missing images
    exists = df["image_path"].apply(lambda p: Path(p).exists())
    missing = (~exists).sum()
    if missing:
        print(f"  [Fitz17k] Dropping {missing} rows with missing image files")
    df = df[exists].reset_index(drop=True)

    # Apply taxonomy mapping
    df["three_partition"] = df["three_partition_label"]
    df["universal_label"] = df["three_partition"].apply(fitz17k_to_universal)
    df["universal_idx"] = df["universal_label"].map(UNIVERSAL_TO_IDX)

    df["ham10k_label"] = df["label"].apply(fitz17k_to_ham10k)
    df["ham10k_idx"] = df["ham10k_label"].map(lambda x: HAM10K_TO_IDX.get(x, -1))

    df["fitzpatrick"] = df["fitzpatrick"].astype(int)
    df["fitz_bucket"] = df["fitzpatrick"].apply(bucket_fitzpatrick)

    return df


class Fitzpatrick17kDataset(Dataset):
    """
    PyTorch Dataset for Fitzpatrick17k.

    Args:
        df:         DataFrame from load_fitzpatrick17k_metadata()
        transform:  Albumentations transform pipeline
        label_mode: 'universal' (3-class, all rows) or
                    'ham10k' (7-class, filters to mappable rows)
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