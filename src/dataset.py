"""
dataset.py
----------
PyTorch Dataset for HAM10000 skin lesion classification.
Handles loading, splitting, and class imbalance.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# 7 lesion classes in HAM10000
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Args:
        df         : DataFrame with columns [image_id, dx, image_path]
        transform  : Albumentations transform pipeline
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = CLASS_TO_IDX[row['dx']]
        return image, label


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV and attach image paths.

    Args:
        data_dir: path to data/raw/ containing HAM10000_metadata.csv and images

    Returns:
        DataFrame with image_path column attached
    """
    csv_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(csv_path)

    # HAM10000 images are split across two folders
    image_dirs = [
        os.path.join(data_dir, 'HAM10000_images_part_1'),
        os.path.join(data_dir, 'HAM10000_images_part_2'),
    ]

    # Build image_id -> path mapping
    image_path_map = {}
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            for fname in os.listdir(image_dir):
                if fname.endswith('.jpg'):
                    image_id = fname.replace('.jpg', '')
                    image_path_map[image_id] = os.path.join(image_dir, fname)

    df['image_path'] = df['image_id'].map(image_path_map)
    df = df.dropna(subset=['image_path'])  # drop rows with missing images

    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['dx'].value_counts()}")

    return df


def split_dataset(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """
    Stratified train/val/test split to preserve class distribution.
    """
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size),
        stratify=df['dx'], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size),
        stratify=temp_df['dx'], random_state=random_state
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Rare classes (melanoma, df, vasc) get sampled more frequently.
    """
    class_counts = df['dx'].value_counts().to_dict()
    weights = [1.0 / class_counts[dx] for dx in df['dx']]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def get_dataloaders(train_df, val_df, test_df, train_transform, val_transform, batch_size=32):
    """
    Build DataLoaders for train/val/test splits.
    Training uses WeightedRandomSampler for class balance.
    """
    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=val_transform)
    test_dataset = HAM10000Dataset(test_df, transform=val_transform)

    sampler = get_weighted_sampler(train_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader