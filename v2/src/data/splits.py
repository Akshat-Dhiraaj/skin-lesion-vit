"""
src/data/splits.py
------------------
Patient-level / lesion-level stratified splitting for dermoscopy datasets.

Why this module exists
----------------------
The original skin-lesion-vit repo used image-level random splits via
sklearn.train_test_split. That's a known leak source on HAM10000:
multiple images can be taken of the same lesion in a single session,
so an image-level split lets near-duplicates leak from train into val/test.
On ISIC 2019/2020, BCN20000, and PAD-UFES the issue is more severe —
patients contribute many lesions, all sharing skin tone, lighting, and
imaging device characteristics.

This module wraps sklearn.StratifiedGroupKFold so every dataset in the
project goes through one tested function. The grouping column varies:
  - HAM10000:        group_col='lesion_id' (no patient_id field exposed;
                     lesion_id is the tightest available proxy and is
                     what the official HAM10000 paper recommends).
  - ISIC 2019/2020:  group_col='patient_id' (or 'lesion_id' if studying
                     lesion-level rather than patient-level generalization).
  - BCN20000:        group_col='patient_id'.
  - PAD-UFES-20:     group_col='patient_id'.
  - DDI:             group_col='lesion_id' (one image per lesion).
  - Fitzpatrick17k:  group_col='md5hash' (no patient grouping; effectively
                     image-level, but use the hash to dedup near-duplicates).
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def split_dataset(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified group-aware train/val/test split.

    Uses two passes of StratifiedGroupKFold:
        Pass 1: separate test from (train + val).
        Pass 2: separate val from train within (train + val).

    The exact split sizes will not equal val_frac and test_frac exactly,
    because group sizes vary. Fractions are targets, not guarantees.

    Args:
        df: Source DataFrame. Must contain label_col and group_col with
            no NaNs in either.
        label_col: Column for class label, used for stratification.
        group_col: Column for the grouping key (lesion_id, patient_id,
            etc.). All rows sharing a value here go to the same split.
        val_frac: Target validation fraction of total samples.
        test_frac: Target test fraction of total samples.
        random_state: Seed for both StratifiedGroupKFold passes.

    Returns:
        (train_df, val_df, test_df), each with reset 0-based index.
        Disjoint by group_col. Class distributions are approximately
        preserved subject to group constraints.

    Raises:
        ValueError: if fractions are out of range, sum to >= 1, or if
            group_col / label_col contains NaN.
    """
    if not (0 < val_frac < 1):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if not (0 < test_frac < 1):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if val_frac + test_frac >= 1:
        raise ValueError(
            f"val_frac + test_frac must be < 1, got {val_frac + test_frac}"
        )

    for col in (label_col, group_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in DataFrame")
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(
                f"{n} rows have NaN in column '{col}'. "
                f"Drop or impute before splitting."
            )

    # Pass 1 — split off test
    n_splits_outer = max(2, round(1.0 / test_frac))
    sgkf_outer = StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state
    )
    train_val_idx, test_idx = next(
        sgkf_outer.split(df, df[label_col], groups=df[group_col])
    )
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Pass 2 — split val out of (train+val)
    val_frac_within = val_frac / (1.0 - test_frac)
    n_splits_inner = max(2, round(1.0 / val_frac_within))
    sgkf_inner = StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=random_state
    )
    train_idx, val_idx = next(
        sgkf_inner.split(
            train_val_df,
            train_val_df[label_col],
            groups=train_val_df[group_col],
        )
    )
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def assert_no_group_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> None:
    """
    Raise AssertionError if any group_col value appears in more than one split.

    Use this in CI tests and as a sanity check inside training scripts
    after every call to split_dataset(). The cost is negligible.
    """
    train_groups = set(train_df[group_col].unique())
    val_groups = set(val_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())

    overlaps = {
        "train ∩ val": train_groups & val_groups,
        "train ∩ test": train_groups & test_groups,
        "val ∩ test": val_groups & test_groups,
    }
    leaks = {k: v for k, v in overlaps.items() if v}
    if leaks:
        msg_parts = [
            f"{k}: {len(v)} groups (e.g. {sorted(list(v))[:3]})"
            for k, v in leaks.items()
        ]
        raise AssertionError(
            "Group leakage detected across splits:\n  " + "\n  ".join(msg_parts)
        )


def report_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Build a summary table of per-split image counts, group counts, and
    per-class image counts. Useful to print before training, and to log
    to MLflow as an artifact.
    """
    all_classes = sorted(
        pd.concat([train_df[label_col], val_df[label_col], test_df[label_col]]).unique()
    )
    rows = []
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        row: dict = {
            "split": name,
            "n_images": len(sub),
            "n_groups": int(sub[group_col].nunique()),
        }
        for cls in all_classes:
            row[f"n_{cls}"] = int((sub[label_col] == cls).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def split_summary_string(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    group_col: str,
) -> str:
    """One-line-per-split human-readable summary."""
    n_total = len(train_df) + len(val_df) + len(test_df)
    lines = [f"Split summary  (total = {n_total} images)"]
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = 100.0 * len(sub) / n_total
        lines.append(
            f"  {name:5s}: {len(sub):6d} images  "
            f"({pct:5.2f}%)  |  "
            f"{sub[group_col].nunique():5d} groups  |  "
            f"{sub[label_col].nunique()} classes"
        )
    return "\n".join(lines)


# ── Convenience wrappers per dataset ──────────────────────────────────────────

def split_ham10000(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """HAM10000 convenience: groups by lesion_id, stratifies by dx."""
    return split_dataset(
        df, label_col="dx", group_col="lesion_id",
        val_frac=val_frac, test_frac=test_frac, random_state=random_state,
    )


def split_isic_patient(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ISIC 2019/2020/BCN20000/PAD-UFES: groups by patient_id, stratifies by diagnosis."""
    return split_dataset(
        df, label_col="diagnosis", group_col="patient_id",
        val_frac=val_frac, test_frac=test_frac, random_state=random_state,
    )