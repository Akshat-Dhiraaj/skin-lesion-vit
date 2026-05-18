"""
src/tta.py
----------
Test-Time Augmentation inference for any registered model.

For each split image, runs inference under 8 augmented views and averages
the softmax probabilities. Saves prob arrays for downstream ensemble.

Usage:
    # Both val and test (default - needed for post-hoc analysis)
    python -m src.tta --model convnextv2_large --data_dir <path>

    # Just one split
    python -m src.tta --model convnextv2_large --data_dir <path> --split test

Outputs (per model, in experiments/<model_name>/):
    tta_test_probs.npy      tta_test_labels.npy
    tta_val_probs.npy       tta_val_labels.npy
    tta_probs.npy           tta_labels.npy        (test split, kept for backward compat)
    tta_results.txt         human-readable report
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ham10000 import CLASS_NAMES, HAM10000Dataset, load_ham10000
from src.data.transforms import get_norm_stats
from src.models.registry import REGISTRY, build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"


# ── 8 TTA views ───────────────────────────────────────────────────────────────

def get_tta_transforms(image_size: int, model_type: str | None = None) -> list:
    """8 deterministic augmented views."""
    mean, std = get_norm_stats(model_type)
    norm = A.Normalize(mean=list(mean), std=list(std))
    base = [A.Resize(image_size, image_size)]
    tail = [norm, ToTensorV2()]

    return [
        A.Compose(base + tail),                                               # 1. Original
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + tail),                   # 2. H-flip
        A.Compose(base + [A.VerticalFlip(p=1.0)] + tail),                     # 3. V-flip
        A.Compose(base + [A.Rotate(limit=(90, 90), p=1.0)] + tail),           # 4. Rot 90
        A.Compose(base + [A.Rotate(limit=(180, 180), p=1.0)] + tail),         # 5. Rot 180
        A.Compose(base + [A.Rotate(limit=(270, 270), p=1.0)] + tail),         # 6. Rot 270
        A.Compose(base + [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + tail),  # 7. HV-flip
        A.Compose(
            base + [A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=0, p=1.0)] + tail
        ),  # 8. Slight zoom
    ]


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    split_df,
    image_size: int,
    batch_size: int = 16,
    num_workers: int = 4,
    model_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TTA on a split. Returns (avg_probs, labels)."""
    model.eval()
    transforms = get_tta_transforms(image_size, model_type)

    view_probs_all = []
    labels_arr = None

    for view_idx, transform in enumerate(transforms):
        print(f"  TTA view {view_idx + 1}/{len(transforms)}...")
        ds = HAM10000Dataset(split_df, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        view_probs = []
        view_labels = []
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            with torch.amp.autocast(_DEVICE_TYPE):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            view_probs.append(probs)
            view_labels.append(labels.numpy())

        view_probs = np.concatenate(view_probs)

        # Handle NaN from AMP overflow
        if np.isnan(view_probs).any():
            n_nan = int(np.isnan(view_probs).any(axis=1).sum())
            print(f"    [WARN] {n_nan} NaN predictions - replacing with uniform")
            view_probs = np.where(np.isnan(view_probs), 1.0 / view_probs.shape[1], view_probs)

        view_probs_all.append(view_probs)

        if labels_arr is None:
            labels_arr = np.concatenate(view_labels)

    avg_probs = np.mean(view_probs_all, axis=0)
    return avg_probs, labels_arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both",
                        help="Which split to run TTA on. 'both' is needed for post-hoc.")
    args = parser.parse_args()

    cfg = REGISTRY[args.model]
    ckpt_dir = Path(args.ckpt_base) / cfg.name
    ckpt_path = ckpt_dir / "best_model.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print(f"        Run training first: python scripts/train.py --model {args.model}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  TTA - {cfg.name}")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Image size: {cfg.input_size}")
    print(f"Splits    : {args.split}")

    # Load splits
    train_df, val_df, test_df = load_ham10000(args.data_dir, random_state=args.random_state)

    splits_to_run = {}
    if args.split in ("val", "both"):
        splits_to_run["val"] = val_df
    if args.split in ("test", "both"):
        splits_to_run["test"] = test_df

    # Build model and load weights once
    model = build_model(cfg).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Read baseline test AUC from results.txt for delta reporting
    baseline_auc = None
    results_path = ckpt_dir / "results.txt"
    if results_path.exists():
        try:
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Test AUC"):
                    baseline_auc = float(line.split(":")[1].strip())
                    break
        except Exception:
            pass

    # Process each requested split
    summary_lines = [f"{cfg.name} - TTA (8 views)\n"]

    for split_name, split_df in splits_to_run.items():
        print(f"\n--- Running TTA on {split_name} split ({len(split_df)} images) ---")
        try:
            probs, labels = tta_predict(
                model, split_df,
                image_size=cfg.input_size,
                batch_size=cfg.batch_size,
                num_workers=args.num_workers,
                model_type=cfg.model_type,
            )
        except Exception as e:
            print(f"[ERROR] TTA failed on {split_name}: {e}")
            traceback.print_exc()
            continue

        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        acc = float((probs.argmax(1) == labels).mean())

        print(f"\n{split_name.upper()} TTA AUC: {auc:.4f}")
        print(f"{split_name.upper()} TTA Acc: {acc:.4f}")

        # Save (split-aware naming + backward-compat for test)
        np.save(ckpt_dir / f"tta_{split_name}_probs.npy", probs)
        np.save(ckpt_dir / f"tta_{split_name}_labels.npy", labels)
        if split_name == "test":
            # Backward compat with the older src/ensemble.py
            np.save(ckpt_dir / "tta_probs.npy", probs)
            np.save(ckpt_dir / "tta_labels.npy", labels)

        # Add to summary
        summary_lines.append(f"\n[{split_name.upper()}]")
        summary_lines.append(f"TTA AUC : {auc:.4f}")
        summary_lines.append(f"TTA Acc : {acc:.4f}")
        if split_name == "test" and baseline_auc is not None:
            summary_lines.append(f"Baseline: {baseline_auc:.4f}  (delta {auc - baseline_auc:+.4f})")
        summary_lines.append("")
        summary_lines.append(classification_report(labels, probs.argmax(1), target_names=CLASS_NAMES))

    # Write report
    report_path = ckpt_dir / "tta_results.txt"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nTTA report saved: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)