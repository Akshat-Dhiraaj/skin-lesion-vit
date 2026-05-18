"""
scripts/sample_efficiency.py
----------------------------
Train a registry model on a lesion-level fraction of HAM10000 train set.

Subsamples by `lesion_id` (NOT by image) so the split-leakage guarantee
holds. Val and test sets are unchanged. Writes results to a frac-suffixed
experiment dir so each (model, frac) pair has its own folder.

Usage:
    python scripts/sample_efficiency.py --model pandem_real_probe --frac 0.25
    python scripts/sample_efficiency.py --model efficientnetv2_m   --frac 0.50

Output dir: experiments/sample_efficiency/<model>_<frac>/
  best_model.pth, results.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.ham10000 import CLASS_NAMES, get_dataloaders, load_ham10000
from src.data.transforms import get_train_transform, get_val_transform
from src.engine import (
    DEVICE, _enable_grad_checkpointing, _print_header,
    evaluate, train_one_epoch,
)
from src.models.registry import REGISTRY, build_model, build_optimizer
from sklearn.metrics import classification_report
import mlflow


def subsample_train_by_lesion(train_df, fraction: float, seed: int = 42):
    """Keep `fraction` of unique lesion_ids (stratified by dx mode per lesion)."""
    if fraction >= 1.0:
        return train_df.reset_index(drop=True)
    # Per-lesion dominant class (lesions usually have a consistent dx)
    lesion_class = train_df.groupby("lesion_id")["dx"].agg(lambda x: x.mode().iloc[0])
    rng = np.random.default_rng(seed)
    keep_ids = []
    for cls in lesion_class.unique():
        lesions_in_cls = lesion_class[lesion_class == cls].index.to_numpy()
        n_keep = max(1, int(round(len(lesions_in_cls) * fraction)))
        keep_ids.extend(rng.choice(lesions_in_cls, size=n_keep, replace=False))
    keep_ids = set(keep_ids)
    sub = train_df[train_df["lesion_id"].isin(keep_ids)].reset_index(drop=True)
    print(f"  [subsample] frac={fraction:.2f}  lesions: {len(keep_ids)} (of {len(lesion_class)})  "
          f"images: {len(sub)} (of {len(train_df)})")
    return sub


def run_with_subsample(model_key: str, fraction: float, data_dir: str,
                        ckpt_base: str = "experiments/sample_efficiency",
                        num_workers: int = 0, random_state: int = 42) -> dict:
    cfg = REGISTRY[model_key]
    _print_header(cfg)
    print(f"Sample efficiency mode: fraction={fraction:.2f}")

    # Load full splits, then subsample train only
    train_df, val_df, test_df = load_ham10000(data_dir, random_state=random_state)
    train_df = subsample_train_by_lesion(train_df, fraction, seed=random_state)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        get_train_transform(cfg.input_size, cfg.model_type),
        get_val_transform(cfg.input_size, cfg.model_type),
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )

    model = build_model(cfg).to(DEVICE)
    _enable_grad_checkpointing(model)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)

    scheduler = None
    if cfg.lr_backbone > 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.1,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    ckpt_dir = Path(ckpt_base) / f"{cfg.name}_frac{int(fraction*100):03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    mlflow.set_experiment("skin-lesion-v2-sample-eff")
    with mlflow.start_run(run_name=f"{cfg.name}_frac{int(fraction*100):03d}"):
        mlflow.log_params({
            "model_key": model_key, "fraction": fraction,
            "n_train_images": len(train_df),
            "n_train_lesions": train_df["lesion_id"].nunique(),
            "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        })
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                          scaler, cfg.accum_steps)
            val_m = evaluate(model, val_loader)
            print(f"  Loss: {train_loss:.4f} | Val AUC: {val_m['auc']:.4f} | Val Acc: {val_m['acc']:.4f}")
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_auc": val_m["auc"], "val_acc": val_m["acc"]},
                step=epoch,
            )
            if val_m["auc"] > best_auc:
                best_auc = val_m["auc"]
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                print(f"  [BEST] Val AUC: {best_auc:.4f}")

        print("\nLoading best for test...")
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", map_location=DEVICE))
        test_m = evaluate(model, test_loader)
        print(f"\nTest AUC: {test_m['auc']:.4f} | Test Acc: {test_m['acc']:.4f}")
        mlflow.log_metrics({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

        report = classification_report(test_m["labels"], test_m["probs"].argmax(1),
                                       target_names=CLASS_NAMES)
        (ckpt_dir / "results.txt").write_text(
            f"{cfg.name} - sample efficiency (frac={fraction:.2f})\n\n"
            f"Train lesions: {train_df['lesion_id'].nunique()}\n"
            f"Train images : {len(train_df)}\n\n"
            f"Best Val AUC : {best_auc:.4f}\n"
            f"Test AUC     : {test_m['auc']:.4f}\n"
            f"Test Acc     : {test_m['acc']:.4f}\n\n"
            f"{report}", encoding="utf-8",
        )

    return {"best_val_auc": best_auc, "test_auc": test_m["auc"], "test_acc": test_m["acc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--frac", type=float, required=True,
                        help="Fraction of train lesions to keep (0 < frac <= 1)")
    parser.add_argument("--data_dir", default=r"D:\ML\skin-lesion-vit\data\raw")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    assert 0 < args.frac <= 1.0, "fraction must be in (0, 1]"
    run_with_subsample(args.model, args.frac, args.data_dir,
                       num_workers=args.num_workers, random_state=args.random_state)


if __name__ == "__main__":
    main()
