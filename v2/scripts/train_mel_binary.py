"""
scripts/train_mel_binary.py
---------------------------
Binary melanoma specialist (OA1) — v2 port of v1 Phase 19.

Wraps a registry backbone with a single sigmoid logit head, trains with
BinaryFocalLoss + 1:1 balanced sampler. Mel idx in HAM10000 = 0.

Usage:
    python scripts/train_mel_binary.py --backbone pandem_real_probe --epochs 15 --num_workers 0
    python scripts/train_mel_binary.py --backbone convnextv2_large  --epochs 12 --num_workers 0
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.data.ham10000 import CLASS_TO_IDX, load_ham10000
from src.data.transforms import get_train_transform, get_val_transform
from src.engine import DEVICE, _enable_grad_checkpointing, _print_header
from src.losses import BinaryFocalLoss
from src.models.registry import REGISTRY, build_model, build_optimizer

MEL_IDX = CLASS_TO_IDX["mel"]  # 0


class MelBinaryDataset(Dataset):
    """Wraps HAM10000 DataFrame, returns (image, 1 if mel else 0)."""

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
        label = 1 if int(row["label"]) == MEL_IDX else 0
        return image, label


def get_binary_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """50/50 mel/non-mel sampler."""
    is_mel = (df["label"] == MEL_IDX).values
    n_mel, n_non = int(is_mel.sum()), int((~is_mel).sum())
    weights = np.where(is_mel, 1.0 / max(n_mel, 1), 1.0 / max(n_non, 1))
    return WeightedRandomSampler(weights=weights.tolist(),
                                  num_samples=len(weights), replacement=True)


def get_dataloaders(train_df, val_df, test_df, train_tf, val_tf, batch_size, num_workers):
    train_ds = MelBinaryDataset(train_df, train_tf)
    val_ds   = MelBinaryDataset(val_df,   val_tf)
    test_ds  = MelBinaryDataset(test_df,  val_tf)
    sampler  = get_binary_sampler(train_df)
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader) -> dict:
    """Returns {auc, probs, labels} for binary mel score."""
    model.eval()
    all_p, all_y = [], []
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(DEVICE)
        with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            logits = model(x)
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        if np.isnan(p).any():
            p = np.where(np.isnan(p), 0.5, p)
        all_p.append(p); all_y.append(y.numpy())
    probs = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    auc = float(roc_auc_score(labels, probs)) if labels.sum() > 0 else float("nan")
    return {"auc": auc, "probs": probs, "labels": labels}


def recall_at_precision(probs: np.ndarray, labels: np.ndarray,
                        target_precision: float) -> tuple[float, float]:
    """Returns (recall_at_target_precision, threshold)."""
    p, r, t = precision_recall_curve(labels, probs)
    # p and r have length len(t)+1; align with t by ignoring the last point
    p, r = p[:-1], r[:-1]
    mask = p >= target_precision
    if not mask.any():
        return 0.0, float("nan")
    best_idx = np.where(mask)[0][np.argmax(r[mask])]
    return float(r[best_idx]), float(t[best_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data_dir", default=r"D:\ML\skin-lesion-vit\data\raw")
    parser.add_argument("--ckpt_base", default="experiments/mel_binary")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal alpha")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal gamma")
    parser.add_argument("--lr_head", type=float, default=None,
                        help="Override head LR (default: cfg.lr_head from registry)")
    parser.add_argument("--lr_backbone", type=float, default=None,
                        help="Override backbone LR (default: cfg.lr_backbone)")
    parser.add_argument("--name_suffix", type=str, default="",
                        help="Suffix appended to experiment dir name")
    args = parser.parse_args()

    # Build the backbone with num_classes=1 (Linear head emits one logit)
    cfg = REGISTRY[args.backbone]
    overrides = {"num_classes": 1,
                 "name": f"mel_binary_{cfg.name}{args.name_suffix}",
                 "epochs": args.epochs}
    if args.lr_head is not None: overrides["lr_head"] = args.lr_head
    if args.lr_backbone is not None: overrides["lr_backbone"] = args.lr_backbone
    cfg = replace(cfg, **overrides)
    _print_header(cfg)

    train_df, val_df, test_df = load_ham10000(args.data_dir, random_state=args.random_state)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        get_train_transform(cfg.input_size, cfg.model_type),
        get_val_transform(cfg.input_size, cfg.model_type),
        batch_size=cfg.batch_size, num_workers=args.num_workers,
    )

    model = build_model(cfg).to(DEVICE)
    _enable_grad_checkpointing(model)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    scheduler = None
    if cfg.lr_backbone > 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
            steps_per_epoch=steps_per_epoch, epochs=cfg.epochs, pct_start=0.1,
        )

    criterion = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    ckpt_dir = Path(args.ckpt_base) / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    mlflow.set_experiment("skin-lesion-v2-mel-binary")
    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params({"backbone": args.backbone, "epochs": cfg.epochs,
                           "alpha": args.alpha, "gamma": args.gamma,
                           "batch_size": cfg.batch_size, "accum_steps": cfg.accum_steps,
                           "lr_head": cfg.lr_head, "lr_backbone": cfg.lr_backbone,
                           "name_suffix": args.name_suffix})
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for step, (x, y) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
                    logits = model(x)
                    loss = criterion(logits, y) / cfg.accum_steps
                scaler.scale(loss).backward()
                total_loss += loss.item() * cfg.accum_steps
                if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    if scheduler is not None: scheduler.step()
                    optimizer.zero_grad()

            val_m = evaluate_binary(model, val_loader)
            print(f"  Loss: {total_loss/len(train_loader):.4f} | Val mel AUC: {val_m['auc']:.4f}")
            mlflow.log_metrics({"train_loss": total_loss/len(train_loader),
                                "val_mel_auc": val_m["auc"]}, step=epoch)
            if val_m["auc"] > best_auc:
                best_auc = val_m["auc"]
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                print(f"  [BEST] Val mel AUC: {best_auc:.4f}")

        # Test
        print("\nLoading best for test...")
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", map_location=DEVICE))
        test_m = evaluate_binary(model, test_loader)
        r95, t95 = recall_at_precision(test_m["probs"], test_m["labels"], 0.95)
        r90, t90 = recall_at_precision(test_m["probs"], test_m["labels"], 0.90)
        r80, t80 = recall_at_precision(test_m["probs"], test_m["labels"], 0.80)
        print(f"\nTest mel AUC:           {test_m['auc']:.4f}")
        print(f"Recall @ Precision=0.95: {r95:.4f}  (threshold={t95:.4f})")
        print(f"Recall @ Precision=0.90: {r90:.4f}  (threshold={t90:.4f})")
        print(f"Recall @ Precision=0.80: {r80:.4f}  (threshold={t80:.4f})")
        mlflow.log_metrics({"test_mel_auc": test_m["auc"],
                            "test_recall_at_p95": r95,
                            "test_recall_at_p90": r90,
                            "test_recall_at_p80": r80})

        # Save artifacts
        np.save(ckpt_dir / "test_probs.npy", test_m["probs"])
        np.save(ckpt_dir / "test_labels.npy", test_m["labels"])
        (ckpt_dir / "results.txt").write_text(
            f"{cfg.name}\n\n"
            f"Backbone : {args.backbone}\n"
            f"Loss     : BinaryFocalLoss(alpha={args.alpha}, gamma={args.gamma})\n\n"
            f"Best Val mel AUC : {best_auc:.4f}\n"
            f"Test mel AUC     : {test_m['auc']:.4f}\n\n"
            f"Recall @ P=0.95 : {r95:.4f}  (threshold={t95:.4f})\n"
            f"Recall @ P=0.90 : {r90:.4f}  (threshold={t90:.4f})\n"
            f"Recall @ P=0.80 : {r80:.4f}  (threshold={t80:.4f})\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
