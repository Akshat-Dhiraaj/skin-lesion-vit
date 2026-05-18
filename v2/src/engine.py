"""
src/engine.py
-------------
Unified training and evaluation loop for all phases.

Changes vs the original skin-lesion-vit engine.py
--------------------------------------------------
- All models share the same forward(x) -> logits call. No per-model_type
  dispatch in the training loop — that complexity lives in registry.py.
- torch.amp.autocast(device_type) replaces the cuda-only autocast so
  the loop runs correctly on CPU for debugging.
- Gradient clipping added (max_norm=1.0) — stabilises transformer fine-tunes.
- Handles remaining gradient accumulation steps at end of epoch.
- _enable_grad_checkpointing() covers both HF backbone and timm patterns.
- run_phase() returns a result dict so callers can chain phases.
- NaN handling in evaluate() — EfficientNetV2 + AMP can overflow to NaN.
"""

from __future__ import annotations

from pathlib import Path
import math

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from src.data.ham10000 import CLASS_NAMES, get_dataloaders, load_ham10000
from src.data.transforms import get_train_transform, get_val_transform
from src.models.registry import (
    REGISTRY,
    ModelConfig,
    build_model,
    build_optimizer,
)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_DEVICE_TYPE: str = "cuda" if DEVICE == "cuda" else "cpu"  # for torch.amp


# ── Core loops ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    accum_steps: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.amp.autocast(_DEVICE_TYPE):
            loss = criterion(model(images), labels) / accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> dict:
    """
    Run inference on loader and return metrics + raw arrays.

    Returns:
        dict with keys: auc (float), acc (float),
                        probs (ndarray N×C), labels (ndarray N,)
    """
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE)
        with torch.amp.autocast(_DEVICE_TYPE):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Handle NaN from AMP overflow (EfficientNetV2 is prone to this)
    if np.isnan(all_probs).any():
        n_nan = int(np.isnan(all_probs).any(axis=1).sum())
        print(f"  ⚠ Found {n_nan} NaN predictions — replacing with uniform 1/{all_probs.shape[1]}")
        all_probs = np.where(np.isnan(all_probs), 1.0 / all_probs.shape[1], all_probs)

    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    acc = float((all_probs.argmax(1) == all_labels).mean())

    return {"auc": float(auc), "acc": acc, "probs": all_probs, "labels": all_labels}


# ── Main phase runner ─────────────────────────────────────────────────────────

def run_phase(
    model_key: str,
    data_dir: str,
    ckpt_base: str = "experiments",
    num_workers: int = 4,
    random_state: int = 42,
) -> dict:
    """
    Full train + val + test cycle for one registry entry.

    Args:
        model_key:   Key into REGISTRY (e.g. "dinov2_registers_ft").
        data_dir:    Path to HAM10000_metadata.csv and image folders.
        ckpt_base:   Parent directory for checkpoints and results.txt.
        num_workers: DataLoader workers (use 0 on Windows if fork errors).
        random_state: Seed for dataset splitting.

    Returns:
        {"best_val_auc", "test_auc", "test_acc"}
    """
    cfg: ModelConfig = REGISTRY[model_key]

    _print_header(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_ham10000(
        data_dir, random_state=random_state
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        get_train_transform(cfg.input_size, cfg.model_type),
        get_val_transform(cfg.input_size, cfg.model_type),
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(DEVICE)
    _enable_grad_checkpointing(model)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
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

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(ckpt_base) / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    # ── Training ──────────────────────────────────────────────────────────────
    mlflow.set_experiment("skin-lesion-v2")

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params({
            "model_key":       model_key,
            "model_id":        cfg.model_id,
            "freeze_backbone": cfg.freeze_backbone,
            "input_size":      cfg.input_size,
            "batch_size":      cfg.batch_size,
            "lr_head":         cfg.lr_head,
            "lr_backbone":     cfg.lr_backbone,
            "epochs":          cfg.epochs,
            "accum_steps":     cfg.accum_steps,
        })

        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, cfg.accum_steps
            )
            val_m = evaluate(model, val_loader)

            print(
                f"  Loss: {train_loss:.4f} | "
                f"Val AUC: {val_m['auc']:.4f} | "
                f"Val Acc: {val_m['acc']:.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_auc": val_m["auc"], "val_acc": val_m["acc"]},
                step=epoch,
            )

            if val_m["auc"] > best_auc:
                best_auc = val_m["auc"]
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                print(f"  [BEST] Val AUC: {best_auc:.4f}")

        # ── Test eval ─────────────────────────────────────────────────────────
        print("\nLoading best checkpoint for test evaluation...")
        model.load_state_dict(
            torch.load(ckpt_dir / "best_model.pth", map_location=DEVICE)
        )
        test_m = evaluate(model, test_loader)

        print(
            f"\nTest AUC: {test_m['auc']:.4f} | "
            f"Test Acc: {test_m['acc']:.4f}"
        )
        mlflow.log_metrics({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

        # ── results.txt ───────────────────────────────────────────────────────
        result_path = ckpt_dir / "results.txt"
        report = classification_report(
            test_m["labels"], test_m["probs"].argmax(1),
            target_names=CLASS_NAMES,
        )
        result_path.write_text(
            f"{cfg.name}\n\n"
            f"Best Val AUC : {best_auc:.4f}\n"
            f"Test AUC     : {test_m['auc']:.4f}\n"
            f"Test Acc     : {test_m['acc']:.4f}\n\n"
            f"{report}",
            encoding="utf-8",
        )
        print(f"Results saved to {result_path}")

    return {
        "best_val_auc": best_auc,
        "test_auc": test_m["auc"],
        "test_acc": test_m["acc"],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(cfg: ModelConfig) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {cfg.name.upper()}")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU : {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")


def _enable_grad_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing to save VRAM on transformer backbones.
    Tries the HuggingFace API first, then the timm pattern.
    Silently skips if the backbone doesn't support it.
    """
    # HuggingFace backbones (DINOv2Classifier, PanDermClassifier)
    backbone = getattr(model, "backbone", None) or getattr(model, "vision", None)
    if backbone is not None and hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled (HF backbone)")
            return
        except Exception:
            pass

    # timm backbones (ConvNeXt-V2 etc.) — set_grad_checkpointing()
    inner = getattr(model, "backbone", model)
    if hasattr(inner, "set_grad_checkpointing"):
        try:
            inner.set_grad_checkpointing(enable=True)
            print("Gradient checkpointing: enabled (timm backbone)")
            return
        except Exception:
            pass

    print("Gradient checkpointing: not available for this backbone")