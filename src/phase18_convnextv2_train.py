"""
phase18_convnextv2_train.py
----------------------------
Trains ConvNeXt-V2 Large at 384px as a NEW model that brings genuine
architectural diversity to the existing ensemble.

WHY THIS MODEL
--------------
Current ensemble (eff, dino-FT, p13, mm) has a hidden weakness: 3 of the 4
models share the DINOv2+Registers backbone. Only EfficientNetV2 is
architecturally distinct. This means the ensemble's mistakes on the mel/nv
boundary are highly correlated — adding more views or more SLSQP restarts
cannot fix correlated errors.

ConvNeXt-V2 is a modern CNN family that:
  - Uses sparse convolutions and a fully-convolutional masked autoencoder
    pre-training (FCMAE) — completely different inductive bias from
    self-supervised ViTs (DINOv2) and supervised CNNs (EfficientNetV2).
  - Has been shown to match or exceed ViT performance on dermatology
    benchmarks while making different errors.
  - At 384px the receptive field captures both local lesion texture and
    global skin context — important for dermoscopy.

Expected standalone test AUC: 0.965–0.975
Expected ensemble contribution: +0.003–0.006 AUC by reducing mel/nv error
correlation in the meta-learner.

TRAINING CONFIG (chosen for RTX 4060 8GB)
------------------------------------------
  Model        : convnextv2_large.fcmae_ft_in22k_in1k_384  (197M params)
  Input size   : 384 × 384  (matches model's pretraining resolution)
  Batch size   : 4  (real)  → effective 32 with accum_steps=8
  Epochs       : 25
  LR head      : 1e-3
  LR backbone  : 3e-6   (lower than typical 1e-5 because 197M params overfit fast)
  Scheduler    : OneCycleLR with 10% warmup
  Optimizer    : AdamW, weight_decay=1e-4
  AMP          : torch.amp.autocast (cuda)
  Sampler      : WeightedRandomSampler (existing, handles class imbalance)
  Augmentation : same train_transform as other phases (heavy)

VRAM BUDGET (RTX 4060 8GB)
---------------------------
  ConvNeXt-V2 Large @ 384px, batch 4, AMP:
    forward activations  ≈ 4.0 GB
    parameters (fp32)    ≈ 0.8 GB  (197M × 4 bytes)
    optimizer state      ≈ 1.6 GB  (AdamW: 2× params)
    gradients            ≈ 0.8 GB
    total                ≈ 7.2 GB  (fits)
  Gradient checkpointing enabled to add safety margin.

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase18_convnextv2_train.py
"""

import os
import sys
import gc
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import mlflow
import timm
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

# Force UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_BASE = BASE_DIR / "experiments"
PHASE_DIR = CKPT_BASE / "phase18_convnextv2_large"
PHASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 7
CLASS_NAMES  = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']

# Hyperparameters (hardcoded with rationale — not parameterized)
INPUT_SIZE   = 384         # ConvNeXt-V2 Large was pretrained at 384
BATCH_SIZE   = 4           # 8GB VRAM safe with AMP + grad checkpointing
ACCUM_STEPS  = 8           # effective batch = 32
EPOCHS       = 25          # OneCycleLR warmup + decay
LR_HEAD      = 1e-3        # randomly-initialized 7-class head
LR_BACKBONE  = 3e-6        # 197M params — lower than typical 1e-5 to avoid overfitting
WEIGHT_DECAY = 1e-4
WARMUP_PCT   = 0.10        # 10% of total steps
GRAD_CLIP    = 1.0         # standard gradient clipping
MODEL_ID     = "convnextv2_large.fcmae_ft_in22k_in1k_384"

torch.manual_seed(42)
np.random.seed(42)

# Project imports
sys.path.append(str(BASE_DIR / "src"))
from dataset import load_metadata, split_dataset, get_dataloaders
from transforms import get_train_transform, get_val_transform


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    Build ConvNeXt-V2 Large with timm.
    The pretrained head is replaced for 7-class output.
    Gradient checkpointing enabled for 8GB VRAM safety.
    """
    print(f"\nBuilding {MODEL_ID}...")
    model = timm.create_model(
        MODEL_ID,
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    # Enable gradient checkpointing on the backbone stages
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
        print("  Gradient checkpointing: enabled")
    else:
        print("  Gradient checkpointing: not available on this model")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parameters trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    return model.to(DEVICE)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Discriminative LR: lower for backbone (pretrained), higher for head (random).
    timm ConvNeXt names the classifier 'head' (or 'head.fc').
    """
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # timm convnext classifier is at .head (Linear or Sequential ending in fc)
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    print(f"  Head params    : {sum(p.numel() for p in head_params):,}")
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ── Train / eval loops ─────────────────────────────────────────────────────────

def _autocast():
    """torch.amp.autocast — replaces deprecated torch.cuda.amp.autocast."""
    return torch.amp.autocast(device_type=DEVICE)


def train_epoch(model, loader, optimizer, scheduler, scaler) -> float:
    """One training epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with _autocast():
            logits = model(images)
            loss   = criterion(logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    """Evaluate on a loader. Returns macro AUC, accuracy, and per-class data."""
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        with _autocast():
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Normalize for roc_auc_score (float32 accumulation can shift row sums)
    all_probs = np.clip(all_probs, 1e-10, 1.0)
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    acc = (all_probs.argmax(axis=1) == all_labels).mean()
    return {"auc": auc, "acc": acc, "probs": all_probs, "labels": all_labels}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 18: ConvNeXt-V2 Large (FCMAE → in22k → in1k → HAM10000)")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_metadata(str(DATA_DIR))
    train_df, val_df, test_df = split_dataset(df)

    train_tf = get_train_transform(INPUT_SIZE)
    val_tf   = get_val_transform(INPUT_SIZE)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, train_tf, val_tf,
        batch_size=BATCH_SIZE,
    )

    # ── Model + optimizer + scheduler ─────────────────────────────────────────
    model     = build_model()
    optimizer = build_optimizer(model)

    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps,
        pct_start=WARMUP_PCT,
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nTraining schedule:")
    print(f"  Effective batch size : {BATCH_SIZE * ACCUM_STEPS}")
    print(f"  Steps per epoch      : {len(train_loader) // ACCUM_STEPS}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps         : {int(total_steps * WARMUP_PCT)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    mlflow.set_experiment("skin-lesion-all-phases")
    best_auc = 0.0

    with mlflow.start_run(run_name="phase18_convnextv2_large"):
        mlflow.log_params({
            "model_id":     MODEL_ID,
            "input_size":   INPUT_SIZE,
            "batch_size":   BATCH_SIZE,
            "accum_steps":  ACCUM_STEPS,
            "epochs":       EPOCHS,
            "lr_head":      LR_HEAD,
            "lr_backbone":  LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_pct":   WARMUP_PCT,
            "grad_clip":    GRAD_CLIP,
            "phase":        18,
        })

        for epoch in range(EPOCHS):
            print(f"\n── Epoch {epoch + 1}/{EPOCHS} ──")

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_m      = evaluate(model, val_loader)

            print(f"  Loss: {train_loss:.4f}  |  "
                  f"Val AUC: {val_m['auc']:.4f}  |  Val Acc: {val_m['acc']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_auc":    val_m["auc"],
                "val_acc":    val_m["acc"],
            }, step=epoch)

            if val_m["auc"] > best_auc:
                best_auc  = val_m["auc"]
                ckpt_path = PHASE_DIR / "best_model.pth"
                torch.save(model.state_dict(), str(ckpt_path))
                print(f"  [BEST] New best Val AUC: {best_auc:.4f}  →  {ckpt_path}")

        # ── Test set evaluation ───────────────────────────────────────────────
        print("\n── Test set evaluation ──")
        # Reload best weights
        best_sd = torch.load(str(PHASE_DIR / "best_model.pth"), map_location=DEVICE)
        model.load_state_dict(best_sd)
        test_m = evaluate(model, test_loader)

        print(f"\nTest AUC: {test_m['auc']:.4f}  |  Test Acc: {test_m['acc']:.4f}")
        mlflow.log_metrics({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

        # ── Save results ──────────────────────────────────────────────────────
        with open(str(PHASE_DIR / "results.txt"), "w", encoding="utf-8") as f:
            f.write(f"Phase 18: ConvNeXt-V2 Large @ {INPUT_SIZE}px\n")
            f.write(f"Model: {MODEL_ID}\n\n")
            f.write(f"Best Val AUC : {best_auc:.4f}\n")
            f.write(f"Test AUC     : {test_m['auc']:.4f}\n")
            f.write(f"Test Acc     : {test_m['acc']:.4f}\n\n")
            f.write(classification_report(
                test_m["labels"], test_m["probs"].argmax(axis=1),
                target_names=CLASS_NAMES,
            ))
        print(f"Results saved to {PHASE_DIR / 'results.txt'}")

    # Cleanup
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'=' * 65}")
    print(f"Phase 18 complete. Best Val AUC: {best_auc:.4f}")
    print(f"{'=' * 65}")
    return best_auc


if __name__ == "__main__":
    main()
