"""
phase19_mel_binary_train.py
----------------------------
Trains a binary mel-vs-rest classifier — a melanoma SPECIALIST that dedicates
100% of its capacity to the mel/nv decision boundary.

WHY THIS MATTERS
----------------
Per-class AUC analysis of the current ensemble:
  vasc:  0.9998  (saturated)
  df:    1.0000  (saturated)
  bcc:   0.9960  (near-saturated)
  akiec: 0.9903
  bkl:   0.9568
  nv:    0.9778
  mel:   0.9528  ← the only class with real movement room

A 7-class model splits its representational capacity across 7 decision
boundaries. A binary classifier trained ONLY on "is this mel? yes/no" can
learn a sharper mel/nv boundary because the gradient signal is concentrated
on that single distinction.

The output P(mel) is added as a 7th input feature dimension to the meta-learner
in Phase 21 (or used as a confidence multiplier on the mel column of the main
ensemble — meta-learner decides which works better).

ARCHITECTURE
------------
DINOv2+Registers backbone (same as Phase 10 and 13) + binary head with
focal loss. We reuse the Phase 10 fully-fine-tuned checkpoint as the starting
point — it has already learned strong dermoscopic features, so we only need
to retrain the head and let the late backbone layers adjust.

  Loss      : Binary focal loss (gamma=2.0, alpha=0.75 for mel positive class)
  Backbone  : Phase 10 checkpoint, last 4 layers unfrozen
  Head      : Linear(768 → 2)
  Sampler   : Custom 1:1 mel:non-mel sampling (NOT inverse-frequency from
              dataset.get_weighted_sampler, because we need 50/50 mel exposure)

WHY 1:1 SAMPLING (not inverse frequency)
-----------------------------------------
Inverse-frequency sampling oversamples ALL rare classes (mel, df, vasc, etc.)
proportionally. For binary mel-vs-rest training we want maximum mel exposure
specifically — 1:1 mel:non-mel makes every gradient step have equal positive
and negative signal.

EXPECTED OUTCOMES
------------------
  Mel binary AUC : 0.96–0.98  (vs 0.9528 for 7-class ensemble on mel)
  When fed into Phase 21 meta-learner: +0.005–0.010 on macro AUC,
  primarily through improving mel column.

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase19_mel_binary_train.py
"""

import os
import sys
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
from transformers import AutoModel
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_BASE = BASE_DIR / "experiments"
PHASE_DIR = CKPT_BASE / "phase19_mel_binary"
PHASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2          # binary: 0 = not-mel, 1 = mel
INPUT_SIZE  = 224        # match DINOv2's pretraining resolution

# Hyperparameters
BATCH_SIZE   = 16
ACCUM_STEPS  = 4         # effective batch = 64
EPOCHS       = 15        # binary problem converges fast
LR_HEAD      = 1e-3
LR_BACKBONE  = 5e-6      # match Phase 10's careful backbone LR
WEIGHT_DECAY = 1e-4
WARMUP_PCT   = 0.10
GRAD_CLIP    = 1.0

# Focal loss params (chosen for hard-positive mining)
FOCAL_GAMMA = 2.0        # standard, focuses on misclassified examples
FOCAL_ALPHA = 0.75       # weight for the mel positive class (boost recall)

# Number of unfrozen final transformer layers (DINOv2-base has 12)
N_UNFROZEN_LAYERS = 4

torch.manual_seed(42)
np.random.seed(42)

sys.path.append(str(BASE_DIR / "src"))
from dataset import load_metadata, split_dataset
from transforms import get_train_transform, get_val_transform


# ── Binary dataset ─────────────────────────────────────────────────────────────

class BinaryMelDataset(Dataset):
    """
    HAM10000 dataset with binary labels: 1 if mel, 0 otherwise.
    Uses albumentations transform pipeline (same format as HAM10000Dataset).
    """

    def __init__(self, df: pd.DataFrame, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        # Pre-compute binary labels
        self.binary_labels = (self.df["dx"] == "mel").astype(np.int64).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, int(self.binary_labels[idx])


def get_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    1:1 mel:non-mel sampling.
    Each mel sample gets weight 1 / n_mel.
    Each non-mel sample gets weight 1 / n_non_mel.
    Result: every batch is approximately 50% mel, 50% non-mel.
    """
    is_mel       = (df["dx"] == "mel").values
    n_mel        = int(is_mel.sum())
    n_non_mel    = len(df) - n_mel
    weights      = np.where(is_mel, 1.0 / n_mel, 1.0 / n_non_mel)
    print(f"  Balanced sampler: n_mel={n_mel}, n_non_mel={n_non_mel}, "
          f"target ratio 1:1")
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(df),
        replacement=True,
    )


# ── Model ──────────────────────────────────────────────────────────────────────

class DINOv2BinaryClassifier(nn.Module):
    """
    DINOv2+Registers backbone with a 2-class linear head.
    Loaded from Phase 10 checkpoint with the original 7-class head replaced.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base"
        )
        hidden_size = self.backbone.config.hidden_size  # 768
        self.head   = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


def load_from_phase10(model: DINOv2BinaryClassifier, ckpt_path: Path):
    """
    Load Phase 10 checkpoint into our binary model.
    The 7-class head won't fit our 2-class head — we load only the backbone weights.
    """
    print(f"Loading Phase 10 backbone from {ckpt_path}...")
    try:
        sd = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    except TypeError:
        sd = torch.load(str(ckpt_path), map_location=DEVICE)

    # Phase 10 saved state has keys like 'backbone.embeddings.*' and 'head.*'
    # We only want backbone.* keys; we'll let head.* remain freshly initialized.
    backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)

    # Only complain about unexpected keys (head from old checkpoint)
    head_missing_only = [k for k in missing if not k.startswith("head.")]
    if head_missing_only:
        print(f"  WARNING: missing keys not in head: {head_missing_only[:3]}...")
    print(f"  Loaded {len(backbone_sd)} backbone tensors.")
    print(f"  Head reinitialized for {NUM_CLASSES} classes.")


def freeze_early_layers(model: DINOv2BinaryClassifier, n_unfrozen: int = N_UNFROZEN_LAYERS):
    """
    Freeze all backbone parameters except:
      - the last n_unfrozen transformer layers
      - the layernorm at the end
    Head is always trainable.
    """
    # Freeze everything in backbone first
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer layers
    encoder_layers = model.backbone.encoder.layer
    n_total = len(encoder_layers)
    for i in range(n_total - n_unfrozen, n_total):
        for p in encoder_layers[i].parameters():
            p.requires_grad = True

    # Unfreeze final layernorm if present
    if hasattr(model.backbone, "layernorm"):
        for p in model.backbone.layernorm.parameters():
            p.requires_grad = True

    # Head always trainable
    for p in model.head.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Unfrozen last {n_unfrozen}/{n_total} transformer layers + head")
    print(f"  Trainable: {n_trainable:,} / {n_total_p:,} "
          f"({100 * n_trainable / n_total_p:.1f}%)")


# ── Focal loss ─────────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification with class weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where alpha_t = alpha if target == 1 (mel) else (1 - alpha).
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2), targets: (B,) with values in {0, 1}
        log_probs = F.log_softmax(logits, dim=1)            # (B, 2)
        probs     = log_probs.exp()                         # (B, 2)

        # Pick the log-probs and probs for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)

        # alpha_t: alpha for positives (mel), 1-alpha for negatives
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )

        loss = -alpha_t * (1 - pt).pow(self.gamma) * log_pt
        return loss.mean()


# ── Train / eval ──────────────────────────────────────────────────────────────

def _autocast():
    return torch.amp.autocast(device_type=DEVICE)


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion) -> float:
    model.train()
    total_loss = 0.0
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_binary(model, loader) -> dict:
    """Binary evaluation: AUC of P(mel) vs binary labels, plus accuracy."""
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        with _autocast():
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)   # (N, 2)
    all_labels = np.concatenate(all_labels)  # (N,)

    # Binary AUC: P(mel) vs is_mel
    p_mel = all_probs[:, 1]
    auc   = roc_auc_score(all_labels, p_mel)
    acc   = (all_probs.argmax(axis=1) == all_labels).mean()

    # Mel-specific metrics
    mel_recall = ((all_probs.argmax(axis=1) == 1) & (all_labels == 1)).sum() / max(
        (all_labels == 1).sum(), 1
    )
    mel_precision = ((all_probs.argmax(axis=1) == 1) & (all_labels == 1)).sum() / max(
        (all_probs.argmax(axis=1) == 1).sum(), 1
    )

    return {
        "auc":           auc,
        "acc":           acc,
        "mel_recall":    mel_recall,
        "mel_precision": mel_precision,
        "probs":         all_probs,
        "labels":        all_labels,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 19: Binary Mel-vs-Rest Specialist")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_metadata(str(DATA_DIR))
    train_df, val_df, test_df = split_dataset(df)

    n_mel_train = (train_df["dx"] == "mel").sum()
    n_mel_val   = (val_df["dx"]   == "mel").sum()
    n_mel_test  = (test_df["dx"]  == "mel").sum()
    print(f"\nMel positives: train={n_mel_train}  val={n_mel_val}  test={n_mel_test}")

    train_tf = get_train_transform(INPUT_SIZE)
    val_tf   = get_val_transform(INPUT_SIZE)

    train_ds = BinaryMelDataset(train_df, train_tf)
    val_ds   = BinaryMelDataset(val_df,   val_tf)
    test_ds  = BinaryMelDataset(test_df,  val_tf)

    sampler = get_balanced_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DINOv2BinaryClassifier(NUM_CLASSES)
    phase10_ckpt = CKPT_BASE / "phase10_dinov2_registers_finetune" / "best_model.pth"

    if not phase10_ckpt.exists():
        print(f"WARNING: Phase 10 checkpoint not found at {phase10_ckpt}")
        print("         Initializing from HuggingFace pretrained weights only.")
    else:
        load_from_phase10(model, phase10_ckpt)

    freeze_early_layers(model, N_UNFROZEN_LAYERS)
    model = model.to(DEVICE)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps,
        pct_start=WARMUP_PCT,
    )
    scaler    = torch.amp.GradScaler("cuda")
    criterion = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    print(f"\nTraining setup:")
    print(f"  Loss              : BinaryFocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    print(f"  Effective batch   : {BATCH_SIZE * ACCUM_STEPS}")
    print(f"  Total opt. steps  : {total_steps}")
    print(f"  Sampler           : 1:1 mel:non-mel (WeightedRandomSampler)")

    # ── Train ─────────────────────────────────────────────────────────────────
    mlflow.set_experiment("skin-lesion-all-phases")
    best_auc = 0.0

    with mlflow.start_run(run_name="phase19_mel_binary"):
        mlflow.log_params({
            "input_size":     INPUT_SIZE,
            "batch_size":     BATCH_SIZE,
            "accum_steps":    ACCUM_STEPS,
            "epochs":         EPOCHS,
            "lr_head":        LR_HEAD,
            "lr_backbone":    LR_BACKBONE,
            "focal_gamma":    FOCAL_GAMMA,
            "focal_alpha":    FOCAL_ALPHA,
            "n_unfrozen":     N_UNFROZEN_LAYERS,
            "phase":          19,
            "init_from":      "phase10" if phase10_ckpt.exists() else "imagenet",
        })

        for epoch in range(EPOCHS):
            print(f"\n── Epoch {epoch + 1}/{EPOCHS} ──")
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, criterion,
            )
            val_m = evaluate_binary(model, val_loader)

            print(f"  Loss: {train_loss:.4f}  |  "
                  f"Val Mel-AUC: {val_m['auc']:.4f}  |  "
                  f"Mel Recall: {val_m['mel_recall']:.4f}  |  "
                  f"Mel Precision: {val_m['mel_precision']:.4f}")

            mlflow.log_metrics({
                "train_loss":    train_loss,
                "val_mel_auc":   val_m["auc"],
                "val_acc":       val_m["acc"],
                "mel_recall":    val_m["mel_recall"],
                "mel_precision": val_m["mel_precision"],
            }, step=epoch)

            if val_m["auc"] > best_auc:
                best_auc = val_m["auc"]
                torch.save(model.state_dict(), str(PHASE_DIR / "best_model.pth"))
                print(f"  [BEST] New best Val Mel-AUC: {best_auc:.4f}")

        # ── Test set evaluation ───────────────────────────────────────────────
        print("\n── Test set evaluation ──")
        best_sd = torch.load(
            str(PHASE_DIR / "best_model.pth"),
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(best_sd)
        test_m = evaluate_binary(model, test_loader)

        print(f"\nTest Mel-AUC : {test_m['auc']:.4f}")
        print(f"Test Acc     : {test_m['acc']:.4f}")
        print(f"Mel Recall   : {test_m['mel_recall']:.4f}")
        print(f"Mel Precision: {test_m['mel_precision']:.4f}")

        mlflow.log_metrics({
            "test_mel_auc":       test_m["auc"],
            "test_acc":           test_m["acc"],
            "test_mel_recall":    test_m["mel_recall"],
            "test_mel_precision": test_m["mel_precision"],
        })

        # ── Save results ──────────────────────────────────────────────────────
        with open(str(PHASE_DIR / "results.txt"), "w", encoding="utf-8") as f:
            f.write("Phase 19: Binary Mel-vs-Rest Specialist\n")
            f.write(f"Backbone: DINOv2+Registers (Phase 10 init), "
                    f"last {N_UNFROZEN_LAYERS} layers + head trained\n")
            f.write(f"Loss: BinaryFocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})\n\n")
            f.write(f"Best Val Mel-AUC : {best_auc:.4f}\n")
            f.write(f"Test Mel-AUC     : {test_m['auc']:.4f}\n")
            f.write(f"Test Mel Recall  : {test_m['mel_recall']:.4f}\n")
            f.write(f"Test Mel Precision: {test_m['mel_precision']:.4f}\n\n")

            preds_binary = test_m["probs"].argmax(axis=1)
            f.write(classification_report(
                test_m["labels"], preds_binary,
                target_names=["non-mel", "mel"],
            ))
        print(f"Results saved to {PHASE_DIR / 'results.txt'}")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'=' * 65}")
    print(f"Phase 19 complete. Best Val Mel-AUC: {best_auc:.4f}")
    print(f"{'=' * 65}")
    return best_auc


if __name__ == "__main__":
    main()
