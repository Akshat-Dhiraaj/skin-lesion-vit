"""
engine.py
---------
Unified training engine for all 9 phases.
Swap MODEL_CONFIG to change phase — no other code changes needed.

Usage:
    python src/engine.py --phase 1
    python src/engine.py --phase 4
"""

import os
import sys
import argparse

# Fix Windows CP1252 encoding issue — force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import timm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    ViTForImageClassification,
)

# ── Add src to path ───────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from dataset import load_metadata, split_dataset, get_dataloaders
from transforms import get_train_transform, get_val_transform

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 7
DATA_DIR     = r"D:\ML\skin-lesion-vit\data\raw"
CKPT_BASE    = r"D:\ML\skin-lesion-vit\experiments"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Phase configs ─────────────────────────────────────────────────────────────
PHASE_CONFIGS = {
    1: {
        "name"            : "phase1_resnet50_frozen",
        "model_type"      : "timm",
        "model_id"        : "resnet50",
        "freeze_backbone" : True,
        "batch_size"      : 64,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 0.0,
        "epochs"          : 15,
        "accum_steps"     : 1,
        "input_size"      : 224,
    },
    2: {
        "name"            : "phase2_efficientnetv2",
        "model_type"      : "timm",
        "model_id"        : "tf_efficientnetv2_m",
        "freeze_backbone" : False,
        "batch_size"      : 16,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 1e-5,
        "epochs"          : 20,
        "accum_steps"     : 4,
        "input_size"      : 384,
    },
    3: {
        "name"            : "phase3_vit_base",
        "model_type"      : "vit_hf",
        "model_id"        : "google/vit-base-patch16-224",
        "freeze_backbone" : False,
        "freeze_layers"   : 8,
        "batch_size"      : 16,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 1e-5,
        "epochs"          : 20,
        "accum_steps"     : 4,
        "input_size"      : 224,
    },
    4: {
        "name"            : "phase4_dinov2",
        "model_type"      : "dinov2",
        "model_id"        : "facebook/dinov2-base",
        "freeze_backbone" : False,
        "batch_size"      : 16,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 1e-5,
        "epochs"          : 20,
        "accum_steps"     : 4,
        "input_size"      : 224,
    },
    5: {
        "name"            : "phase5_dinov2_registers",
        "model_type"      : "dinov2",
        "model_id"        : "facebook/dinov2-with-registers-base",
        "freeze_backbone" : False,
        "batch_size"      : 16,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 1e-5,
        "epochs"          : 20,
        "accum_steps"     : 4,
        "input_size"      : 224,
    },
    6: {
        "name"            : "phase6_aimv2",
        "model_type"      : "aimv2",
        "model_id"        : "apple/aimv2-large-patch14-224",
        "freeze_backbone" : True,   # frozen linear probe first — 307M params
        "batch_size"      : 16,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 0.0,
        "epochs"          : 20,
        "accum_steps"     : 4,
        "input_size"      : 224,
    },
    7: {
        "name"            : "phase7_siglip",
        "model_type"      : "siglip",
        "model_id"        : "google/siglip-so400m-patch14-224",
        "freeze_backbone" : True,   # linear probe — 400M vision encoder
        "batch_size"      : 8,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 0.0,
        "epochs"          : 20,
        "accum_steps"     : 8,
        "input_size"      : 224,
    },
    10: {
        "name"            : "phase10_dinov2_registers_finetune",
        "model_type"      : "dinov2",
        "model_id"        : "facebook/dinov2-with-registers-base",
        "freeze_backbone" : False,
        "batch_size"      : 8,
        "lr_head"         : 1e-3,
        "lr_backbone"     : 5e-6,
        "epochs"          : 25,
        "accum_steps"     : 8,
        "input_size"      : 224,
    },
}


# ── Model builders ────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear classification head."""
    def __init__(self, model_id, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone  = AutoModel.from_pretrained(model_id)
        hidden_size    = self.backbone.config.hidden_size
        self.head      = nn.Linear(hidden_size, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head(cls_token)


class AIMv2Classifier(nn.Module):
    """AIMv2 backbone + linear classification head via timm."""
    def __init__(self, model_id, num_classes, freeze_backbone=True):
        super().__init__()
        # Use timm's AIMv2 implementation — more stable than HF transformers
        self.backbone = timm.create_model(
            "aimv2_large_patch14_224",
            pretrained=True,
            num_classes=0  # remove classification head
        )
        hidden_size = self.backbone.num_features
        self.head   = nn.Linear(hidden_size, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        return self.head(features)


class SigLIPClassifier(nn.Module):
    """SigLIP-SO400M vision encoder + linear head."""
    def __init__(self, model_id, num_classes):
        super().__init__()
        from transformers import AutoModel
        full_model        = AutoModel.from_pretrained(model_id)
        self.vision_model = full_model.vision_model
        hidden_size       = self.vision_model.config.hidden_size
        self.head         = nn.Linear(hidden_size, num_classes)
        # Freeze vision encoder — linear probe only
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled  = outputs.pooler_output
        return self.head(pooled)


def build_model(cfg):
    """Build model from phase config."""
    mtype = cfg["model_type"]
    mid   = cfg["model_id"]
    nc    = NUM_CLASSES

    if mtype == "timm":
        model = timm.create_model(mid, pretrained=True, num_classes=nc)
        if cfg["freeze_backbone"]:
            for name, p in model.named_parameters():
                if "classifier" not in name and "fc" not in name and "head" not in name:
                    p.requires_grad = False

    elif mtype == "vit_hf":
        model = ViTForImageClassification.from_pretrained(
            mid, num_labels=nc, ignore_mismatched_sizes=True
        )
        # Freeze first N transformer blocks
        n_freeze = cfg.get("freeze_layers", 8)
        for p in model.parameters():
            p.requires_grad = False
        for i in range(n_freeze, 12):
            for p in model.vit.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif mtype == "dinov2":
        model = DINOv2Classifier(mid, nc, freeze_backbone=cfg["freeze_backbone"])

    elif mtype == "aimv2":
        model = AIMv2Classifier(mid, nc, freeze_backbone=cfg["freeze_backbone"])

    elif mtype == "siglip":
        model = SigLIPClassifier(mid, nc)

    else:
        raise ValueError(f"Unknown model_type: {mtype}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model: {mid}")
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    return model


def build_optimizer(model, cfg):
    head_params     = []
    backbone_params = []
    head_keywords   = ["classifier", "fc", "head"]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params and cfg["lr_backbone"] > 0:
        param_groups.append({"params": backbone_params, "lr": cfg["lr_backbone"]})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg["lr_head"]})

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ── Training loop ─────────────────────────────────────────────────────────────

def get_logits(model, batch, cfg):
    """Handle different model output formats."""
    mtype = cfg["model_type"]
    if mtype == "vit_hf":
        return model(batch).logits
    return model(batch)


def train_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with autocast():
            logits = get_logits(model, images, cfg)
            loss   = criterion(logits, labels) / cfg["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % cfg["accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg["accum_steps"]

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE)
        with autocast():
            logits = get_logits(model, images, cfg)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc        = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    acc        = (all_probs.argmax(1) == all_labels).mean()
    return {"auc": auc, "acc": acc, "probs": all_probs, "labels": all_labels}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_phase(phase_num):
    cfg  = PHASE_CONFIGS[phase_num]
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num}: {name.upper()}")
    print(f"{'='*60}\n")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Data
    df = load_metadata(DATA_DIR)
    train_df, val_df, test_df = split_dataset(df)

    train_tf = get_train_transform(cfg["input_size"])
    val_tf   = get_val_transform(cfg["input_size"])

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, train_tf, val_tf,
        batch_size=cfg["batch_size"]
    )

    # Model
    model = build_model(cfg).to(DEVICE)
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        model.vit.encoder.gradient_checkpointing = True

    optimizer = build_optimizer(model, cfg)
    scheduler = None
    if cfg["lr_backbone"] > 0:
        total_steps = (len(train_loader) // cfg["accum_steps"]) * cfg["epochs"]
        scheduler   = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[cfg["lr_backbone"], cfg["lr_head"]],
            total_steps=total_steps,
            pct_start=0.1,
        )

    scaler   = GradScaler()
    ckpt_dir = os.path.join(CKPT_BASE, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc = 0.0

    mlflow.set_experiment("skin-lesion-all-phases")

    with mlflow.start_run(run_name=name):
        mlflow.log_params({**cfg, "phase": phase_num, "device": DEVICE})

        for epoch in range(cfg["epochs"]):
            print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
            val_m      = evaluate(model, val_loader, cfg)

            print(f"  Loss: {train_loss:.4f} | Val AUC: {val_m['auc']:.4f} | Val Acc: {val_m['acc']:.4f}")
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_auc": val_m["auc"], "val_acc": val_m["acc"]},
                step=epoch
            )

            if val_m["auc"] > best_auc:
                best_auc  = val_m["auc"]
                ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [BEST] New best AUC: {best_auc:.4f}")

        # Final test evaluation
        print("\nRunning test set evaluation...")
        test_m = evaluate(model, test_loader, cfg)
        print(f"\nTest AUC: {test_m['auc']:.4f} | Test Acc: {test_m['acc']:.4f}")
        mlflow.log_metrics({"test_auc": test_m["auc"], "test_acc": test_m["acc"]})

        # Save results summary
        result_path = os.path.join(ckpt_dir, "results.txt")
        with open(result_path, "w") as f:
            f.write(f"Phase {phase_num}: {name}\n")
            f.write(f"Best Val AUC : {best_auc:.4f}\n")
            f.write(f"Test AUC     : {test_m['auc']:.4f}\n")
            f.write(f"Test Acc     : {test_m['acc']:.4f}\n\n")
            f.write(classification_report(
                test_m["labels"], test_m["probs"].argmax(1),
                target_names=["mel","nv","bcc","akiec","bkl","df","vasc"]
            ))
        print(f"Results saved to {result_path}")

    print(f"\nPhase {phase_num} complete. Best Val AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=list(PHASE_CONFIGS.keys()),
                        help="Phase number to run (1-7, 10)")
    args = parser.parse_args()
    run_phase(args.phase)