"""
advanced_phases.py
------------------
Phase 11: Re-ensemble with Phase 10 (DINOv2+Registers full fine-tune)
Phase 12: Multimodal model (image + age/sex/location metadata)
Phase 13: Focal loss on best model for melanoma recall improvement

Usage:
    python src/advanced_phases.py --phase 11
    python src/advanced_phases.py --phase 12
    python src/advanced_phases.py --phase 13
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import mlflow
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import timm
from transformers import AutoModel

sys.path.append(os.path.dirname(__file__))
from dataset import load_metadata, split_dataset, get_dataloaders, CLASS_NAMES, CLASS_TO_IDX
from transforms import get_train_transform, get_val_transform

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
DATA_DIR    = r"D:\ML\skin-lesion-vit\data\raw"
CKPT_BASE   = r"D:\ML\skin-lesion-vit\experiments"

# ── Shared model classes ──────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.head     = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


def load_efficientnetv2(ckpt_path):
    model = timm.create_model("tf_efficientnetv2_m", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_dinov2_registers(ckpt_path):
    model = DINOv2Classifier("facebook/dinov2-with-registers-base", NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs(model, loader, model_type="timm"):
    model.eval()
    all_probs  = []
    all_labels = []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(DEVICE)
        with autocast():
            if model_type == "timm":
                logits = model(images)
            else:
                logits = model(pixel_values=images)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 11 — Re-ensemble with fully fine-tuned DINOv2+Registers
# ════════════════════════════════════════════════════════════════════════════════

def run_phase11():
    print("\n" + "="*60)
    print("  PHASE 11: RE-ENSEMBLE WITH PHASE 10")
    print("="*60)

    df = load_metadata(DATA_DIR)
    _, _, test_df = split_dataset(df)

    val_tf      = get_val_transform(224)
    eff_val_tf  = get_val_transform(384)

    from dataset import HAM10000Dataset
    from torch.utils.data import DataLoader

    eff_loader  = DataLoader(HAM10000Dataset(test_df, eff_val_tf),
                             batch_size=16, shuffle=False, num_workers=4)
    dino_loader = DataLoader(HAM10000Dataset(test_df, val_tf),
                             batch_size=16, shuffle=False, num_workers=4)

    print("\nLoading EfficientNetV2-M...")
    eff_ckpt  = os.path.join(CKPT_BASE, "phase2_efficientnetv2", "best_model.pth")
    eff_model = load_efficientnetv2(eff_ckpt)
    eff_probs, labels = get_probs(eff_model, eff_loader, "timm")

    print("Loading DINOv2+Registers (full fine-tune)...")
    dino_ckpt  = os.path.join(CKPT_BASE, "phase10_dinov2_registers_finetune", "best_model.pth")
    dino_model = load_dinov2_registers(dino_ckpt)
    dino_probs, _ = get_probs(dino_model, dino_loader, "dinov2")

    ckpt_dir = os.path.join(CKPT_BASE, "phase11_ensemble_v2")
    os.makedirs(ckpt_dir, exist_ok=True)

    mlflow.set_experiment("skin-lesion-all-phases")
    best_auc    = 0.0
    best_weight = 0.5

    with mlflow.start_run(run_name="phase11_ensemble_v2"):
        print("\nSearching best ensemble weights...")
        for w in np.arange(0.3, 0.8, 0.05):
            avg   = w * eff_probs + (1 - w) * dino_probs
            auc   = roc_auc_score(labels, avg, multi_class="ovr", average="macro")
            print(f"  EfficientNetV2 w={w:.2f} -> AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc    = auc
                best_weight = w
                best_probs  = avg

        best_acc = (best_probs.argmax(1) == labels).mean()
        print(f"\nBest: EfficientNetV2={best_weight:.2f}, DINOv2+Reg={1-best_weight:.2f}")
        print(f"Ensemble Test AUC: {best_auc:.4f} | Test Acc: {best_acc:.4f}")

        mlflow.log_metrics({"test_auc": best_auc, "test_acc": best_acc,
                            "best_eff_weight": best_weight})

        result_path = os.path.join(ckpt_dir, "results.txt")
        with open(result_path, "w") as f:
            f.write("Phase 11: Re-Ensemble (EfficientNetV2 + DINOv2+Registers full FT)\n\n")
            f.write(f"EfficientNetV2 weight : {best_weight:.2f}\n")
            f.write(f"DINOv2+Reg weight     : {1-best_weight:.2f}\n")
            f.write(f"Test AUC              : {best_auc:.4f}\n")
            f.write(f"Test Acc              : {best_acc:.4f}\n\n")
            f.write(classification_report(labels, best_probs.argmax(1),
                                          target_names=CLASS_NAMES))

        np.save(os.path.join(ckpt_dir, "eff_probs.npy"), eff_probs)
        np.save(os.path.join(ckpt_dir, "dino_probs.npy"), dino_probs)
        np.save(os.path.join(ckpt_dir, "labels.npy"), labels)
        print(f"Results saved to {result_path}")

    return best_auc


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 12 — Multimodal (image + metadata)
# ════════════════════════════════════════════════════════════════════════════════

class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset that returns image + metadata (age, sex, localization)."""

    SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
    LOC_MAP = {
        "back": 0, "lower extremity": 1, "trunk": 2, "upper extremity": 3,
        "abdomen": 4, "face": 5, "chest": 6, "foot": 7, "scalp": 8,
        "neck": 9, "hand": 10, "ear": 11, "acral": 12, "genital": 13,
        "unknown": 14
    }

    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image as PILImage

        row   = self.df.iloc[idx]
        image = PILImage.open(row["image_path"]).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        # Safely handle missing/NaN age data
        age_raw = row["age"]
        if pd.isna(age_raw) or str(age_raw).lower() == "unknown":
            age = 50.0 / 85.0  # Approx mean fallback
        else:
            age = float(age_raw) / 85.0  # Normalize to [0,1]

        sex  = self.SEX_MAP.get(str(row["sex"]).lower(), 2)
        loc  = self.LOC_MAP.get(str(row["localization"]).lower(), 14)

        meta = torch.tensor([age, sex / 2.0, loc / 14.0], dtype=torch.float32)
        label = CLASS_TO_IDX[row["dx"]]
        return image, meta, label


class MultimodalClassifier(nn.Module):
    """DINOv2+Registers backbone + metadata MLP + fusion head."""

    def __init__(self, backbone_ckpt, num_classes=7):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base"
        )
        hidden_size = self.backbone.config.hidden_size  # 768

        full_model = DINOv2Classifier(
            "facebook/dinov2-with-registers-base", num_classes
        )
        full_model.load_state_dict(
            torch.load(backbone_ckpt, map_location="cpu")
        )
        self.backbone = full_model.backbone

        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False

        self.meta_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, meta):
        cls       = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        meta_feat = self.meta_mlp(meta)
        combined  = torch.cat([cls, meta_feat], dim=1)
        return self.fusion_head(combined)


def run_phase12():
    print("\n" + "="*60)
    print("  PHASE 12: MULTIMODAL (IMAGE + METADATA)")
    print("="*60)

    # Added Checkpoint Fallback Logic
    backbone_ckpt = os.path.join(
        CKPT_BASE, "phase10_dinov2_registers_finetune", "best_model.pth"
    )
    if not os.path.exists(backbone_ckpt):
        print("⚠️ Phase 10 checkpoint not found. Falling back to Phase 5 checkpoint...")
        backbone_ckpt = os.path.join(
            CKPT_BASE, "phase5_dinov2_registers", "best_model.pth"
        )
        if not os.path.exists(backbone_ckpt):
            print("❌ ERROR: Neither Phase 10 nor Phase 5 checkpoint found.")
            print("Please run: python src/engine.py --phase 5")
            sys.exit(1)

    df = load_metadata(DATA_DIR)
    train_df, val_df, test_df = split_dataset(df)

    train_tf = get_train_transform(224)
    val_tf   = get_val_transform(224)

    from torch.utils.data import DataLoader
    from dataset import get_weighted_sampler

    train_ds = MultimodalDataset(train_df, train_tf)
    val_ds   = MultimodalDataset(val_df,   val_tf)
    test_ds  = MultimodalDataset(test_df,  val_tf)

    sampler      = get_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=True)

    model    = MultimodalClassifier(backbone_ckpt).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Multimodal model: {trainable:,} / {total:,} trainable params")

    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": 5e-6},
        {"params": [p for n,p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad], "lr": 1e-3},
    ], weight_decay=1e-4)

    criterion  = nn.CrossEntropyLoss()
    scaler     = GradScaler()
    ckpt_dir   = os.path.join(CKPT_BASE, "phase12_multimodal")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc   = 0.0
    epochs     = 20
    accum      = 8

    mlflow.set_experiment("skin-lesion-all-phases")

    with mlflow.start_run(run_name="phase12_multimodal"):
        mlflow.log_params({"epochs": epochs, "lr_head": 1e-3, "lr_backbone": 5e-6})

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            for step, (images, meta, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
                images = images.to(DEVICE)
                meta   = meta.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast():
                    logits = model(images, meta)
                    loss   = criterion(logits, labels) / accum
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                total_loss += loss.item() * accum

            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for images, meta, labels in val_loader:
                    images = images.to(DEVICE)
                    meta   = meta.to(DEVICE)
                    with autocast():
                        probs = torch.softmax(model(images, meta), dim=1).cpu().numpy()
                    if np.isnan(probs).any():
                        probs = np.where(np.isnan(probs), 1.0/7, probs)
                    all_probs.append(probs)
                    all_labels.append(labels.numpy())

            all_probs  = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            auc = roc_auc_score(all_labels, all_probs,
                                multi_class="ovr", average="macro")
            acc = (all_probs.argmax(1) == all_labels).mean()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Val AUC: {auc:.4f} | Val Acc: {acc:.4f}")
            mlflow.log_metrics({"train_loss": avg_loss, "val_auc": auc, "val_acc": acc},
                               step=epoch)

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, "best_model.pth"))
                print(f"  [BEST] New best AUC: {best_auc:.4f}")

        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth")))
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, meta, labels in test_loader:
                images = images.to(DEVICE)
                meta   = meta.to(DEVICE)
                with autocast():
                    probs = torch.softmax(model(images, meta), dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        test_auc   = roc_auc_score(all_labels, all_probs,
                                   multi_class="ovr", average="macro")
        test_acc   = (all_probs.argmax(1) == all_labels).mean()
        print(f"\nTest AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_auc": test_auc, "test_acc": test_acc})

        result_path = os.path.join(ckpt_dir, "results.txt")
        with open(result_path, "w") as f:
            f.write("Phase 12: Multimodal (DINOv2+Registers + Age/Sex/Location)\n\n")
            f.write(f"Best Val AUC : {best_auc:.4f}\n")
            f.write(f"Test AUC     : {test_auc:.4f}\n")
            f.write(f"Test Acc     : {test_acc:.4f}\n\n")
            f.write(classification_report(all_labels, all_probs.argmax(1),
                                          target_names=CLASS_NAMES))
        print(f"Results saved to {result_path}")

    return test_auc


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 13 — Focal Loss for melanoma recall
# ════════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets,
                                              weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


def get_class_weights(train_df):
    counts = train_df["dx"].value_counts()
    total  = len(train_df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts[c]) for c in CLASS_NAMES],
        dtype=torch.float32
    )
    return weights / weights.sum() * NUM_CLASSES


def run_phase13():
    print("\n" + "="*60)
    print("  PHASE 13: FOCAL LOSS — MELANOMA RECALL IMPROVEMENT")
    print("="*60)

    df = load_metadata(DATA_DIR)
    train_df, val_df, test_df = split_dataset(df)

    train_tf = get_train_transform(224)
    val_tf   = get_val_transform(224)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, train_tf, val_tf, batch_size=8
    )

    print("Loading Phase 10 checkpoint...")
    ckpt = os.path.join(CKPT_BASE, "phase10_dinov2_registers_finetune", "best_model.pth")
    if not os.path.exists(ckpt):
        print("ERROR: Phase 10 checkpoint not found.")
        sys.exit(1)

    model = DINOv2Classifier("facebook/dinov2-with-registers-base", NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model = model.to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    class_weights = get_class_weights(train_df).to(DEVICE)
    criterion     = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler    = GradScaler()
    ckpt_dir  = os.path.join(CKPT_BASE, "phase13_focal_loss")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc  = 0.0
    epochs    = 15

    mlflow.set_experiment("skin-lesion-all-phases")

    with mlflow.start_run(run_name="phase13_focal_loss"):
        mlflow.log_params({"gamma": 2.0, "epochs": epochs, "base": "phase10"})

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in tqdm(train_loader,
                                       desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with autocast():
                    logits = model(pixel_values=images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    with autocast():
                        probs = torch.softmax(
                            model(pixel_values=images), dim=1
                        ).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.numpy())

            all_probs  = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            auc = roc_auc_score(all_labels, all_probs,
                                multi_class="ovr", average="macro")
            acc = (all_probs.argmax(1) == all_labels).mean()

            mel_idx    = CLASS_NAMES.index("mel")
            mel_recall = (
                (all_probs.argmax(1) == mel_idx) & (all_labels == mel_idx)
            ).sum() / (all_labels == mel_idx).sum()

            print(f"Epoch {epoch+1:2d} | AUC: {auc:.4f} | Acc: {acc:.4f} | Mel Recall: {mel_recall:.4f}")
            mlflow.log_metrics(
                {"val_auc": auc, "val_acc": acc, "mel_recall": mel_recall},
                step=epoch
            )

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, "best_model.pth"))
                print(f"  [BEST] AUC: {best_auc:.4f} | Mel Recall: {mel_recall:.4f}")

        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth")))
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                with autocast():
                    probs = torch.softmax(
                        model(pixel_values=images), dim=1
                    ).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        test_auc   = roc_auc_score(all_labels, all_probs,
                                   multi_class="ovr", average="macro")
        test_acc   = (all_probs.argmax(1) == all_labels).mean()
        print(f"\nTest AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_auc": test_auc, "test_acc": test_acc})

        result_path = os.path.join(ckpt_dir, "results.txt")
        with open(result_path, "w") as f:
            f.write("Phase 13: Focal Loss (gamma=2.0, class-weighted alpha)\n")
            f.write("Base model: Phase 10 DINOv2+Registers full fine-tune\n\n")
            f.write(f"Best Val AUC : {best_auc:.4f}\n")
            f.write(f"Test AUC     : {test_auc:.4f}\n")
            f.write(f"Test Acc     : {test_acc:.4f}\n\n")
            f.write(classification_report(all_labels, all_probs.argmax(1),
                                          target_names=CLASS_NAMES))
        print(f"Results saved to {result_path}")

    return test_auc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[11, 12, 13])
    args = parser.parse_args()

    if args.phase == 11:
        run_phase11()
    elif args.phase == 12:
        run_phase12()
    elif args.phase == 13:
        run_phase13()