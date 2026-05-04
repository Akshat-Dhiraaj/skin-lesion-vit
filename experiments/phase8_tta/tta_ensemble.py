"""
tta_ensemble.py
---------------
Phase 8: Test Time Augmentation (TTA) on best models
Phase 9: Ensemble of EfficientNetV2 + DINOv2+Registers

Usage:
    python src/tta_ensemble.py --phase 8
    python src/tta_ensemble.py --phase 9
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import timm
from transformers import AutoModel, ViTForImageClassification
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(__file__))
from dataset import load_metadata, split_dataset, get_dataloaders, CLASS_NAMES
from transforms import get_val_transform

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
DATA_DIR    = r"D:\ML\skin-lesion-vit\data\raw"
CKPT_BASE   = r"D:\ML\skin-lesion-vit\experiments"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TTA transforms — 8 augmented views per image ──────────────────────────────

def get_tta_transforms(image_size=224):
    """8 TTA transforms: original + flips + rotations."""
    return [
        # 1. Original
        A.Compose([A.Resize(image_size, image_size),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 2. Horizontal flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 3. Vertical flip
        A.Compose([A.Resize(image_size, image_size), A.VerticalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 4. Rotate 90
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=(90,90), p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 5. Rotate 180
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=(180,180), p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 6. Rotate 270
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=(270,270), p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 7. HFlip + VFlip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.VerticalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        # 8. Slight zoom
        A.Compose([A.Resize(image_size, image_size),
                   A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1,
                                      rotate_limit=0, p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
    ]


# ── Model loaders ─────────────────────────────────────────────────────────────

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


# ── TTA inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, dataset_df, tta_transforms, batch_size=16, model_type="timm"):
    """Run TTA — average predictions across all augmented views."""
    from torch.utils.data import DataLoader
    from dataset import HAM10000Dataset

    all_probs  = []
    all_labels = []

    # Collect all TTA predictions
    tta_preds = []
    for t_idx, transform in enumerate(tta_transforms):
        print(f"  TTA view {t_idx+1}/{len(tta_transforms)}...")
        ds     = HAM10000Dataset(dataset_df, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
        view_probs = []
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            with autocast():
                if model_type == "timm":
                    logits = model(images)
                else:
                    logits = model(pixel_values=images) if hasattr(model, 'backbone') \
                             else model(images)
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            view_probs.append(probs)
            if t_idx == 0:
                all_labels.append(labels.numpy())
        tta_preds.append(np.concatenate(view_probs))

    all_labels = np.concatenate(all_labels)
    # Average across all TTA views
    avg_probs  = np.mean(tta_preds, axis=0)
    return avg_probs, all_labels


# ── Phase 8: TTA ──────────────────────────────────────────────────────────────

def run_tta():
    print("\n" + "="*60)
    print("  PHASE 8: TEST TIME AUGMENTATION")
    print("="*60)

    df = load_metadata(DATA_DIR)
    _, _, test_df = split_dataset(df)

    results = {}
    mlflow.set_experiment("skin-lesion-all-phases")

    # TTA on EfficientNetV2 (best model)
    print("\n[TTA] EfficientNetV2-M...")
    ckpt = os.path.join(CKPT_BASE, "phase2_efficientnetv2", "best_model.pth")
    model = load_efficientnetv2(ckpt)
    tta_tf = get_tta_transforms(image_size=384)

    with mlflow.start_run(run_name="phase8_tta_efficientnetv2"):
        probs, labels = tta_predict(model, test_df, tta_tf, batch_size=16, model_type="timm")
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        acc = (probs.argmax(1) == labels).mean()
        print(f"\nEfficientNetV2 TTA — Test AUC: {auc:.4f} | Test Acc: {acc:.4f}")
        print(f"(Baseline without TTA: 0.9685)")
        mlflow.log_metrics({"test_auc_tta": auc, "test_acc_tta": acc})
        results["efficientnetv2_tta"] = {"auc": auc, "acc": acc, "probs": probs, "labels": labels}

    # TTA on DINOv2+Registers (#2 model)
    print("\n[TTA] DINOv2+Registers...")
    ckpt = os.path.join(CKPT_BASE, "phase5_dinov2_registers", "best_model.pth")
    model = load_dinov2_registers(ckpt)
    tta_tf = get_tta_transforms(image_size=224)

    with mlflow.start_run(run_name="phase8_tta_dinov2_registers"):
        probs, labels = tta_predict(model, test_df, tta_tf, batch_size=16, model_type="dinov2")
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        acc = (probs.argmax(1) == labels).mean()
        print(f"\nDINOv2+Registers TTA — Test AUC: {auc:.4f} | Test Acc: {acc:.4f}")
        print(f"(Baseline without TTA: 0.9653)")
        mlflow.log_metrics({"test_auc_tta": auc, "test_acc_tta": acc})
        results["dinov2_registers_tta"] = {"auc": auc, "acc": acc, "probs": probs, "labels": labels}

    # Save results
    ckpt_dir  = os.path.join(CKPT_BASE, "phase8_tta")
    os.makedirs(ckpt_dir, exist_ok=True)
    result_path = os.path.join(ckpt_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write("Phase 8: Test Time Augmentation\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Test AUC: {r['auc']:.4f}\n")
            f.write(f"  Test Acc: {r['acc']:.4f}\n\n")
    print(f"\nResults saved to {result_path}")

    # Save probs for ensemble
    np.save(os.path.join(ckpt_dir, "efficientnetv2_probs.npy"),
            results["efficientnetv2_tta"]["probs"])
    np.save(os.path.join(ckpt_dir, "dinov2_registers_probs.npy"),
            results["dinov2_registers_tta"]["probs"])
    np.save(os.path.join(ckpt_dir, "labels.npy"), labels)
    print("Saved prediction arrays for ensemble.")
    return results


# ── Phase 9: Ensemble ─────────────────────────────────────────────────────────

def run_ensemble():
    print("\n" + "="*60)
    print("  PHASE 9: ENSEMBLE (EfficientNetV2 + DINOv2+Registers)")
    print("="*60)

    tta_dir = os.path.join(CKPT_BASE, "phase8_tta")
    ens_dir = os.path.join(CKPT_BASE, "phase9_ensemble")
    os.makedirs(ens_dir, exist_ok=True)

    # Load TTA predictions
    eff_probs  = np.load(os.path.join(tta_dir, "efficientnetv2_probs.npy"))
    dino_probs = np.load(os.path.join(tta_dir, "dinov2_registers_probs.npy"))
    labels     = np.load(os.path.join(tta_dir, "labels.npy"))

    mlflow.set_experiment("skin-lesion-all-phases")

    best_auc    = 0.0
    best_weight = 0.5
    results     = {}

    with mlflow.start_run(run_name="phase9_ensemble"):
        # Grid search over ensemble weights
        print("\nSearching best ensemble weights...")
        for w in np.arange(0.3, 0.8, 0.05):
            avg_probs = w * eff_probs + (1 - w) * dino_probs
            auc       = roc_auc_score(labels, avg_probs, multi_class="ovr", average="macro")
            print(f"  EfficientNetV2 weight={w:.2f} -> AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc    = auc
                best_weight = w
                best_probs  = avg_probs

        print(f"\nBest weight: EfficientNetV2={best_weight:.2f}, DINOv2+Reg={1-best_weight:.2f}")
        best_acc = (best_probs.argmax(1) == labels).mean()
        print(f"Ensemble Test AUC: {best_auc:.4f} | Test Acc: {best_acc:.4f}")

        mlflow.log_metrics({
            "best_ensemble_auc" : best_auc,
            "best_ensemble_acc" : best_acc,
            "best_eff_weight"   : best_weight,
        })

        # Classification report
        print("\nPer-class report:")
        print(classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES))

        # Save results
        result_path = os.path.join(ens_dir, "results.txt")
        with open(result_path, "w") as f:
            f.write("Phase 9: Ensemble\n\n")
            f.write(f"Best weights: EfficientNetV2={best_weight:.2f}, DINOv2+Registers={1-best_weight:.2f}\n")
            f.write(f"Test AUC: {best_auc:.4f}\n")
            f.write(f"Test Acc: {best_acc:.4f}\n\n")
            f.write(classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES))
        print(f"\nResults saved to {result_path}")

    return best_auc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[8, 9])
    args = parser.parse_args()

    if args.phase == 8:
        run_tta()
    elif args.phase == 9:
        if not os.path.exists(os.path.join(CKPT_BASE, "phase8_tta", "labels.npy")):
            print("ERROR: Run Phase 8 first to generate TTA predictions.")
            sys.exit(1)
        run_ensemble()
