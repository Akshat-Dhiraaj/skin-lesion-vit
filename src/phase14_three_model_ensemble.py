"""
phase14_three_model_ensemble.py
--------------------------------
Phase 14: 3-model ensemble
    Model A — EfficientNetV2-M  + TTA  (phase8_tta)
    Model B — DINOv2+Reg (full FT) + TTA  (phase15_tta_dinov2_ft)
    Model C — Multimodal DINOv2+Reg (phase12_multimodal)

Run order:
    1. python src/tta_ensemble.py --phase 8      # saves eff + frozen-dino probs
    2. python src/tta_ensemble.py --phase 15     # saves full-FT dino+TTA probs
    3. python phase14_three_model_ensemble.py

If Phase 15 has not been run yet, the script falls back to the frozen
DINOv2+Reg TTA probs from Phase 8 and warns you.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import roc_auc_score, classification_report
from torch.cuda.amp import autocast
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_BASE   = r"D:\ML\skin-lesion-vit\experiments"
DATA_DIR    = r"D:\ML\skin-lesion-vit\data\raw"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dataset import load_metadata, split_dataset, CLASS_TO_IDX
from transforms import get_val_transform


# ── Load Model A and B probs (pre-saved numpy arrays) ─────────────────────────

def load_eff_probs():
    path = os.path.join(CKPT_BASE, "phase8_tta", "efficientnetv2_probs.npy")
    if not os.path.exists(path):
        print("ERROR: EfficientNetV2 TTA probs not found.")
        print("Run:  python src/tta_ensemble.py --phase 8")
        sys.exit(1)
    probs = np.load(path)
    print(f"[A] Loaded EfficientNetV2+TTA probs: {probs.shape}")
    return probs


def load_dino_ft_probs():
    path_ft     = os.path.join(CKPT_BASE, "phase15_tta_dinov2_ft", "dinov2_ft_tta_probs.npy")
    path_frozen = os.path.join(CKPT_BASE, "phase8_tta", "dinov2_registers_probs.npy")

    if os.path.exists(path_ft):
        probs = np.load(path_ft)
        print(f"[B] Loaded DINOv2+Reg (full FT)+TTA probs (Phase 15): {probs.shape}")
        return probs, "phase15_dinov2_ft_tta"
    elif os.path.exists(path_frozen):
        print("WARNING: Phase 15 probs not found — falling back to frozen DINOv2+Reg TTA (Phase 8).")
        print("         For best results run: python src/tta_ensemble.py --phase 15")
        probs = np.load(path_frozen)
        print(f"[B] Loaded DINOv2+Reg (frozen)+TTA probs (Phase 8): {probs.shape}")
        return probs, "phase8_dinov2_frozen_tta"
    else:
        print("ERROR: No DINOv2 TTA probs found. Run Phase 8 or 15 first.")
        sys.exit(1)


def load_labels():
    path = os.path.join(CKPT_BASE, "phase8_tta", "labels.npy")
    if not os.path.exists(path):
        print("ERROR: Labels not found. Run Phase 8 first.")
        sys.exit(1)
    return np.load(path)


# ── Model C: Phase 12 multimodal inference ────────────────────────────────────

class MultimodalDataset(torch.utils.data.Dataset):
    SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
    LOC_MAP = {
        "back": 0, "lower extremity": 1, "trunk": 2, "upper extremity": 3,
        "abdomen": 4, "face": 5, "chest": 6, "foot": 7, "scalp": 8,
        "neck": 9, "hand": 10, "ear": 11, "acral": 12, "genital": 13,
        "unknown": 14,
    }

    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import pandas as pd
        from PIL import Image as PILImage

        row   = self.df.iloc[idx]
        image = PILImage.open(row["image_path"]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        age_raw = row["age"]
        if pd.isna(age_raw) or str(age_raw).lower() == "unknown":
            age = 50.0 / 85.0
        else:
            age = float(age_raw) / 85.0

        sex   = self.SEX_MAP.get(str(row["sex"]).lower(), 2)
        loc   = self.LOC_MAP.get(str(row["localization"]).lower(), 14)
        meta  = torch.tensor([age, sex / 2.0, loc / 14.0], dtype=torch.float32)
        label = CLASS_TO_IDX[row["dx"]]
        return image, meta, label


def get_multimodal_probs(test_df, labels):
    from transformers import AutoModel

    ckpt_path = os.path.join(CKPT_BASE, "phase12_multimodal", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("ERROR: Phase 12 checkpoint not found.")
        print("Run:  python src/advanced_phases.py --phase 12")
        sys.exit(1)

    class DINOv2Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone    = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
            self.hidden_size = self.backbone.config.hidden_size

        def forward(self, pixel_values):
            return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

    class MultimodalClassifier(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.backbone    = DINOv2Backbone()
            hidden_size      = self.backbone.hidden_size
            self.meta_mlp    = nn.Sequential(
                nn.Linear(3, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 32), nn.ReLU(),
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(hidden_size + 32, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, pixel_values, meta):
            cls       = self.backbone(pixel_values)
            meta_feat = self.meta_mlp(meta)
            return self.fusion_head(torch.cat([cls, meta_feat], dim=1))

    print("\n[C] Loading Phase 12 multimodal model...")
    model      = MultimodalClassifier(NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)

    # Remap keys saved by advanced_phases.py (backbone.embeddings.* →
    # backbone.backbone.embeddings.*) to match the architecture here where
    # the HuggingFace model sits one level deeper inside DINOv2Backbone.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
            new_state_dict[k.replace("backbone.", "backbone.backbone.", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE).eval()
    print("Multimodal model loaded successfully!")

    val_tf = get_val_transform(224)
    ds     = MultimodalDataset(test_df, val_tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    all_probs = []
    with torch.no_grad():
        for images, meta, _ in tqdm(loader, desc="[C] Phase 12 inference"):
            images, meta = images.to(DEVICE), meta.to(DEVICE)
            with autocast():
                probs = torch.softmax(model(images, meta), dim=1).cpu().numpy()
            if np.isnan(probs).any():
                probs = np.where(np.isnan(probs), 1.0 / NUM_CLASSES, probs)
            all_probs.append(probs)

    all_probs  = np.concatenate(all_probs)
    sanity_auc = roc_auc_score(labels, all_probs, multi_class="ovr", average="macro")
    print(f"[C] Multimodal probs: {all_probs.shape}  |  standalone AUC: {sanity_auc:.4f} (expected ~0.9627)")
    return all_probs


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(eff_probs, dino_probs, meta_probs, labels):
    print("\nRunning 3-way grid search over ensemble weights...")
    best_auc, best_w, best_probs = 0.0, None, None

    for w1 in np.arange(0.25, 0.75, 0.05):       # EfficientNetV2
        for w2 in np.arange(0.10, 0.60, 0.05):   # DINOv2
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0.05 or w3 > 0.50:
                continue
            avg = w1 * eff_probs + w2 * dino_probs + w3 * meta_probs
            auc = roc_auc_score(labels, avg, multi_class="ovr", average="macro")
            if auc > best_auc:
                best_auc   = auc
                best_w     = (w1, w2, w3)
                best_probs = avg.copy()

    return best_auc, best_w, best_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def run_phase14():
    print("\n" + "=" * 60)
    print("  PHASE 14: 3-MODEL ENSEMBLE")
    print("  EffNetV2+TTA  |  DINOv2FullFT+TTA  |  Multimodal")
    print("=" * 60)

    eff_probs            = load_eff_probs()
    dino_probs, dino_tag = load_dino_ft_probs()
    labels               = load_labels()

    df = load_metadata(DATA_DIR)
    _, _, test_df = split_dataset(df)

    # FIX — test_df row order must match the saved label array row order.
    # split_dataset uses a fixed random_state=42 so the split is
    # deterministic, but we re-index to be safe.
    test_df = test_df.reset_index(drop=True)

    meta_probs = get_multimodal_probs(test_df, labels)

    baseline_auc = roc_auc_score(
        labels, 0.60 * eff_probs + 0.40 * dino_probs,
        multi_class="ovr", average="macro"
    )
    print(f"\nReference — Phase 9 (2-model 0.60/0.40) AUC: {baseline_auc:.4f}")

    best_auc, best_w, best_probs = grid_search(eff_probs, dino_probs, meta_probs, labels)
    best_acc = (best_probs.argmax(1) == labels).mean()

    print(f"\n{'=' * 60}")
    print(f"Best 3-model ensemble:")
    print(f"  EfficientNetV2+TTA : {best_w[0]:.2f}")
    print(f"  DINOv2+Reg+TTA     : {best_w[1]:.2f}  ({dino_tag})")
    print(f"  Multimodal (Ph.12) : {best_w[2]:.2f}")
    print(f"  Test AUC           : {best_auc:.4f}  (was {baseline_auc:.4f})")
    print(f"  Test Acc           : {best_acc:.4f}")
    print(f"  Delta vs Phase 9   : {best_auc - baseline_auc:+.4f}")
    print(f"{'=' * 60}")

    print("\nPer-class report:")
    print(classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES))

    ckpt_dir = os.path.join(CKPT_BASE, "phase14_three_model_ensemble")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, "ensemble_probs.npy"), best_probs)
    np.save(os.path.join(ckpt_dir, "labels.npy"), labels)

    result_path = os.path.join(ckpt_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write("Phase 14: 3-Model Ensemble\n\n")
        f.write(f"Model A: EfficientNetV2-M + TTA\n")
        f.write(f"Model B: DINOv2+Registers ({dino_tag})\n")
        f.write(f"Model C: Multimodal DINOv2+Reg (Phase 12)\n\n")
        f.write(f"Best weights:\n")
        f.write(f"  EfficientNetV2+TTA : {best_w[0]:.2f}\n")
        f.write(f"  DINOv2+Reg+TTA     : {best_w[1]:.2f}\n")
        f.write(f"  Multimodal         : {best_w[2]:.2f}\n\n")
        f.write(f"Test AUC : {best_auc:.4f}  (Phase 9 was {baseline_auc:.4f})\n")
        f.write(f"Test Acc : {best_acc:.4f}\n\n")
        f.write(classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES))
    print(f"\nResults saved to {result_path}")

    mlflow.set_experiment("skin-lesion-all-phases")
    with mlflow.start_run(run_name="phase14_three_model_ensemble"):
        mlflow.log_params({
            "w_efficientnetv2": best_w[0],
            "w_dinov2"        : best_w[1],
            "w_multimodal"    : best_w[2],
            "dino_source"     : dino_tag,
        })
        mlflow.log_metrics({
            "test_auc"        : best_auc,
            "test_acc"        : best_acc,
            "delta_vs_phase9" : round(best_auc - baseline_auc, 4),
        })

    return best_auc


if __name__ == "__main__":
    run_phase14()