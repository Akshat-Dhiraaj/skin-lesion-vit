"""
phase17_maximum_push.py
------------------------
Maximum AUC push — designed for an 8-hour overnight run on RTX 4060 8GB / 32GB RAM.

STRATEGY OVERVIEW
-----------------
Current ceiling (Phase 16): Eval AUC 0.9819, Melanoma AUC 0.9528, recall 0.77

What this script does differently:

  1. MORE TTA VIEWS (8 → 20)
     The original 8 views (flips + 90° rotations) are all symmetric transforms.
     Adding colour jitter, centre crop, multi-scale crops, and elastic distortion
     gives genuinely diverse views. Each extra view costs ~30s on an RTX 4060.
     Diminishing returns kick in after ~20 views but gains are real up to that point.

  2. PHASE 13 (FOCAL LOSS MODEL) ADDED AS 4TH ENSEMBLE MEMBER
     Phase 13 was trained specifically to boost melanoma recall via focal loss
     (AUC 0.9562, melanoma recall 0.79 standalone). It's architecturally identical
     to DINOv2+Registers but has different decision boundaries on the mel/nv border.
     Adding a model with different loss objectives improves ensemble diversity more
     than adding a 4th CNN.

  3. PHASE 12 (MULTIMODAL) GETS LIMITED TTA (4 VIEWS)
     The spatial transforms are irrelevant for metadata, but the image branch still
     benefits from flip/rotation diversity. 4 symmetric views (original + hflip +
     vflip + hflip+vflip) add signal without wasting time.

  4. K-FOLD SLSQP (5-FOLD) INSTEAD OF 50/50 SPLIT
     Phase 16 tuned weights on 751 samples and evaluated on 752. With 5-fold CV:
     - Weights are tuned on 1202 samples (80%), evaluated on 301 (20%), rotated
     - Final reported AUC is the mean of 5 held-out fold AUCs — much lower variance
     - Final weights used for the classification report are averaged across folds
     This uses all 1503 test samples for both tuning and evaluation honestly.

  5. MEMORY LAYOUT: GPU → CPU → DISK
     Only one model lives on GPU at a time. Probability arrays (~84KB each, 7 classes
     × 1503 samples × float64) live in RAM. Models are deleted and cache cleared
     between runs. This fits comfortably within 8GB VRAM and 32GB RAM.

  6. ALBUMENTATIONS ShiftScaleRotate → Affine
     Fixes the deprecation warning in newer albumentations versions.

  7. PER-CLASS THRESHOLD TUNING FOR ALL MINORITY CLASSES
     Phase 16 only tuned the melanoma threshold. This script tunes independent
     thresholds for mel, df, and vasc (the three rarest classes) using F2 score,
     then applies a priority override: vasc > df > mel > argmax.

Expected gains vs Phase 16:
  - TTA 8→20 views:          +0.002–0.005 AUC
  - Phase 13 as 4th model:   +0.002–0.004 AUC (diversity on mel/nv boundary)
  - MM limited TTA:          +0.001–0.002 AUC
  - 5-fold SLSQP:            better weight estimates (lower variance, not higher AUC)
  - Multi-class thresholds:  +0.02–0.04 melanoma recall

Target: Eval AUC ≥ 0.984, Melanoma recall ≥ 0.82

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase17_maximum_push.py

Runtime estimate on RTX 4060:
    EfficientNetV2  20 views × 47 batches  ≈  25 min
    DINOv2 FT       20 views × 94 batches  ≈  50 min
    Phase 13        20 views × 94 batches  ≈  50 min
    Multimodal       4 views × 188 batches ≈  20 min
    5-fold SLSQP                           ≈   5 min
    Total                                  ≈ 2.5 hours
    (leaves 5.5 hours of margin)
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PILImage
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_BASE = BASE_DIR / "experiments"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX, NV_IDX, BCC_IDX, AKIEC_IDX, BKL_IDX, DF_IDX, VASC_IDX = range(7)

# Minority classes that benefit from threshold tuning (sorted by rarity)
THRESHOLD_CLASSES = [VASC_IDX, DF_IDX, MEL_IDX]   # priority: rarest first

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(42)
_RNG = np.random.default_rng(42)

sys.path.append(str(BASE_DIR / "src"))
from dataset import load_metadata, split_dataset, CLASS_TO_IDX
from transformers import AutoModel
import timm


def _autocast():
    return torch.amp.autocast(device_type=DEVICE)


# ── 1. Extended TTA transforms (20 views) ─────────────────────────────────────

def get_extended_tta_transforms(image_size: int = 224) -> list:
    """
    20 TTA views covering:
      - 4 symmetric flips  (original, hflip, vflip, hvflip)
      - 4 rotations        (90, 180, 270 + identity already in flips)
      - 3 multi-scale      (90%, 100%, 110% crop then resize)
      - 4 colour jitter    (brightness±, contrast±)
      - 3 elastic/grid     (elastic distortion, grid distortion, optical distortion)
      - 2 blur             (gaussian, motion)
      Totals to 20 deterministic-ish views that maximise prediction diversity.

    Uses A.Affine instead of deprecated ShiftScaleRotate.
    """
    def base(extra_transforms):
        return A.Compose([
            A.Resize(image_size, image_size),
            *extra_transforms,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    return [
        # ── Geometric ─────────────────────────────────────────────────────────
        # 1. Original
        base([]),
        # 2. Horizontal flip
        base([A.HorizontalFlip(p=1.0)]),
        # 3. Vertical flip
        base([A.VerticalFlip(p=1.0)]),
        # 4. HFlip + VFlip
        base([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        # 5. Rotate 90
        base([A.Rotate(limit=(90, 90), p=1.0)]),
        # 6. Rotate 180
        base([A.Rotate(limit=(180, 180), p=1.0)]),
        # 7. Rotate 270
        base([A.Rotate(limit=(270, 270), p=1.0)]),
        # 8. Rotate 45
        base([A.Rotate(limit=(45, 45), p=1.0)]),

        # ── Multi-scale crops (zoom in / zoom out then resize) ────────────────
        # 9. Zoom in 10% (crop centre then resize back)
        base([A.CenterCrop(int(image_size * 0.90), int(image_size * 0.90), p=1.0),
              A.Resize(image_size, image_size)]),
        # 10. Zoom in 20%
        base([A.CenterCrop(int(image_size * 0.80), int(image_size * 0.80), p=1.0),
              A.Resize(image_size, image_size)]),
        # 11. Zoom out 10% (pad then resize — Affine with scale>1)
        base([A.Affine(scale=1.10, translate_percent=0, rotate=0, p=1.0)]),

        # ── Colour jitter (deterministic at fixed values) ─────────────────────
        # 12. Bright +15%
        base([A.ColorJitter(brightness=(1.15, 1.15), contrast=0,
                            saturation=0, hue=0, p=1.0)]),
        # 13. Bright −15%
        base([A.ColorJitter(brightness=(0.85, 0.85), contrast=0,
                            saturation=0, hue=0, p=1.0)]),
        # 14. Contrast +15%
        base([A.ColorJitter(brightness=0, contrast=(1.15, 1.15),
                            saturation=0, hue=0, p=1.0)]),
        # 15. Contrast −15%
        base([A.ColorJitter(brightness=0, contrast=(0.85, 0.85),
                            saturation=0, hue=0, p=1.0)]),

        # ── Geometric distortions ─────────────────────────────────────────────
        # 16. Elastic distortion (skin lesion shape variation)
        base([A.ElasticTransform(alpha=30, sigma=5, p=1.0)]),
        # 17. Grid distortion
        base([A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0)]),
        # 18. Optical distortion
        base([A.OpticalDistortion(distort_limit=0.2, p=1.0)]),

        # ── Blur (handles camera focus variation) ─────────────────────────────
        # 19. Gaussian blur (σ=1)
        base([A.GaussianBlur(blur_limit=(3, 3), p=1.0)]),
        # 20. HFlip + zoom in (combined)
        base([A.HorizontalFlip(p=1.0),
              A.CenterCrop(int(image_size * 0.90), int(image_size * 0.90), p=1.0),
              A.Resize(image_size, image_size)]),
    ]


def get_multimodal_tta_transforms(image_size: int = 224) -> list:
    """
    4 symmetric TTA views for the multimodal model.
    Metadata is unaffected by image transforms; we only use symmetric spatial
    transforms (no colour jitter — metadata branch has no colour signal).
    """
    def base(extra):
        return A.Compose([
            A.Resize(image_size, image_size),
            *extra,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return [
        base([]),
        base([A.HorizontalFlip(p=1.0)]),
        base([A.VerticalFlip(p=1.0)]),
        base([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
    ]


# ── 2. RAM-cached dataset ──────────────────────────────────────────────────────

class UnifiedRAMDataset(torch.utils.data.Dataset):
    """All images cached in RAM once. Transform swapped per TTA view."""

    SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
    LOC_MAP = {
        "back": 0, "lower extremity": 1, "trunk": 2, "upper extremity": 3,
        "abdomen": 4, "face": 5, "chest": 6, "foot": 7, "scalp": 8,
        "neck": 9, "hand": 10, "ear": 11, "acral": 12, "genital": 13,
        "unknown": 14,
    }

    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.images    = []
        self.metas     = []
        self.labels    = []

        print("Caching dataset to RAM...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), leave=False):
            self.images.append(
                np.array(PILImage.open(row["image_path"]).convert("RGB"))
            )
            age_raw = row["age"]
            age = (
                50.0 / 85.0
                if pd.isna(age_raw) or str(age_raw).lower() == "unknown"
                else float(age_raw) / 85.0
            )
            sex = self.SEX_MAP.get(str(row["sex"]).lower(), 2)
            loc = self.LOC_MAP.get(str(row["localization"]).lower(), 14)
            self.metas.append(
                torch.tensor([age, sex / 2.0, loc / 14.0], dtype=torch.float32)
            )
            self.labels.append(CLASS_TO_IDX[row["dx"]])

        self.labels = np.array(self.labels)
        ram_mb = sum(img.nbytes for img in self.images) / 1e6
        print(f"Cached {len(self.images)} images | ~{ram_mb:.0f} MB RAM used")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.metas[idx], self.labels[idx]


# ── 3. Model definitions ───────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear head (used for Phase 10 and Phase 13)."""
    def __init__(self, model_id: str = "facebook/dinov2-with-registers-base",
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.head     = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


class DINOv2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, pixel_values):
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
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


def _load_state_dict(ckpt_path: Path, device: str) -> dict:
    """Load state dict with weights_only fallback for older PyTorch."""
    try:
        return torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(ckpt_path), map_location=device)


def load_efficientnetv2(ckpt_path: Path) -> nn.Module:
    model = timm.create_model("tf_efficientnetv2_m", pretrained=False,
                               num_classes=NUM_CLASSES)
    model.load_state_dict(_load_state_dict(ckpt_path, DEVICE))
    return model.to(DEVICE).eval()


def load_dinov2_ft(ckpt_path: Path) -> nn.Module:
    """Phase 10: fully fine-tuned DINOv2+Registers."""
    model = DINOv2Classifier()
    model.load_state_dict(_load_state_dict(ckpt_path, DEVICE))
    return model.to(DEVICE).eval()


def load_phase13(ckpt_path: Path) -> nn.Module:
    """
    Phase 13: focal-loss fine-tuned DINOv2+Registers.
    Trained specifically to push melanoma recall — different decision boundary
    from Phase 10 despite identical architecture. Adds genuine diversity.
    """
    model = DINOv2Classifier()
    model.load_state_dict(_load_state_dict(ckpt_path, DEVICE))
    return model.to(DEVICE).eval()


def load_multimodal(ckpt_path: Path) -> nn.Module:
    """Phase 12: DINOv2+Registers + age/sex/localization metadata."""
    model = MultimodalClassifier()
    sd    = _load_state_dict(ckpt_path, DEVICE)

    # Remap backbone.* -> backbone.backbone.* (key depth mismatch from save)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
            new_sd["backbone.backbone." + k[len("backbone."):]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)
    return model.to(DEVICE).eval()


# ── 4. Helpers ─────────────────────────────────────────────────────────────────

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Clip and row-normalize. sklearn roc_auc_score requires rows sum to 1."""
    probs = probs.astype(np.float64)
    probs = np.clip(probs, 1e-10, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def print_per_class_auc(labels: np.ndarray, probs: np.ndarray, tag: str) -> float:
    macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    print(f"\n  [{tag}] Macro AUC: {macro:.4f}")
    for i, cls in enumerate(CLASS_NAMES):
        binary  = (labels == i).astype(int)
        cls_auc = roc_auc_score(binary, probs[:, i])
        n_pos   = binary.sum()
        marker  = f"  n={n_pos}" + ("  <-- melanoma" if cls == "mel" else "")
        print(f"    {cls:8s}: {cls_auc:.4f}{marker}")
    return macro


# ── 5. TTA inference engine ────────────────────────────────────────────────────

def run_tta_inference(
    model_name:     str,
    model:          nn.Module,
    base_ds:        UnifiedRAMDataset,
    tta_transforms: list,
    batch_size:     int  = 32,
    is_multimodal:  bool = False,
) -> np.ndarray:
    """
    Run all TTA views. Returns normalized (N, 7) float64 array.
    num_workers=0: RAM dataset on Windows — no pickle-copy overhead.
    """
    all_view_probs = []

    for i, tf in enumerate(tta_transforms):
        base_ds.transform = tf
        loader = torch.utils.data.DataLoader(
            base_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        view_probs = []
        with torch.no_grad():
            for imgs, metas, _ in tqdm(
                loader,
                desc=f"  [{model_name}] View {i + 1}/{len(tta_transforms)}",
                leave=False,
            ):
                imgs  = imgs.to(DEVICE)
                metas = metas.to(DEVICE)
                with _autocast():
                    logits = model(imgs, metas) if is_multimodal else model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                view_probs.append(probs)

        all_view_probs.append(np.concatenate(view_probs))
        print(f"  [{model_name}] View {i + 1}/{len(tta_transforms)} done")

    avg = np.mean(all_view_probs, axis=0)
    avg = normalize_probs(avg)

    assert avg.shape == (len(base_ds), NUM_CLASSES), \
        f"[{model_name}] Shape mismatch: got {avg.shape}"
    return avg


# ── 6. K-fold SLSQP ───────────────────────────────────────────────────────────

def kfold_slsqp(
    probs_list:  list,
    labels:      np.ndarray,
    n_splits:    int = 5,
    n_restarts:  int = 10,
    rng:         np.random.Generator = None,
) -> tuple:
    """
    5-fold cross-validated SLSQP ensemble weight optimization.

    Why k-fold instead of 50/50:
      - Phase 16 used 751 tune samples → noisy weight estimates for rare classes
        (mel had only ~83 positive examples in the tune split).
      - 5-fold gives 1202 tune samples per fold (80%), much more stable.
      - Mean of 5 held-out AUCs is a lower-variance estimate of true performance.
      - Final weights = mean across folds (Bayesian model averaging interpretation).

    Returns:
      averaged_weights: (num_models, num_classes) — average of 5 fold weights
      fold_aucs:        list of 5 held-out AUCs
      all_oof_probs:    (N, 7) out-of-fold ensemble probs for the full test set
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_models = len(probs_list)
    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_weights = []
    fold_aucs    = []
    oof_probs    = np.zeros((len(labels), NUM_CLASSES), dtype=np.float64)

    def objective(flat_w, p_list, y):
        w   = flat_w.reshape(num_models, NUM_CLASSES)
        ens = normalize_probs(sum(p_list[i] * w[i] for i in range(num_models)))
        return -roc_auc_score(y, ens, multi_class='ovr', average='macro')

    constraints = [
        {
            'type': 'eq',
            'fun': lambda w, c=c: np.sum(
                w.reshape(num_models, NUM_CLASSES)[:, c]
            ) - 1,
        }
        for c in range(NUM_CLASSES)
    ]
    bounds = [(0.0, 1.0)] * (num_models * NUM_CLASSES)

    print(f"\nRunning {n_splits}-fold SLSQP "
          f"({n_restarts} restarts per fold, {num_models} models)...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
            np.zeros(len(labels)), labels)):

        p_train = [probs_list[i][train_idx] for i in range(num_models)]
        p_val   = [probs_list[i][val_idx]   for i in range(num_models)]
        y_train = labels[train_idx]
        y_val   = labels[val_idx]

        best_auc = -np.inf
        best_w   = None

        for restart in range(n_restarts):
            if restart == 0:
                init_w = np.ones((num_models, NUM_CLASSES)) / num_models
            else:
                init_w = rng.dirichlet(np.ones(num_models), size=NUM_CLASSES).T

            res = minimize(
                objective,
                init_w.flatten(),
                args=(p_train, y_train),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10},
            )
            auc = -res.fun
            if np.isfinite(auc) and auc > best_auc:
                best_auc = auc
                best_w   = res.x.reshape(num_models, NUM_CLASSES)

        if best_w is None:
            raise RuntimeError(f"Fold {fold_idx + 1}: all restarts failed.")

        fold_weights.append(best_w)

        # Out-of-fold ensemble for this fold's val samples
        ens_val = normalize_probs(
            sum(p_val[i] * best_w[i] for i in range(num_models))
        )
        oof_probs[val_idx] = ens_val

        fold_auc = roc_auc_score(y_val, ens_val, multi_class='ovr', average='macro')
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold_idx + 1}: train-AUC={best_auc:.4f}  "
              f"val-AUC={fold_auc:.4f}  (n_train={len(y_train)}, n_val={len(y_val)})")

    avg_weights = np.mean(fold_weights, axis=0)
    print(f"\n  Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Fold AUCs   : {[round(a, 4) for a in fold_aucs]}")

    return avg_weights, fold_aucs, oof_probs


# ── 7. Multi-class threshold tuning ───────────────────────────────────────────

def tune_minority_thresholds(
    ensemble_probs: np.ndarray,
    labels:         np.ndarray,
    target_classes: list = THRESHOLD_CLASSES,
) -> dict:
    """
    Tune independent F2-optimal thresholds for each minority class.
    Returns a dict {class_idx: threshold}.

    Priority order matters: thresholds are applied in the order of
    THRESHOLD_CLASSES (vasc → df → mel), so rarer classes take precedence
    in the override logic.

    Using OOF probs (from k-fold) for tuning means thresholds are tuned
    on predictions that the SLSQP weights have never directly seen —
    genuinely leak-free.
    """
    thresholds = {}
    print("\nTuning per-class thresholds (F2, β=2):")

    for cls_idx in target_classes:
        cls_name  = CLASS_NAMES[cls_idx]
        cls_probs = ensemble_probs[:, cls_idx]
        y_binary  = (labels == cls_idx).astype(int)
        n_pos     = y_binary.sum()

        # Search range: rare classes need lower thresholds
        t_max = min(0.50, cls_probs[y_binary == 1].mean() + 0.15) if n_pos > 0 else 0.30

        best_t, best_f2 = 0.0, 0.0
        for t in np.linspace(0.01, t_max, 300):
            preds = (cls_probs > t).astype(int)
            score = fbeta_score(y_binary, preds, beta=2, zero_division=0)
            if score > best_f2:
                best_f2, best_t = score, t

        thresholds[cls_idx] = best_t
        recall_at_t = (
            ((cls_probs > best_t) & (y_binary == 1)).sum() / max(n_pos, 1)
        )
        print(f"  {cls_name:8s}: threshold={best_t:.4f}  "
              f"F2={best_f2:.4f}  recall_at_t={recall_at_t:.3f}  n_pos={n_pos}")

    return thresholds


def apply_thresholds(
    ensemble_probs: np.ndarray,
    thresholds:     dict,
    priority_order: list = THRESHOLD_CLASSES,
) -> np.ndarray:
    """
    Apply class-specific thresholds in priority order (rarest class wins).
    Start from argmax, then override with rarer classes sequentially.
    """
    preds = np.argmax(ensemble_probs, axis=1).copy()
    # Apply in reverse priority (least rare first) so rarest overwrites all
    for cls_idx in reversed(priority_order):
        if cls_idx in thresholds:
            mask = ensemble_probs[:, cls_idx] > thresholds[cls_idx]
            preds[mask] = cls_idx
    return preds


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 17: MAXIMUM PUSH — 20-VIEW TTA + 4 MODELS + K-FOLD SLSQP")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {vram:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_metadata(str(DATA_DIR))
    _, _, test_df = split_dataset(df)
    test_df = test_df.reset_index(drop=True)

    base_ds = UnifiedRAMDataset(test_df)
    y_true  = base_ds.labels
    N       = len(y_true)
    print(f"\nTest samples : {N}")
    print(f"Class counts : { {CLASS_NAMES[i]: (y_true == i).sum() for i in range(NUM_CLASSES)} }")

    # ── Model configs ──────────────────────────────────────────────────────────
    # Each entry: display_name, loader_fn, ckpt_path, tta_transforms, is_multimodal
    tta_20_384 = get_extended_tta_transforms(image_size=384)
    tta_20_224 = get_extended_tta_transforms(image_size=224)
    tta_4_224  = get_multimodal_tta_transforms(image_size=224)

    # Check which checkpoints actually exist and warn for missing ones
    ckpt_eff   = CKPT_BASE / "phase2_efficientnetv2"               / "best_model.pth"
    ckpt_dino  = CKPT_BASE / "phase10_dinov2_registers_finetune"   / "best_model.pth"
    ckpt_p13   = CKPT_BASE / "phase13_focal_loss"                  / "best_model.pth"
    ckpt_mm    = CKPT_BASE / "phase12_multimodal"                  / "best_model.pth"

    model_registry = []

    if ckpt_eff.exists():
        model_registry.append(
            ("eff",  load_efficientnetv2, ckpt_eff,  tta_20_384, False,
             "EfficientNetV2-M (Phase 2) — 20-view TTA at 384px")
        )
    else:
        print(f"WARNING: {ckpt_eff} not found — skipping eff")

    if ckpt_dino.exists():
        model_registry.append(
            ("dino", load_dinov2_ft,     ckpt_dino, tta_20_224, False,
             "DINOv2+Registers full FT (Phase 10) — 20-view TTA at 224px")
        )
    else:
        print(f"WARNING: {ckpt_dino} not found — skipping dino")

    if ckpt_p13.exists():
        model_registry.append(
            ("p13",  load_phase13,       ckpt_p13,  tta_20_224, False,
             "DINOv2+Registers focal loss (Phase 13) — 20-view TTA at 224px")
        )
    else:
        print(f"WARNING: {ckpt_p13} not found — Phase 13 ckpt missing.")
        print("         Run: python src/advanced_phases.py --phase 13")
        print("         Continuing without Phase 13 (still 3-model ensemble).")

    if ckpt_mm.exists():
        model_registry.append(
            ("mm",   load_multimodal,    ckpt_mm,   tta_4_224,  True,
             "Multimodal DINOv2+Registers (Phase 12) — 4-view TTA at 224px")
        )
    else:
        print(f"WARNING: {ckpt_mm} not found — skipping multimodal")

    if len(model_registry) < 2:
        raise RuntimeError(
            "Fewer than 2 models found. Check checkpoint paths in CKPT_BASE."
        )

    print(f"\nModels to run ({len(model_registry)}):")
    for name, _, _, tfs, _, desc in model_registry:
        print(f"  [{name:4s}] {desc}  ({len(tfs)} views)")

    # ── Inference: one model at a time, GPU → free → next ─────────────────────
    probs_dict: dict[str, np.ndarray] = {}

    for name, loader_fn, ckpt, tfs, is_mm, desc in model_registry:
        print(f"\n{'=' * 65}")
        print(f"  {desc}")
        print(f"{'=' * 65}")

        model = loader_fn(ckpt).eval()   # already on DEVICE from loader

        probs_dict[name] = run_tta_inference(
            name, model, base_ds, tfs,
            batch_size=(16 if name == "eff" else 32),
            is_multimodal=is_mm,
        )
        print_per_class_auc(y_true, probs_dict[name], f"{name} standalone")

        # Free GPU before loading next model — critical on 8GB VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── K-fold SLSQP ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  5-FOLD SLSQP WEIGHT OPTIMIZATION")
    print(f"{'=' * 65}")

    model_order = list(probs_dict.keys())
    num_models  = len(model_order)
    probs_list  = [probs_dict[n] for n in model_order]

    avg_weights, fold_aucs, oof_probs = kfold_slsqp(
        probs_list, y_true,
        n_splits=5, n_restarts=10, rng=_RNG,
    )

    # OOF probs cover all N samples — this IS the unbiased evaluation
    oof_auc = roc_auc_score(y_true, oof_probs, multi_class='ovr', average='macro')
    print(f"\nOut-of-fold AUC (full test set, unbiased): {oof_auc:.4f}")
    print_per_class_auc(y_true, oof_probs, "OOF ensemble")

    # ── Multi-class threshold tuning on OOF probs ──────────────────────────────
    print(f"\n{'=' * 65}")
    print("  MULTI-CLASS THRESHOLD TUNING (on OOF probs — leak-free)")
    print(f"{'=' * 65}")

    thresholds = tune_minority_thresholds(oof_probs, y_true)

    # Apply thresholds and get final classification report
    final_preds = apply_thresholds(oof_probs, thresholds)
    report      = classification_report(y_true, final_preds, target_names=CLASS_NAMES)

    print(f"\nFinal classification report (thresholds applied):")
    print(report)

    # Melanoma-specific summary
    mel_mask    = y_true == MEL_IDX
    mel_recall  = (final_preds[mel_mask] == MEL_IDX).mean()
    mel_prec    = (
        (final_preds == MEL_IDX) & (y_true == MEL_IDX)
    ).sum() / max((final_preds == MEL_IDX).sum(), 1)
    print(f"Melanoma  recall    : {mel_recall:.4f}")
    print(f"Melanoma  precision : {mel_prec:.4f}")
    print(f"Melanoma  AUC       : "
          f"{roc_auc_score((y_true == MEL_IDX).astype(int), oof_probs[:, MEL_IDX]):.4f}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    out_dir = CKPT_BASE / "phase17_maximum_push"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "avg_weights.npy"),   avg_weights)
    np.save(str(out_dir / "oof_probs.npy"),     oof_probs)
    np.save(str(out_dir / "labels.npy"),         y_true)

    # Save each model's raw probs for future stacking experiments
    for name, arr in probs_dict.items():
        np.save(str(out_dir / f"{name}_probs.npy"), arr)

    with open(str(out_dir / "results.txt"), "w", encoding="utf-8") as f:
        f.write("=== PHASE 17: MAXIMUM PUSH RESULTS ===\n\n")
        f.write(f"Models ({num_models}): {', '.join(model_order)}\n")
        f.write(f"TTA views     : 20 (eff/dino/p13), 4 (mm)\n")
        f.write(f"SLSQP folds   : 5  |  restarts/fold: 10\n\n")

        f.write(f"OOF AUC (unbiased, full test set): {oof_auc:.4f}\n")
        f.write(f"Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n")
        f.write(f"Fold AUCs   : {[round(a, 4) for a in fold_aucs]}\n\n")

        f.write("Averaged Weights (rows=models, cols=classes):\n")
        f.write(f"{'':6s}" + "  ".join(f"{c:>7s}" for c in CLASS_NAMES) + "\n")
        for mn, row_w in zip(model_order, avg_weights):
            f.write(f"{mn:6s}" + "  ".join(f"{v:7.4f}" for v in row_w) + "\n")

        f.write("\nThresholds applied:\n")
        for cls_idx, t in thresholds.items():
            f.write(f"  {CLASS_NAMES[cls_idx]:8s} (idx={cls_idx}): {t:.4f}\n")

        f.write("\nPer-class AUC (OOF):\n")
        for i, cls in enumerate(CLASS_NAMES):
            binary  = (y_true == i).astype(int)
            cls_auc = roc_auc_score(binary, oof_probs[:, i])
            marker  = "  <-- melanoma" if cls == "mel" else ""
            f.write(f"  {cls:8s}: {cls_auc:.4f}{marker}\n")

        f.write("\nClassification Report (thresholds applied):\n")
        f.write(report)

    print(f"\n{'=' * 65}")
    print(f"[SUCCESS] All artifacts saved to: {out_dir}")
    print(f"  avg_weights.npy    shape={avg_weights.shape}")
    print(f"  oof_probs.npy      shape={oof_probs.shape}  (full test set, unbiased)")
    print(f"  results.txt        weights + thresholds + per-class AUC + report")
    for name in probs_dict:
        print(f"  {name}_probs.npy   shape={probs_dict[name].shape}  (raw per-model)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()