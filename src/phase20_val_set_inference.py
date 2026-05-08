"""
phase20_val_set_inference.py
-----------------------------
Run TTA inference on the VALIDATION SET (1502 samples) for all 6 trained models.
Save both raw logits AND softmax probabilities for each model + each split
(val + test).

WHY VALIDATION SET INFERENCE
-----------------------------
Phase 16/17 tuned ensemble weights on a 50/50 split of the test set, leaving
only 752 samples for honest evaluation and only ~83 mel positives for tuning.

The cleaner approach: since all models were trained without ever seeing the
val set or the test set, we can:
  - Use the val set (1502 samples, ~167 mel) to fit meta-learners (SLSQP,
    XGBoost, logistic regression)
  - Use the FULL test set (1503 samples, ~167 mel) for honest evaluation
  - This doubles the sample size for both fitting AND evaluation

OUTPUTS (saved to experiments/phase20_val_inference/)
------------------------------------------------------
  For each model in {eff, dino_ft, p13, p12_mm, convnext, mel_binary}:
    {model}_val_probs.npy       # (1502, 7) softmax probabilities
    {model}_val_logits.npy      # (1502, 7) pre-softmax logits
    {model}_test_probs.npy      # (1503, 7)
    {model}_test_logits.npy     # (1503, 7)

  Also saved:
    val_labels.npy              # (1502,) integer labels
    test_labels.npy             # (1503,)
    val_image_ids.npy           # (1502,) string IDs for traceability
    test_image_ids.npy          # (1503,)

  Special case for mel_binary (Phase 19): outputs are (N, 2). We save
    mel_binary_val_probs.npy as (N, 2) and the meta-learner in Phase 21
    extracts column 1 (P(mel)) as a single feature.

KEY DESIGN DECISIONS
--------------------
1. RAM cache: All 1502 val + 1503 test images loaded into RAM ONCE
   (~2.4 GB total). Transforms are swapped per TTA view → no disk I/O after
   first load.

2. Per-model TTA budget: more views for stronger models, fewer for the
   binary specialist (which already has high-quality decisions).
     EfficientNetV2 @ 384  : 20 views
     DINOv2 FT @ 224       : 20 views
     Phase 13 @ 224        : 20 views
     Multimodal @ 224      :  4 views (spatial-only — metadata is invariant)
     ConvNeXt-V2 @ 384     : 20 views
     Mel binary @ 224      : 12 views (all geometric — colour augmentation
                                       hurts a specialist trained on mel/nv)

3. Logits saved alongside probs: Phase 21 will compare softmax(mean(logits))
   vs mean(softmax(logits)) — logit averaging often outperforms.

4. num_workers=0 for the RAM dataset on Windows (workers would pickle-copy
   the full image cache, quadrupling RAM usage).

5. All probs normalized to sum to 1 before saving (clip + row-normalize)
   so downstream sklearn calls don't fail on float precision drift.

VRAM MANAGEMENT
---------------
Only ONE model on GPU at a time. After each model:
  del model; gc.collect(); torch.cuda.empty_cache()

ConvNeXt-V2 Large is the largest (197M params, ~1.2GB at fp16). Even with
batch=16 it fits comfortably on 8GB. Checkpoint loaded with weights_only=True
when supported.

RUNTIME ESTIMATE
----------------
Per model (val + test = 3005 samples):
  20 views × 3005 / batch_size = ~190 batches × 20 = ~3800 forward passes
  EfficientNetV2 @ 384:  ~30 min
  DINOv2 FT @ 224:       ~25 min
  Phase 13 @ 224:        ~25 min
  Multimodal @ 224:       ~5 min (only 4 views)
  ConvNeXt @ 384:        ~35 min
  Mel binary @ 224:      ~15 min (12 views)
Total: ~2.25 hours.

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase20_val_set_inference.py
"""

import os
import sys
import gc
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from transformers import AutoModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_BASE = BASE_DIR / "experiments"
OUT_DIR   = CKPT_BASE / "phase20_val_inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX     = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(42)
np.random.seed(42)

sys.path.append(str(BASE_DIR / "src"))
from dataset import load_metadata, split_dataset, CLASS_TO_IDX


def _autocast():
    return torch.amp.autocast(device_type=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TTA TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def _base_compose(image_size: int, extra: list) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        *extra,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_full_tta_transforms(image_size: int) -> list:
    """20 deterministic TTA views (geometric + colour + distortion + blur)."""
    s = image_size
    return [
        # Symmetric flips (4)
        _base_compose(s, []),
        _base_compose(s, [A.HorizontalFlip(p=1.0)]),
        _base_compose(s, [A.VerticalFlip(p=1.0)]),
        _base_compose(s, [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        # Rotations (4)
        _base_compose(s, [A.Rotate(limit=(45, 45), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(90, 90), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(180, 180), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(270, 270), p=1.0)]),
        # Multi-scale (3)
        _base_compose(s, [A.CenterCrop(int(s * 0.90), int(s * 0.90), p=1.0),
                          A.Resize(s, s)]),
        _base_compose(s, [A.CenterCrop(int(s * 0.80), int(s * 0.80), p=1.0),
                          A.Resize(s, s)]),
        _base_compose(s, [A.Affine(scale=1.10, translate_percent=0,
                                   rotate=0, p=1.0)]),
        # Colour jitter (4)
        _base_compose(s, [A.ColorJitter(brightness=(1.15, 1.15), contrast=0,
                                        saturation=0, hue=0, p=1.0)]),
        _base_compose(s, [A.ColorJitter(brightness=(0.85, 0.85), contrast=0,
                                        saturation=0, hue=0, p=1.0)]),
        _base_compose(s, [A.ColorJitter(brightness=0, contrast=(1.15, 1.15),
                                        saturation=0, hue=0, p=1.0)]),
        _base_compose(s, [A.ColorJitter(brightness=0, contrast=(0.85, 0.85),
                                        saturation=0, hue=0, p=1.0)]),
        # Geometric distortions (3)
        _base_compose(s, [A.ElasticTransform(alpha=30, sigma=5, p=1.0)]),
        _base_compose(s, [A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0)]),
        _base_compose(s, [A.OpticalDistortion(distort_limit=0.2, p=1.0)]),
        # Blur (1)
        _base_compose(s, [A.GaussianBlur(blur_limit=(3, 3), p=1.0)]),
        # Combined (1)
        _base_compose(s, [A.HorizontalFlip(p=1.0),
                          A.CenterCrop(int(s * 0.90), int(s * 0.90), p=1.0),
                          A.Resize(s, s)]),
    ]


def get_geometric_tta_transforms(image_size: int) -> list:
    """
    12 geometric-only TTA views — for the mel binary specialist.
    No colour augmentation (a specialist trained on mel/nv could be hurt by
    brightness/contrast shifts, since mel detection is partly colour-sensitive).
    """
    s = image_size
    return [
        _base_compose(s, []),
        _base_compose(s, [A.HorizontalFlip(p=1.0)]),
        _base_compose(s, [A.VerticalFlip(p=1.0)]),
        _base_compose(s, [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(45, 45), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(90, 90), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(180, 180), p=1.0)]),
        _base_compose(s, [A.Rotate(limit=(270, 270), p=1.0)]),
        _base_compose(s, [A.CenterCrop(int(s * 0.90), int(s * 0.90), p=1.0),
                          A.Resize(s, s)]),
        _base_compose(s, [A.CenterCrop(int(s * 0.80), int(s * 0.80), p=1.0),
                          A.Resize(s, s)]),
        _base_compose(s, [A.Affine(scale=1.10, translate_percent=0,
                                   rotate=0, p=1.0)]),
        _base_compose(s, [A.HorizontalFlip(p=1.0),
                          A.CenterCrop(int(s * 0.90), int(s * 0.90), p=1.0),
                          A.Resize(s, s)]),
    ]


def get_multimodal_tta_transforms(image_size: int = 224) -> list:
    """4 symmetric spatial views — colour jitter unhelpful for metadata branch."""
    s = image_size
    return [
        _base_compose(s, []),
        _base_compose(s, [A.HorizontalFlip(p=1.0)]),
        _base_compose(s, [A.VerticalFlip(p=1.0)]),
        _base_compose(s, [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 2. RAM DATASET
# ══════════════════════════════════════════════════════════════════════════════

class RAMDataset(torch.utils.data.Dataset):
    """All images cached in RAM. Transform swapped externally per TTA view."""

    SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
    LOC_MAP = {
        "back": 0, "lower extremity": 1, "trunk": 2, "upper extremity": 3,
        "abdomen": 4, "face": 5, "chest": 6, "foot": 7, "scalp": 8,
        "neck": 9, "hand": 10, "ear": 11, "acral": 12, "genital": 13,
        "unknown": 14,
    }

    def __init__(self, df: pd.DataFrame, name: str = ""):
        self.df        = df.reset_index(drop=True)
        self.transform = None     # set externally per TTA view
        self.images    = []
        self.metas     = []
        self.labels    = []
        self.image_ids = []

        print(f"  Caching {name} ({len(df)} images) to RAM...")
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
            self.image_ids.append(row["image_id"])

        self.labels = np.array(self.labels, dtype=np.int64)
        ram_mb = sum(img.nbytes for img in self.images) / 1e6
        print(f"  {name} cached: {len(self.images)} images, ~{ram_mb:.0f} MB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.metas[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """Phase 10 / Phase 13: DINOv2+Registers backbone + 7-class linear head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base"
        )
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


class DINOv2Backbone(nn.Module):
    """Wrapper used by the multimodal model."""

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base"
        )
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, pixel_values):
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]


class MultimodalClassifier(nn.Module):
    """Phase 12: image + metadata fusion."""

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


class DINOv2BinaryClassifier(nn.Module):
    """Phase 19: DINOv2+Registers backbone + 2-class head."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base"
        )
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


# ── Loaders ────────────────────────────────────────────────────────────────────

def _safe_load(ckpt_path: Path) -> dict:
    try:
        return torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(str(ckpt_path), map_location=DEVICE)


def load_efficientnetv2(ckpt: Path) -> nn.Module:
    model = timm.create_model(
        "tf_efficientnetv2_m", pretrained=False, num_classes=NUM_CLASSES,
    )
    model.load_state_dict(_safe_load(ckpt))
    return model.to(DEVICE).eval()


def load_dinov2_ft(ckpt: Path) -> nn.Module:
    model = DINOv2Classifier(NUM_CLASSES)
    model.load_state_dict(_safe_load(ckpt))
    return model.to(DEVICE).eval()


def load_phase13(ckpt: Path) -> nn.Module:
    model = DINOv2Classifier(NUM_CLASSES)
    model.load_state_dict(_safe_load(ckpt))
    return model.to(DEVICE).eval()


def load_multimodal(ckpt: Path) -> nn.Module:
    model = MultimodalClassifier(NUM_CLASSES)
    sd    = _safe_load(ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
            new_sd["backbone.backbone." + k[len("backbone."):]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)
    return model.to(DEVICE).eval()


def load_convnextv2(ckpt: Path) -> nn.Module:
    model = timm.create_model(
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(_safe_load(ckpt))
    return model.to(DEVICE).eval()


def load_mel_binary(ckpt: Path) -> nn.Module:
    model = DINOv2BinaryClassifier(num_classes=2)
    model.load_state_dict(_safe_load(ckpt))
    return model.to(DEVICE).eval()


# ══════════════════════════════════════════════════════════════════════════════
# 4. TTA INFERENCE  (saves both probs and logits)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    p = probs.astype(np.float64)
    p = np.clip(p, 1e-10, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def run_tta_inference(
    model_name:     str,
    model:          nn.Module,
    base_ds:        RAMDataset,
    tta_transforms: list,
    batch_size:     int = 32,
    is_multimodal:  bool = False,
) -> tuple:
    """
    Run all TTA views; return (avg_probs, avg_logits) both averaged across views.

    avg_logits   = mean of pre-softmax logits across views
    avg_probs    = normalize_probs(mean of softmax probs across views)

    Phase 21 will choose between two ensemble strategies:
      - probability averaging (use avg_probs directly)
      - logit averaging (use softmax(avg_logits))
    """
    n_views = len(tta_transforms)
    # Pre-allocate accumulators sized to the first batch's output dim
    sum_probs:  np.ndarray | None = None
    sum_logits: np.ndarray | None = None

    for view_i, tf in enumerate(tta_transforms):
        base_ds.transform = tf
        loader = torch.utils.data.DataLoader(
            base_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        view_probs:  list[np.ndarray] = []
        view_logits: list[np.ndarray] = []

        with torch.no_grad():
            for imgs, metas, _ in tqdm(
                loader,
                desc=f"  [{model_name}] V{view_i + 1}/{n_views}",
                leave=False,
            ):
                imgs  = imgs.to(DEVICE, non_blocking=True)
                metas = metas.to(DEVICE, non_blocking=True)
                with _autocast():
                    logits = model(imgs, metas) if is_multimodal else model(imgs)
                logits_np = logits.float().cpu().numpy()
                probs_np  = torch.softmax(logits, dim=1).float().cpu().numpy()
                view_logits.append(logits_np)
                view_probs.append(probs_np)

        v_logits = np.concatenate(view_logits)   # (N, n_classes)
        v_probs  = np.concatenate(view_probs)

        if sum_probs is None:
            sum_probs  = v_probs.astype(np.float64)
            sum_logits = v_logits.astype(np.float64)
        else:
            sum_probs  += v_probs
            sum_logits += v_logits

    avg_probs  = normalize_probs(sum_probs  / n_views)
    avg_logits = sum_logits / n_views          # NOT normalized — logits don't sum to 1

    return avg_probs, avg_logits


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def model_registry():
    """
    Build the list of models to run, skipping ones whose checkpoints are missing.
    Each entry: (key, loader_fn, ckpt_path, tta_transforms_func, image_size,
                 is_multimodal, n_outputs, batch_size)
    """
    registry = []

    def _add(key, loader, ckpt, tfs_fn, img_size, is_mm, n_out, bs, desc):
        if ckpt.exists():
            registry.append({
                "key":      key,
                "loader":   loader,
                "ckpt":     ckpt,
                "tfs_fn":   tfs_fn,
                "img_size": img_size,
                "is_mm":    is_mm,
                "n_out":    n_out,
                "bs":       bs,
                "desc":     desc,
            })
            print(f"  [+] {key:12s}  {desc}")
        else:
            print(f"  [SKIP] {key:12s}  ckpt not found at {ckpt}")

    print("\nDiscovering checkpoints...")
    _add("eff",
         load_efficientnetv2,
         CKPT_BASE / "phase2_efficientnetv2" / "best_model.pth",
         get_full_tta_transforms, 384, False, NUM_CLASSES, 16,
         "EfficientNetV2-M  @ 384px  20-view TTA")

    _add("dino_ft",
         load_dinov2_ft,
         CKPT_BASE / "phase10_dinov2_registers_finetune" / "best_model.pth",
         get_full_tta_transforms, 224, False, NUM_CLASSES, 32,
         "DINOv2+Reg full FT  @ 224px  20-view TTA")

    _add("p13",
         load_phase13,
         CKPT_BASE / "phase13_focal_loss" / "best_model.pth",
         get_full_tta_transforms, 224, False, NUM_CLASSES, 32,
         "Phase 13 focal loss  @ 224px  20-view TTA")

    _add("p12_mm",
         load_multimodal,
         CKPT_BASE / "phase12_multimodal" / "best_model.pth",
         get_multimodal_tta_transforms, 224, True, NUM_CLASSES, 32,
         "Multimodal (Phase 12)  @ 224px  4-view TTA")

    _add("convnext",
         load_convnextv2,
         CKPT_BASE / "phase18_convnextv2_large" / "best_model.pth",
         get_full_tta_transforms, 384, False, NUM_CLASSES, 8,
         "ConvNeXt-V2 Large (Phase 18)  @ 384px  20-view TTA")

    _add("mel_binary",
         load_mel_binary,
         CKPT_BASE / "phase19_mel_binary" / "best_model.pth",
         get_geometric_tta_transforms, 224, False, 2, 32,
         "Mel-vs-rest binary (Phase 19)  @ 224px  12-view TTA")

    return registry


def main():
    print("=" * 70)
    print("  PHASE 20: VAL + TEST INFERENCE  (probs + logits, all 6 models)")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

    # ── Load + split data ─────────────────────────────────────────────────────
    df = load_metadata(str(DATA_DIR))
    _train_df, val_df, test_df = split_dataset(df)
    val_df  = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ── Cache both splits to RAM ──────────────────────────────────────────────
    print("\nCaching datasets to RAM (one-time cost)...")
    val_ds  = RAMDataset(val_df,  name="VAL")
    test_ds = RAMDataset(test_df, name="TEST")

    # Save labels and image IDs
    np.save(str(OUT_DIR / "val_labels.npy"),     val_ds.labels)
    np.save(str(OUT_DIR / "test_labels.npy"),    test_ds.labels)
    np.save(str(OUT_DIR / "val_image_ids.npy"),  np.array(val_ds.image_ids))
    np.save(str(OUT_DIR / "test_image_ids.npy"), np.array(test_ds.image_ids))
    print(f"\nSaved labels and image_ids to {OUT_DIR}")
    print(f"  val:  {val_ds.labels.shape}   mel positives: {(val_ds.labels == MEL_IDX).sum()}")
    print(f"  test: {test_ds.labels.shape}  mel positives: {(test_ds.labels == MEL_IDX).sum()}")

    # ── Discover models ───────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("MODEL REGISTRY")
    print("─" * 70)
    registry = model_registry()
    if not registry:
        raise RuntimeError("No checkpoints found. Run Phases 2/10/12/13/18/19 first.")
    print(f"\n{len(registry)} models will run.")

    # ── Run inference for each model on val + test ────────────────────────────
    for i, m in enumerate(registry):
        key = m["key"]
        print(f"\n{'═' * 70}")
        print(f"  [{i + 1}/{len(registry)}] {key}: {m['desc']}")
        print(f"{'═' * 70}")

        model = m["loader"](m["ckpt"])

        for split_name, base_ds in (("val", val_ds), ("test", test_ds)):
            print(f"\n  ── {split_name.upper()} split ({len(base_ds)} samples) ──")
            tfs = m["tfs_fn"](m["img_size"])

            avg_probs, avg_logits = run_tta_inference(
                model_name    = f"{key}-{split_name}",
                model         = model,
                base_ds       = base_ds,
                tta_transforms= tfs,
                batch_size    = m["bs"],
                is_multimodal = m["is_mm"],
            )

            assert avg_probs.shape == (len(base_ds), m["n_out"]), \
                f"Bad probs shape: {avg_probs.shape}"
            assert avg_logits.shape == (len(base_ds), m["n_out"]), \
                f"Bad logits shape: {avg_logits.shape}"

            np.save(str(OUT_DIR / f"{key}_{split_name}_probs.npy"),  avg_probs)
            np.save(str(OUT_DIR / f"{key}_{split_name}_logits.npy"), avg_logits)

            # Sanity check: standalone macro AUC on this split
            labels = base_ds.labels
            if m["n_out"] == NUM_CLASSES:
                auc = roc_auc_score(labels, avg_probs, multi_class="ovr", average="macro")
                mel_auc = roc_auc_score(
                    (labels == MEL_IDX).astype(int), avg_probs[:, MEL_IDX],
                )
                print(f"  [{key}-{split_name}]  Macro AUC: {auc:.4f}  |  Mel AUC: {mel_auc:.4f}")
            else:
                # Binary mel classifier: AUC of P(mel) column 1
                bin_labels = (labels == MEL_IDX).astype(int)
                mel_auc = roc_auc_score(bin_labels, avg_probs[:, 1])
                print(f"  [{key}-{split_name}]  Binary Mel-AUC: {mel_auc:.4f}")

        # Free GPU before loading next model
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest = {
        "models":      [m["key"] for m in registry],
        "n_val":       int(len(val_ds)),
        "n_test":      int(len(test_ds)),
        "n_classes":   NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }
    import json
    with open(str(OUT_DIR / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'═' * 70}")
    print(f"[SUCCESS] All inference complete. Artifacts in: {OUT_DIR}")
    print(f"{'═' * 70}")
    print("Files saved:")
    for m in registry:
        for split in ("val", "test"):
            print(f"  {m['key']}_{split}_probs.npy   shape=(N, {m['n_out']})")
            print(f"  {m['key']}_{split}_logits.npy  shape=(N, {m['n_out']})")
    print(f"  val_labels.npy, test_labels.npy")
    print(f"  val_image_ids.npy, test_image_ids.npy")
    print(f"  manifest.json")


if __name__ == "__main__":
    main()
