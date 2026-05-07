"""
phase16_unified_pipeline.py
----------------------------
Unified Inference & Optimization Engine.
Combines RAM-cached TTA, Multi-start SLSQP ensemble weighting,
and Target-Recall threshold tuning into one leak-free pipeline.

Fixes vs the version that crashed:
  FIX 1  — EfficientNetV2 TTA at 384px (its training size); DINOv2 at 224px.
  FIX 2  — SLSQP no longer returns None when restarts don't converge.
  FIX 3  — np.random.seed() removed from module level; explicit RNG passed in.
  FIX 4  — num_workers=0 on RAM-cached dataset (avoids Windows pickle-copy OOM).
  FIX 5  — val_tf renamed NO_TTA_TF and documented.
  FIX 6  — Hardcoded range(3) replaced with range(len(...)).
  FIX 7  — normalize_probs() applied consistently before AUC, threshold tuning,
            and final evaluation — fixing the "scores must sum to 1" crash.
  FIX 8  — Per-class AUC printed at every key stage.
  FIX 9  — torch.load weights_only guarded for older PyTorch.
  FIX 10 — Results file includes human-readable weight table + per-class AUC.

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase16_unified_pipeline.py
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
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_BASE = BASE_DIR / "experiments"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX     = 0

# FIX 3 — torch seed only at module level; numpy RNG passed explicitly so
# SLSQP Dirichlet restarts are reproducible regardless of prior RNG state.
torch.manual_seed(42)
_RNG = np.random.default_rng(42)

sys.path.append(str(BASE_DIR / "src"))
from dataset import load_metadata, split_dataset, CLASS_TO_IDX
from transforms import get_val_transform
from tta_ensemble import load_efficientnetv2, load_dinov2_registers, get_tta_transforms
from transformers import AutoModel

# FIX 1 — each model carries its own correct image resolution.
# EfficientNetV2-M trained at 384px (engine.py phase 2 "input_size": 384).
# Using 224 for both was silently degrading eff predictions.
MODEL_CONFIGS = {
    # key: (loader_fn, ckpt_subpath, img_size, use_tta)
    "eff":  (load_efficientnetv2,   "phase2_efficientnetv2/best_model.pth",             384, True),
    "dino": (load_dinov2_registers, "phase10_dinov2_registers_finetune/best_model.pth", 224, True),
}

# FIX 5 — renamed from val_tf. Multimodal gets no TTA because spatial
# augmentations (flip/rotate) add no signal to the metadata branch.
NO_TTA_TF = [get_val_transform(224)]

# FIX 7 — torch.amp.autocast replaces deprecated torch.cuda.amp.autocast.
# Defined after DEVICE so the device string is resolved at call time.
def _autocast():
    return torch.amp.autocast(device_type=DEVICE)

# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Convert to float64, clip, and row-normalize so every row sums to exactly 1.
    sklearn roc_auc_score with multi_class='ovr' enforces this strictly.
    """
    probs = probs.astype(np.float64) # <-- NEW: Force double precision
    probs = np.clip(probs, 1e-10, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def print_per_class_auc(labels: np.ndarray, probs: np.ndarray, tag: str) -> float:
    """
    FIX 8 — Print macro + per-class AUC. probs must already be normalized.
    Returns macro AUC.
    """
    macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    print(f"\n  [{tag}] Macro AUC: {macro:.4f}")
    for i, cls in enumerate(CLASS_NAMES):
        binary  = (labels == i).astype(int)
        cls_auc = roc_auc_score(binary, probs[:, i])
        marker  = "  <-- melanoma" if cls == "mel" else ""
        print(f"    {cls:8s}: {cls_auc:.4f}{marker}")
    return macro


# ── 1. RAM-Cached Dataset ─────────────────────────────────────────────────────

class UnifiedRAMDataset(torch.utils.data.Dataset):
    """
    Loads all raw images into RAM once.
    self.transform is swapped externally per TTA view so disk I/O happens
    exactly once across all 8 views × 3 models = 24 forward passes.
    """
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

        print("Caching dataset to RAM (happens exactly once)...")
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
        print(f"Cached {len(self.images)} images | ~{ram_mb:.0f} MB RAM")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.metas[idx], self.labels[idx]


# ── 2. Model Definitions ──────────────────────────────────────────────────────

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


def load_multimodal(ckpt_path: Path) -> MultimodalClassifier:
    """
    Load Phase 12 multimodal checkpoint with state_dict key remapping.
    advanced_phases.py stored HF backbone one level shallower than we define it.
    """
    model = MultimodalClassifier()

    # FIX 9 — weights_only added in PyTorch 1.13; graceful fallback for older.
    try:
        sd = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    except TypeError:
        sd = torch.load(str(ckpt_path), map_location=DEVICE)

    # backbone.embeddings.* -> backbone.backbone.embeddings.*
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
            new_sd["backbone.backbone." + k[len("backbone."):]] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd)
    return model.to(DEVICE).eval()


# ── 3. Inference Engine ───────────────────────────────────────────────────────

def run_tta_inference(
    model_name:    str,
    model:         nn.Module,
    base_ds:       UnifiedRAMDataset,
    tta_transforms: list,
    batch_size:    int  = 32,
    is_multimodal: bool = False,
) -> np.ndarray:
    """
    Run all TTA views for one model. Returns NORMALIZED (N, 7) float32 array.

    FIX 4 — num_workers=0: on Windows, multiprocessing workers pickle-copy
             the entire self.images list per worker, multiplying RAM by
             num_workers. Since all images are already in RAM the bottleneck
             is GPU compute not I/O, so workers give zero benefit.

    FIX 7 (partial) — probabilities are normalized inside this function so
             every caller receives a sklearn-safe array.
    """
    all_view_probs = []

    for i, tf in enumerate(tta_transforms):
        base_ds.transform = tf
        loader = torch.utils.data.DataLoader(
            base_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,    # FIX 4
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

    # FIX 7 — normalize immediately so the stored array is always sklearn-safe.
    # Floating-point accumulation across 8 views can shift row sums off 1.0.
    avg = normalize_probs(avg)

    assert avg.shape == (len(base_ds), NUM_CLASSES), \
        f"[{model_name}] Unexpected output shape: {avg.shape}"
    return avg


# ── 4. Optimization ───────────────────────────────────────────────────────────

def optimize_slsqp_multistart(
    probs_list:   list,
    labels:       np.ndarray,
    num_restarts: int = 5,
    rng:          np.random.Generator = None,
) -> tuple:
    """
    Multi-start SLSQP — maximizes ensemble AUC on the provided (tune) split.

    FIX 2 — original returned None when no restart converged (best_res stayed
             None), causing TypeError on downstream arithmetic.
             Now always returns the best result seen, converged or not.
             A RuntimeError is raised only if every restart produces NaN AUC.

    FIX 3 — rng is passed in explicitly so restarts are reproducible
             regardless of how much of the global RNG was consumed before.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_models = len(probs_list)

    def objective(flat_w: np.ndarray) -> float:
        w   = flat_w.reshape(num_models, NUM_CLASSES)
        ens = normalize_probs(
            sum(probs_list[i] * w[i] for i in range(num_models))
        )
        return -roc_auc_score(labels, ens, multi_class='ovr', average='macro')

    constraints = [
        {
            'type': 'eq',
            'fun': lambda w, c=c: np.sum(
                w.reshape(num_models, NUM_CLASSES)[:, c]
            ) - 1,
        }
        for c in range(NUM_CLASSES)
    ]
    bounds  = [(0.0, 1.0)] * (num_models * NUM_CLASSES)

    best_auc = -np.inf
    best_w   = None

    print(f"\nRunning {num_restarts} SLSQP restarts "
          f"({num_models} models × {NUM_CLASSES} classes)...")

    for i in range(num_restarts):
        if i == 0:
            init_w = np.ones((num_models, NUM_CLASSES)) / num_models
        else:
            # Columns of Dirichlet draws sum to 1 → valid starting point
            init_w = rng.dirichlet(np.ones(num_models), size=NUM_CLASSES).T

        res = minimize(
            objective,
            init_w.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9},
        )
        auc    = -res.fun
        status = "converged" if res.success else f"NOT converged: {res.message}"
        print(f"  Restart {i + 1:2d}: AUC={auc:.4f}  [{status}]")

        # FIX 2 — track best regardless of convergence flag
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_w   = res.x.reshape(num_models, NUM_CLASSES)
            if not res.success:
                print(f"           ^ best so far (unconverged — "
                      f"weights may be slightly suboptimal)")

    if best_w is None:
        raise RuntimeError(
            "All SLSQP restarts produced non-finite AUC. "
            "Check that probs are valid and labels match."
        )

    return best_w, best_auc


def tune_melanoma_threshold(
    ensemble_probs: np.ndarray,
    labels:         np.ndarray,
) -> tuple:
    """
    Grid-search the melanoma F2-optimal decision threshold on normalized probs.
    Must only ever be called with tune-split data, never eval-split data.
    """
    mel_probs = ensemble_probs[:, MEL_IDX]
    y_mel     = (labels == MEL_IDX).astype(int)

    best_t, best_f2 = 0.0, 0.0
    for t in np.linspace(0.01, 0.30, 200):
        score = fbeta_score(
            y_mel, (mel_probs > t).astype(int),
            beta=2, zero_division=0,
        )
        if score > best_f2:
            best_f2, best_t = score, t

    return best_t, best_f2


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 16: UNIFIED INFERENCE & OPTIMIZATION ENGINE")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_metadata(str(DATA_DIR))
    _, _, test_df = split_dataset(df)
    test_df = test_df.reset_index(drop=True)

    base_ds = UnifiedRAMDataset(test_df)
    y_true  = base_ds.labels  # integer labels, shape (N,)

    # Stratified 50/50 split of the test set.
    # tune_idx → used for SLSQP weight search + threshold tuning.
    # eval_idx → used only for final honest reporting.
    indices = np.arange(len(y_true))
    tune_idx, eval_idx = train_test_split(
        indices, test_size=0.5, stratify=y_true, random_state=42
    )
    print(f"\nTune samples : {len(tune_idx)}")
    print(f"Eval samples : {len(eval_idx)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    probs_dict: dict[str, np.ndarray] = {}

    for name, (loader_fn, ckpt_sub, img_size, use_tta) in MODEL_CONFIGS.items():
        print(f"\n{'=' * 55}")
        print(f"  [{name}]  size={img_size}px  "
              f"tta={'8 views' if use_tta else 'none'}")
        print(f"{'=' * 55}")

        model = loader_fn(str(CKPT_BASE / ckpt_sub)).to(DEVICE).eval()

        # FIX 1 — correct resolution per model
        tfs = get_tta_transforms(image_size=img_size) if use_tta else NO_TTA_TF

        # run_tta_inference already returns normalize_probs(avg) internally
        probs_dict[name] = run_tta_inference(
            name, model, base_ds, tfs, is_multimodal=False
        )
        # FIX 8 — per-class AUC after every model (probs already normalized)
        print_per_class_auc(y_true, probs_dict[name], f"{name} full-test")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Multimodal — single canonical view (no TTA)
    print(f"\n{'=' * 55}")
    print("  [mm]  size=224px  tta=none")
    print("  (spatial augmentations add no signal to the metadata branch)")
    print(f"{'=' * 55}")

    ckpt_mm = CKPT_BASE / "phase12_multimodal" / "best_model.pth"
    mm_model = load_multimodal(ckpt_mm)
    probs_dict["mm"] = run_tta_inference(
        "mm", mm_model, base_ds, NO_TTA_TF, is_multimodal=True
    )
    print_per_class_auc(y_true, probs_dict["mm"], "mm full-test")
    del mm_model
    gc.collect()
    torch.cuda.empty_cache()

    model_order = list(probs_dict.keys())   # ['eff', 'dino', 'mm']
    num_models  = len(model_order)

    # ── SLSQP on TUNE split ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  OPTIMIZING ENSEMBLE WEIGHTS  (tune split only)")
    print(f"{'=' * 60}")

    p_tune = [probs_dict[n][tune_idx] for n in model_order]
    y_tune = y_true[tune_idx]

    best_weights, tune_auc = optimize_slsqp_multistart(
        p_tune, y_tune, num_restarts=5, rng=_RNG
    )
    print(f"\nBest SLSQP Tune AUC: {tune_auc:.4f}")

    # FIX 7 — normalize the weighted sum before AUC and before threshold tuner
    ens_tune = normalize_probs(
        sum(p_tune[i] * best_weights[i] for i in range(num_models))
    )
    print_per_class_auc(y_tune, ens_tune, "ensemble tune")

    best_t, best_f2 = tune_melanoma_threshold(ens_tune, y_tune)
    print(f"\nOptimal Melanoma Threshold : {best_t:.4f}  (tune F2={best_f2:.4f})")

    # ── Final evaluation on EVAL split ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL EVALUATION  (eval split — never touched during tuning)")
    print(f"{'=' * 60}")

    # FIX 6 — use range(num_models) not hardcoded range(3)
    p_eval = [probs_dict[n][eval_idx] for n in model_order]
    y_eval = y_true[eval_idx]

    # FIX 7 — same normalize_probs as tune path
    ens_eval = normalize_probs(
        sum(p_eval[i] * best_weights[i] for i in range(num_models))
    )

    eval_auc = roc_auc_score(y_eval, ens_eval, multi_class='ovr', average='macro')
    print_per_class_auc(y_eval, ens_eval, "ensemble eval")

    final_preds = np.argmax(ens_eval, axis=1)
    final_preds[ens_eval[:, MEL_IDX] > best_t] = MEL_IDX

    report = classification_report(y_eval, final_preds, target_names=CLASS_NAMES)
    print(f"\nClassification report (melanoma threshold={best_t:.4f} applied):")
    print(report)

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir = CKPT_BASE / "phase16_unified_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "optimized_weights.npy"), best_weights)
    np.save(str(out_dir / "ens_eval_probs.npy"),    ens_eval)
    np.save(str(out_dir / "eval_labels.npy"),        y_eval)

    # FIX 10 — human-readable weight table + per-class AUC in the text file
    with open(str(out_dir / "unified_results.txt"), "w", encoding="utf-8") as f:
        f.write("=== PHASE 16: UNIFIED RESULTS ===\n\n")
        f.write(f"Tune AUC            : {tune_auc:.4f}\n")
        f.write(f"Eval AUC (OOS)      : {eval_auc:.4f}\n")
        f.write(f"Melanoma Threshold  : {best_t:.4f}  (tune F2={best_f2:.4f})\n\n")

        f.write("Optimized Weights (rows=models, cols=classes):\n")
        header = f"{'':6s}" + "  ".join(f"{c:>7s}" for c in CLASS_NAMES)
        f.write(header + "\n")
        for mn, row in zip(model_order, best_weights):
            f.write(f"{mn:6s}" + "  ".join(f"{v:7.4f}" for v in row) + "\n")

        f.write("\nPer-class AUC (eval split):\n")
        for i, cls in enumerate(CLASS_NAMES):
            binary  = (y_eval == i).astype(int)
            cls_auc = roc_auc_score(binary, ens_eval[:, i])
            marker  = "  <-- melanoma" if cls == "mel" else ""
            f.write(f"  {cls:8s}: {cls_auc:.4f}{marker}\n")

        f.write("\nClassification Report (threshold applied):\n")
        f.write(report)

    print(f"\n[SUCCESS] Artifacts saved to: {out_dir}")
    print(f"  optimized_weights.npy  shape={best_weights.shape}  "
          f"(load with np.load, rows=models cols=classes)")
    print(f"  ens_eval_probs.npy     shape={ens_eval.shape}")
    print(f"  unified_results.txt    weights + per-class AUC + report")


if __name__ == "__main__":
    main()