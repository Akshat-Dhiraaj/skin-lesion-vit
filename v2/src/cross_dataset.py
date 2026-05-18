"""
src/cross_dataset.py
--------------------
Cross-dataset evaluation: take a HAM10000-trained model and evaluate it on
DDI / Fitzpatrick17k / ISIC 2019.

Reports:
  - Overall AUC (one-vs-rest macro) under the chosen label_mode
  - Per-class report
  - For datasets with Fitzpatrick labels: per-skin-tone-bucket stratified AUC
    (this is the Path B / Path C fairness story)

Saves prediction probabilities so you can later ensemble across models.

Usage:
    # DDI (universal 3-class)
    python -m src.cross_dataset --model convnextv2_large --dataset ddi \\
        --data_root data/external --label_mode universal

    # Fitzpatrick17k (universal 3-class) with stratification
    python -m src.cross_dataset --model convnextv2_large --dataset fitz17k \\
        --data_root data/external --label_mode universal

    # ISIC 2019 (HAM10000 7-class, drops HAM10000 overlap automatically)
    python -m src.cross_dataset --model convnextv2_large --dataset isic2019 \\
        --data_root data/external --label_mode ham10k
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import get_val_transform
from src.data.external.taxonomy import (
    HAM10K_CLASSES,
    UNIVERSAL_CLASSES,
    HAM10K_MALIGNANT,
    HAM10K_TO_IDX,
    UNIVERSAL_TO_IDX,
    HAM10K_TO_UNIVERSAL,
    FITZ_BUCKET_NAMES,
)
from src.models.registry import REGISTRY, build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"


# ── Dataset dispatch ──────────────────────────────────────────────────────────

def load_external_dataset(name: str, data_root: str, label_mode: str):
    """
    Returns (df, dataset_class, fitz_aware) where fitz_aware is True when
    the dataset has per-sample Fitzpatrick bucket labels for stratified eval.
    """
    data_root = Path(data_root)

    if name == "ddi":
        from src.data.external.ddi import load_ddi_metadata, DDIDataset
        df = load_ddi_metadata(data_root / "DDI")
        return df, DDIDataset, True

    if name in ("fitz17k", "fitzpatrick17k"):
        from src.data.external.fitzpatrick17k import (
            load_fitzpatrick17k_metadata, Fitzpatrick17kDataset
        )
        df = load_fitzpatrick17k_metadata(data_root / "Fitzpatrick17k")
        return df, Fitzpatrick17kDataset, True

    if name == "isic2019":
        from src.data.external.isic2019 import load_isic2019_metadata, ISIC2019Dataset
        df = load_isic2019_metadata(data_root / "ISIC2019")
        return df, ISIC2019Dataset, False

    raise ValueError(f"Unknown dataset: {name}. Choose: ddi, fitz17k, isic2019.")


# ── Prediction adapter (HAM10000 7-class -> universal 3-class) ────────────────

def map_ham10k_probs_to_universal(probs_7: np.ndarray) -> np.ndarray:
    """
    Convert a model's 7-way HAM10000 softmax output into 3-way universal probs.

    universal[malignant] = sum of mel + bcc + akiec
    universal[benign]    = sum of nv + bkl + df + vasc
    universal[non_neop.] = 0 (a HAM10000 model never predicts non-neoplastic)

    Output shape: (N, 3). Rows sum to 1.
    """
    n = probs_7.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)

    for ham_cls, ham_i in HAM10K_TO_IDX.items():
        u = HAM10K_TO_UNIVERSAL[ham_cls]
        u_i = UNIVERSAL_TO_IDX[u]
        out[:, u_i] += probs_7[:, ham_i]

    # The model's prob mass on benign+malignant covers ALL its prediction.
    # Non-neoplastic gets 0 because HAM10000 has no such class. This means
    # for non-neoplastic ground truth, the model can never be "right" — its
    # OvR AUC for that class is undefined (we skip it in macro AUC).
    return out


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs_7class_or_native, labels)."""
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Infer", leave=False):
        images = images.to(DEVICE)
        with torch.amp.autocast(_DEVICE_TYPE):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # AMP NaN guard
    if np.isnan(probs).any():
        probs = np.where(np.isnan(probs), 1.0 / probs.shape[1], probs)

    return probs, labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def macro_ovr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Manual macro OvR AUC, ignores classes with only one label present."""
    n_classes = probs.shape[1]
    aucs = []
    for c in range(n_classes):
        y_c = (labels == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            continue
        try:
            aucs.append(roc_auc_score(y_c, probs[:, c]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_auc(labels: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    """Per-class binary AUC. Returns dict {class_name: auc}. NaN if not computable."""
    out = {}
    for c, name in enumerate(class_names):
        y_c = (labels == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            out[name] = float("nan")
            continue
        try:
            out[name] = float(roc_auc_score(y_c, probs[:, c]))
        except Exception:
            out[name] = float("nan")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    model_key: str,
    dataset_name: str,
    data_root: str,
    label_mode: str,
    ckpt_base: str = "experiments",
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    cfg = REGISTRY[model_key]
    ckpt_dir = Path(ckpt_base) / cfg.name
    ckpt_path = ckpt_dir / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python scripts/train.py --model {model_key}"
        )

    print(f"\n{'=' * 70}")
    print(f"  CROSS-DATASET EVAL")
    print(f"  Model   : {cfg.name}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Mode    : {label_mode}")
    print(f"{'=' * 70}")

    # Load dataset + dataset class
    df, DatasetClass, fitz_aware = load_external_dataset(dataset_name, data_root, label_mode)
    print(f"Loaded {len(df)} samples from {dataset_name}")
    if "universal_label" in df.columns:
        print(f"  universal label distribution:\n{df['universal_label'].value_counts().to_string()}")

    # Build PyTorch dataset + loader
    transform = get_val_transform(cfg.input_size)
    ds = DatasetClass(df, transform=transform, label_mode=label_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Filter df to match the dataset (e.g. ham10k mode drops unmapped rows)
    if label_mode == "ham10k" and "ham10k_idx" in df.columns:
        df_eval = df[df["ham10k_idx"] >= 0].reset_index(drop=True)
    else:
        df_eval = df.reset_index(drop=True)
    print(f"Evaluating on {len(df_eval)} samples ({label_mode} mode)")

    # Build model + load weights
    print(f"\nLoading model from {ckpt_path}...")
    model = build_model(cfg).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Inference (the model outputs 7-class HAM10000 probs)
    print("\nRunning inference...")
    probs_7, labels = predict(model, loader)

    # Map 7-class -> chosen label space if needed
    if label_mode == "universal":
        probs_for_eval = map_ham10k_probs_to_universal(probs_7)
        eval_classes = UNIVERSAL_CLASSES
    else:
        probs_for_eval = probs_7
        eval_classes = HAM10K_CLASSES

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall_auc = macro_ovr_auc(labels, probs_for_eval)
    per_class = per_class_auc(labels, probs_for_eval, eval_classes)

    print(f"\n--- Overall ---")
    print(f"  Macro OvR AUC: {overall_auc:.4f}")
    print(f"  Per-class AUC:")
    for c, a in per_class.items():
        print(f"    {c:18s}  {a:.4f}" if not np.isnan(a) else f"    {c:18s}  N/A")

    # ── Binary malignant vs not (clinical metric) ─────────────────────────────
    if label_mode == "universal":
        # P(malignant) directly from universal probs
        p_mal = probs_for_eval[:, UNIVERSAL_TO_IDX["malignant"]]
        y_mal = (labels == UNIVERSAL_TO_IDX["malignant"]).astype(int)
    else:
        # P(malignant) = sum of HAM10000 malignant classes
        mal_idx_list = [HAM10K_TO_IDX[c] for c in HAM10K_MALIGNANT]
        p_mal = probs_for_eval[:, mal_idx_list].sum(axis=1)
        y_mal_lbls = {HAM10K_TO_IDX[c] for c in HAM10K_MALIGNANT}
        y_mal = np.isin(labels, list(y_mal_lbls)).astype(int)

    if y_mal.sum() > 0 and y_mal.sum() < len(y_mal):
        mal_auc = float(roc_auc_score(y_mal, p_mal))
        print(f"\n--- Binary malignant detection ---")
        print(f"  Malignant AUC: {mal_auc:.4f}")
    else:
        mal_auc = float("nan")

    # ── Fitzpatrick stratification ────────────────────────────────────────────
    fitz_results = {}
    if fitz_aware and "fitz_bucket" in df_eval.columns:
        print(f"\n--- Fitzpatrick-stratified malignant AUC ---")
        for bucket in FITZ_BUCKET_NAMES:
            mask = (df_eval["fitz_bucket"] == bucket).values
            if mask.sum() < 10:
                print(f"  {bucket:8s}  N={int(mask.sum()):4d}  (skipped, too few samples)")
                fitz_results[bucket] = {"n": int(mask.sum()), "auc": float("nan")}
                continue
            y_b = y_mal[mask]
            p_b = p_mal[mask]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                print(f"  {bucket:8s}  N={int(mask.sum()):4d}  (skipped, single-class)")
                fitz_results[bucket] = {"n": int(mask.sum()), "auc": float("nan")}
                continue
            auc_b = float(roc_auc_score(y_b, p_b))
            n_b = int(mask.sum())
            n_pos = int(y_b.sum())
            print(f"  {bucket:8s}  N={n_b:4d}  pos={n_pos:4d}  AUC={auc_b:.4f}")
            fitz_results[bucket] = {"n": n_b, "n_pos": n_pos, "auc": auc_b}

        # Worst-group gap — the headline fairness metric
        valid = [r["auc"] for r in fitz_results.values()
                 if isinstance(r.get("auc"), float) and not np.isnan(r["auc"])]
        if len(valid) >= 2:
            gap = max(valid) - min(valid)
            print(f"\n  Light-to-dark AUC gap: {gap:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(ckpt_base) / "cross_dataset" / dataset_name / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"probs_{label_mode}.npy", probs_for_eval)
    np.save(out_dir / f"probs_ham10k.npy", probs_7)
    np.save(out_dir / "labels.npy", labels)

    # Save fitz_bucket mask for downstream analysis
    if fitz_aware and "fitz_bucket" in df_eval.columns:
        np.save(out_dir / "fitz_buckets.npy",
                df_eval["fitz_bucket"].fillna("unknown").values.astype(object))

    # Text report
    report_path = out_dir / "report.txt"
    lines = [
        f"Cross-dataset eval",
        f"  model      : {cfg.name}",
        f"  dataset    : {dataset_name}",
        f"  label_mode : {label_mode}",
        f"  n_samples  : {len(df_eval)}",
        "",
        f"--- Overall metrics ---",
        f"  Macro OvR AUC : {overall_auc:.4f}",
        f"  Malignant AUC : {mal_auc:.4f}",
        "",
        f"--- Per-class AUC ---",
    ]
    for c, a in per_class.items():
        lines.append(f"  {c:18s}  {a:.4f}" if not np.isnan(a) else f"  {c:18s}  N/A")

    if fitz_results:
        lines.append("")
        lines.append(f"--- Fitzpatrick-stratified malignant AUC ---")
        for b in FITZ_BUCKET_NAMES:
            r = fitz_results.get(b, {})
            n = r.get("n", 0)
            n_pos = r.get("n_pos", 0)
            auc_b = r.get("auc", float("nan"))
            if isinstance(auc_b, float) and not np.isnan(auc_b):
                lines.append(f"  {b:8s}  N={n:4d}  pos={n_pos:4d}  AUC={auc_b:.4f}")
            else:
                lines.append(f"  {b:8s}  N={n:4d}  (N/A)")
        valid = [r["auc"] for r in fitz_results.values()
                 if isinstance(r.get("auc"), float) and not np.isnan(r["auc"])]
        if len(valid) >= 2:
            lines.append(f"")
            lines.append(f"  Light-to-dark AUC gap: {max(valid) - min(valid):.4f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return {
        "model": cfg.name,
        "dataset": dataset_name,
        "label_mode": label_mode,
        "overall_auc": overall_auc,
        "malignant_auc": mal_auc,
        "per_class": per_class,
        "fitz": fitz_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["ddi", "fitz17k", "isic2019"])
    parser.add_argument("--data_root", default="data/external")
    parser.add_argument("--label_mode", choices=["universal", "ham10k"], default="universal")
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run(
        model_key=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        label_mode=args.label_mode,
        ckpt_base=args.ckpt_base,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)