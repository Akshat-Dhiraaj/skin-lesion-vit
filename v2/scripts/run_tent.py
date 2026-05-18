"""
scripts/run_tent.py
-------------------
Run TENT test-time adaptation on a trained registry model against an
external dataset (DDI / Fitz17k). Mirrors src/cross_dataset.py's API but
replaces the no-grad inference step with src.tent.tent_evaluate.

Output:
    experiments/cross_dataset_tent/<dataset>/<model>/
        probs_universal.npy   (or probs_ham10k.npy)
        labels.npy
        fitz_buckets.npy      (when applicable)
        report.txt

Usage:
    python scripts/run_tent.py --model pandem_real_probe --dataset fitz17k \
        --data_root data/external --label_mode universal --lr 1e-3 --n_steps 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.cross_dataset import (
    DEVICE,
    load_external_dataset,
    map_ham10k_probs_to_universal,
)
from src.data.transforms import get_val_transform
from src.models.registry import REGISTRY, build_model
from src.tent import wrap_for_tent, tent_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=["ddi", "fitz17k", "isic2019"])
    parser.add_argument("--data_root", default="data/external")
    parser.add_argument("--label_mode", default="universal", choices=["universal", "ham10k"])
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=1)
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix appended to per-(dataset, model) output dir name. "
                             "Use to keep multiple TENT hyperparam runs side by side.")
    args = parser.parse_args()

    cfg = REGISTRY[args.model]
    ckpt_path = Path(args.ckpt_base) / cfg.name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    # Build + load + wrap for TENT
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = wrap_for_tent(model, copy_first=False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TENT trainable params: {n_trainable:,} (out of {sum(p.numel() for p in model.parameters()):,})")

    # External dataset
    df, DatasetClass, fitz_aware = load_external_dataset(args.dataset, args.data_root, args.label_mode)
    transform = get_val_transform(cfg.input_size, cfg.model_type)
    ds = DatasetClass(df, transform=transform, label_mode=args.label_mode)
    bs = args.batch_size if args.batch_size is not None else cfg.batch_size
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Run TENT
    print(f"Running TENT (lr={args.lr}, n_steps={args.n_steps}, batch_size={bs}) on {len(ds)} samples...")
    probs_t, labels_t = tent_evaluate(model, loader, lr=args.lr, n_steps=args.n_steps, device=DEVICE)
    probs = probs_t.numpy(); labels = labels_t.numpy()

    # Universal projection (HAM-trained model outputs 7-way; project to 3 if needed)
    if args.label_mode == "universal" and probs.shape[1] == 7:
        probs = map_ham10k_probs_to_universal(probs)

    # Fitzpatrick buckets from df (mirror cross_dataset.run filtering)
    if args.label_mode == "ham10k" and "ham10k_idx" in df.columns:
        df_eval = df[df["ham10k_idx"] >= 0].reset_index(drop=True)
    else:
        df_eval = df.reset_index(drop=True)
    buckets = df_eval["fitz_bucket"].values if "fitz_bucket" in df_eval.columns else None

    # Save
    out_dir = Path(args.ckpt_base) / "cross_dataset_tent" / args.dataset / (cfg.name + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"probs_{args.label_mode}.npy", probs)
    np.save(out_dir / "labels.npy", labels)
    if buckets is not None:
        np.save(out_dir / "fitz_buckets.npy", np.asarray(buckets, dtype=object))

    # Report
    lines = [
        f"TENT eval -- {cfg.name} on {args.dataset} ({args.label_mode})",
        f"  TENT lr={args.lr}, n_steps={args.n_steps}, batch_size={bs}",
        f"  n_samples = {len(labels)}",
        "",
    ]
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        lines.append(f"  Macro OvR AUC : {auc:.4f}")
    except Exception as e:
        # Per-class fallback (handles missing-class case)
        ncl = probs.shape[1]
        aucs = []
        for c in range(ncl):
            y = (labels == c).astype(int)
            if 0 < y.sum() < len(y):
                aucs.append(roc_auc_score(y, probs[:, c]))
        if aucs:
            lines.append(f"  Macro OvR AUC : {float(np.mean(aucs)):.4f}  (per-class fallback)")
        else:
            lines.append(f"  Macro OvR AUC : NaN ({e})")

    if args.label_mode == "universal" and probs.shape[1] >= 1:
        mal = (labels == 0).astype(int)
        if 0 < mal.sum() < len(mal):
            lines.append(f"  Malignant AUC : {roc_auc_score(mal, probs[:, 0]):.4f}")
        if buckets is not None:
            lines.append("")
            lines.append("  Fitzpatrick-stratified Malignant AUC:")
            for grp in ["light", "medium", "dark"]:
                mask = np.asarray([str(b) == grp for b in buckets])
                m = mal[mask]
                if mask.sum() == 0 or m.sum() == 0 or m.sum() == mask.sum():
                    lines.append(f"    {grp:8s} N={int(mask.sum()):>4}  pos={int(m.sum()):>4}  AUC=NaN")
                else:
                    a = roc_auc_score(m, probs[mask, 0])
                    lines.append(f"    {grp:8s} N={int(mask.sum()):>4}  pos={int(m.sum()):>4}  AUC={a:.4f}")

    report = "\n".join(lines)
    print()
    print(report)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {out_dir}/report.txt")


if __name__ == "__main__":
    main()
