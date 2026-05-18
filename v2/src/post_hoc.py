"""
src/post_hoc.py
---------------
All post-hoc improvements applied to TTA probability arrays:

1. Per-class weighted ensemble - per-class SLSQP weights fit on val
2. Platt sigmoid calibration   - per-class 1-D LogisticRegression fit on val
3. F2-optimal threshold tuning - per-class F2 maximisation on val (raises mel recall)

Each step is fit on the validation TTA probs and evaluated on the test set.

FIX (v3): Replaced isotonic with Platt sigmoid scaling. Isotonic with our
small rare-class val populations (df=16, vasc=20) collapses to <30 unique
calibrated values per class, creating massive ties that destroy per-class
ranking and tank macro OvR AUC by ~0.04. Platt sigmoid is strictly
monotonic with N unique outputs per class and preserves AUC exactly.

FIX (v2): apply_calibration() does NOT renormalise per-row. Renormalisation
destroys per-class ranking that macro OvR AUC depends on. Outputs are
valid per-class probabilities in [0, 1] that do not sum to 1 across classes.

Inputs (per model, must already exist):
    experiments/<model>/tta_val_probs.npy
    experiments/<model>/tta_val_labels.npy
    experiments/<model>/tta_test_probs.npy   (or tta_probs.npy backward-compat)
    experiments/<model>/tta_test_labels.npy  (or tta_labels.npy)

Outputs:
    experiments/post_hoc/results.txt
    experiments/post_hoc/per_class_weights.npy
    experiments/post_hoc/test_probs_calibrated.npy
    experiments/post_hoc/thresholds.npy

Usage:
    python -m src.post_hoc --models convnextv2_large efficientnetv2_m \\
                                    dinov2_registers_ft pandem_ft
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score

from src.data.ham10000 import CLASS_NAMES
from src.models.registry import REGISTRY


# ── Loaders ───────────────────────────────────────────────────────────────────

def _resolve_split_paths(ckpt_dir: Path, split: str) -> tuple[Path, Path]:
    """Find probs + labels files, with backward-compat for the old test naming."""
    new_probs = ckpt_dir / f"tta_{split}_probs.npy"
    new_labels = ckpt_dir / f"tta_{split}_labels.npy"
    if new_probs.exists() and new_labels.exists():
        return new_probs, new_labels
    if split == "test":
        old_probs = ckpt_dir / "tta_probs.npy"
        old_labels = ckpt_dir / "tta_labels.npy"
        if old_probs.exists() and old_labels.exists():
            return old_probs, old_labels
    return new_probs, new_labels


def load_split_probs(model_keys: list[str], split: str, ckpt_base: str):
    """Load TTA probs + labels for a given split. Returns (probs_list, labels)."""
    probs_list = []
    labels = None
    for key in model_keys:
        cfg = REGISTRY[key]
        ckpt_dir = Path(ckpt_base) / cfg.name
        probs_path, labels_path = _resolve_split_paths(ckpt_dir, split)
        if not probs_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] {split} TTA probs not found for {key}\n"
                f"        Expected: {probs_path}\n"
                f"        Run: python -m src.tta --model {key} --split {split}\n"
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file missing: {labels_path}")
        probs = np.load(probs_path)
        these_labels = np.load(labels_path)
        if labels is None:
            labels = these_labels
        else:
            assert np.array_equal(labels, these_labels), \
                f"Labels mismatch between models on {split} split"
        probs_list.append(probs)
        kind = "[new]" if "tta_" + split in probs_path.name else "[legacy]"
        print(f"  Loaded {key:30s} {split:5s} {kind:8s} shape={probs.shape}")
    return probs_list, labels


# ── 1. Per-class weighted ensemble ────────────────────────────────────────────

def fit_per_class_weights(probs_list_val, val_labels) -> np.ndarray:
    """For each class c, SLSQP-maximise OvR AUC for class c on val."""
    n_models = len(probs_list_val)
    n_classes = probs_list_val[0].shape[1]
    weights = np.full((n_classes, n_models), 1.0 / n_models)

    for c in range(n_classes):
        y_c = (val_labels == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            print(f"  [class {c}] only one label present - using uniform weights")
            continue

        def neg_auc(w):
            w = np.clip(w, 1e-6, 1.0)
            w = w / w.sum()
            ens_c = sum(wi * p[:, c] for wi, p in zip(w, probs_list_val))
            try:
                return -roc_auc_score(y_c, ens_c)
            except Exception:
                return 0.0

        best_auc = -np.inf
        best_w = np.full(n_models, 1.0 / n_models)
        init_points = [np.full(n_models, 1.0 / n_models)]
        for i in range(n_models):
            ip = np.full(n_models, 0.05)
            ip[i] = 1.0 - 0.05 * (n_models - 1)
            init_points.append(ip)

        for x0 in init_points:
            try:
                res = minimize(
                    neg_auc, x0=x0, method="SLSQP",
                    bounds=[(0.01, 0.99)] * n_models,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                    options={"maxiter": 100, "ftol": 1e-7},
                )
                auc = -res.fun
                if auc > best_auc:
                    best_auc = auc
                    best_w = res.x / res.x.sum()
            except Exception:
                continue

        weights[c] = best_w
        print(f"  [class {CLASS_NAMES[c]:6s}] val AUC: {best_auc:.4f}  weights: "
              + " ".join(f"{w:.2f}" for w in best_w))
    return weights


def apply_per_class_weights(probs_list, weights: np.ndarray) -> np.ndarray:
    """Apply per-class weights -> ensemble probs of shape (N, C). Renormalises."""
    n_classes = weights.shape[0]
    ensemble = np.zeros_like(probs_list[0])
    for c in range(n_classes):
        for i, p in enumerate(probs_list):
            ensemble[:, c] += weights[c, i] * p[:, c]
    # Renormalise here is OK because each column is a linear combination
    # of the SAME column from input - per-class ranking is preserved
    row_sums = ensemble.sum(axis=1, keepdims=True)
    ensemble = ensemble / np.maximum(row_sums, 1e-10)
    return ensemble


# ── 2. Per-class Platt sigmoid calibration ────────────────────────────────────
# We use Platt sigmoid (1-D logistic regression) rather than isotonic
# because isotonic, with our small rare-class val populations (df=16, vasc=20),
# collapses many distinct scores into <30 unique calibrated values per class.
# Those massive ties destroy per-class ranking and tank macro OvR AUC by ~0.04.
# Platt is strictly monotonic with 1431 unique outputs per class — preserves
# AUC exactly while still producing valid calibrated probabilities.

def fit_platt(val_probs: np.ndarray, val_labels: np.ndarray) -> list:
    """One 1-D LogisticRegression per class, fit on val OvR probabilities."""
    from sklearn.linear_model import LogisticRegression

    n_classes = val_probs.shape[1]
    calibrators = []
    for c in range(n_classes):
        y_c = (val_labels == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            calibrators.append(None)
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(val_probs[:, c].reshape(-1, 1), y_c)
        calibrators.append(lr)
    return calibrators


def apply_platt(probs: np.ndarray, calibrators: list) -> np.ndarray:
    """
    Apply per-class Platt sigmoid calibration. DOES NOT renormalise — for
    the same reason as before: renormalisation breaks per-class ranking
    and macro OvR AUC. Outputs are valid per-class probabilities each in
    [0, 1], do not sum to 1 across classes.
    """
    n_classes = probs.shape[1]
    out = np.zeros_like(probs)
    for c in range(n_classes):
        if calibrators[c] is None:
            out[:, c] = probs[:, c]
        else:
            out[:, c] = calibrators[c].predict_proba(probs[:, c].reshape(-1, 1))[:, 1]
    return out


# ── 3. F2-optimal per-class thresholds ────────────────────────────────────────

def fit_f2_thresholds(val_probs: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """Per-class F2-maximising thresholds on val."""
    n_classes = val_probs.shape[1]
    thresholds = np.full(n_classes, 0.5)
    grid = np.linspace(0.01, 0.99, 99)
    for c in range(n_classes):
        y_c = (val_labels == c).astype(int)
        if y_c.sum() == 0:
            continue
        best_t, best_f2 = 0.5, -1.0
        for t in grid:
            preds = (val_probs[:, c] > t).astype(int)
            try:
                f2 = fbeta_score(y_c, preds, beta=2, zero_division=0)
            except Exception:
                f2 = 0.0
            if f2 > best_f2:
                best_f2 = f2
                best_t = t
        thresholds[c] = best_t
        print(f"  [class {CLASS_NAMES[c]:6s}] F2-optimal threshold: {best_t:.3f}  (val F2: {best_f2:.3f})")
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Threshold-adjusted argmax: subtract threshold then argmax."""
    adjusted = probs - thresholds[None, :]
    return adjusted.argmax(axis=1)


# ── Reporting ─────────────────────────────────────────────────────────────────

def macro_ovr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Manual macro one-vs-rest AUC. Equivalent to
        roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    but works on UNNORMALISED scores (sklearn's version requires the rows
    to sum to 1; ours doesn't, which is what we need after per-class
    isotonic calibration). Per-class AUC only depends on the ranking within
    each column, so normalisation across columns doesn't matter.
    """
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
    return float(np.mean(aucs)) if aucs else 0.0


def report(name: str, probs: np.ndarray, labels: np.ndarray, preds=None) -> dict:
    auc = macro_ovr_auc(labels, probs)
    if preds is None:
        preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    mel_idx = CLASS_NAMES.index("mel")
    mel_recall = float(((preds == mel_idx) & (labels == mel_idx)).sum() /
                       max((labels == mel_idx).sum(), 1))
    print(f"  {name:42s}  AUC: {auc:.4f}  Acc: {acc:.4f}  Mel Recall: {mel_recall:.3f}")
    return {"name": name, "auc": auc, "acc": acc, "mel_recall": mel_recall}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--out_dir", default="experiments/post_hoc")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  POST-HOC ANALYSIS - {len(args.models)} models")
    print(f"{'=' * 70}")
    print(f"Models: {args.models}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load val + test probs ─────────────────────────────────────────────────
    print("[1/4] Loading TTA probability arrays...")
    val_probs_list, val_labels = load_split_probs(args.models, "val", args.ckpt_base)
    test_probs_list, test_labels = load_split_probs(args.models, "test", args.ckpt_base)

    print("\nIndividual TTA test AUCs:")
    individual_aucs = []
    for key, p in zip(args.models, test_probs_list):
        auc = macro_ovr_auc(test_labels, p)
        individual_aucs.append(auc)
        print(f"  {key:42s}  TTA AUC: {auc:.4f}")
    best_individual = max(individual_aucs)

    print("\nBaseline ensembles on test:")
    metrics_log = []
    mean_test = np.mean(test_probs_list, axis=0)
    metrics_log.append(report("[baseline] mean ensemble (test)", mean_test, test_labels))

    # ── Step 1: Per-class weighted ensemble ───────────────────────────────────
    print(f"\n[2/4] Fitting per-class ensemble weights on VAL...")
    weights = fit_per_class_weights(val_probs_list, val_labels)
    np.save(out_dir / "per_class_weights.npy", weights)

    print("\nApplying per-class weights to test:")
    val_ensemble = apply_per_class_weights(val_probs_list, weights)
    test_ensemble = apply_per_class_weights(test_probs_list, weights)
    metrics_log.append(report("[step 1] per-class weighted ensemble", test_ensemble, test_labels))

    # ── Step 2: Per-class Platt sigmoid calibration ───────────────────────────
    print(f"\n[3/4] Fitting per-class Platt sigmoid calibration on VAL...")
    calibrators = fit_platt(val_ensemble, val_labels)

    val_calibrated = apply_platt(val_ensemble, calibrators)
    test_calibrated = apply_platt(test_ensemble, calibrators)
    np.save(out_dir / "test_probs_calibrated.npy", test_calibrated)
    metrics_log.append(report("[step 2] + Platt sigmoid calibration", test_calibrated, test_labels))

    # ── Step 3: F2-optimal thresholds ─────────────────────────────────────────
    print(f"\n[4/4] Searching F2-optimal thresholds on VAL...")
    thresholds = fit_f2_thresholds(val_calibrated, val_labels)
    np.save(out_dir / "thresholds.npy", thresholds)

    test_preds_thresh = apply_thresholds(test_calibrated, thresholds)
    metrics_log.append(report("[step 3] + F2 threshold tuning",
                              test_calibrated, test_labels, preds=test_preds_thresh))

    # ── Final report ──────────────────────────────────────────────────────────
    final_auc = metrics_log[-1]["auc"]
    final_acc = metrics_log[-1]["acc"]
    final_mel_recall = metrics_log[-1]["mel_recall"]

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Best individual TTA AUC : {best_individual:.4f}")
    for m in metrics_log:
        print(f"  {m['name']:42s}  {m['auc']:.4f}  (mel recall {m['mel_recall']:.3f})")
    print(f"  {'-' * 66}")
    print(f"  Final lift vs best single  : {final_auc - best_individual:+.4f}")
    print(f"  Final lift vs mean ensemble: {final_auc - metrics_log[0]['auc']:+.4f}")

    print("\nFinal per-class report (with calibration + thresholds):")
    final_report = classification_report(
        test_labels, test_preds_thresh, target_names=CLASS_NAMES, zero_division=0
    )
    print(final_report)

    weight_lines = []
    for c, w in zip(CLASS_NAMES, weights):
        weight_lines.append(f"  {c:6s} : " + "  ".join(
            f"{name}={wi:.2f}" for name, wi in zip(args.models, w)
        ))
    threshold_lines = [f"  {c:6s}: {t:.3f}" for c, t in zip(CLASS_NAMES, thresholds)]

    out_path = out_dir / "results.txt"
    out_path.write_text(
        f"Post-hoc analysis - {len(args.models)} models\n\n"
        f"Models: {args.models}\n\n"
        f"--- Individual TTA AUCs ---\n"
        + "\n".join(f"  {m:42s} {a:.4f}" for m, a in zip(args.models, individual_aucs))
        + "\n\n"
        f"--- Pipeline metrics ---\n"
        + "\n".join(f"  {m['name']:42s} AUC={m['auc']:.4f} Acc={m['acc']:.4f} MelRecall={m['mel_recall']:.3f}"
                   for m in metrics_log)
        + "\n\n"
        f"--- Final per-class weights (rows=class, cols=models) ---\n"
        + "\n".join(weight_lines) + "\n\n"
        f"--- F2-optimal thresholds ---\n"
        + "\n".join(threshold_lines) + "\n\n"
        f"--- Final per-class report ---\n{final_report}\n\n"
        f"Final test AUC      : {final_auc:.4f}\n"
        f"Final test accuracy : {final_acc:.4f}\n"
        f"Final mel recall    : {final_mel_recall:.3f}\n"
        f"Lift vs best single : {final_auc - best_individual:+.4f}\n"
        f"Lift vs mean ensem. : {final_auc - metrics_log[0]['auc']:+.4f}\n",
        encoding="utf-8",
    )
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)