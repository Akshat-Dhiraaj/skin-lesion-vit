"""
phase22_quick_checks.py
------------------------
Three quick experiments on the Phase 20 .npy artifacts. CPU-only, ~60 seconds.
No GPU work — purely loads existing arrays and runs sklearn / numpy math.

Baseline (from Phase 21):
    LogReg on 42 stacked logit features  →  Test AUC 0.9840

Checks here:
  1. CALIBRATION
     Fit per-class isotonic regression on LogReg val probs, apply to test.
     AUC is rank-invariant so won't change much, but threshold-based metrics
     (mel recall/precision) typically improve 0.01–0.03 because thresholds
     applied to calibrated probabilities are more meaningful.

  2. CLEANER MEL-BINARY EXPANSION
     Phase 21 expanded mel_binary's (N, 2) outputs into (N, 7) by diffusing
     (1-P(mel))/6 across non-mel columns. The result was 6 perfectly
     collinear columns that LogReg had to fight through regularization.
     This check rebuilds the feature matrix using ONLY P(mel) as a single
     scalar feature: 5 models × 7 logits + 1 P(mel) = 36 features instead
     of 42. Expected: +0.001–0.003 AUC.

  3. CONVNEXT + LOGREG WEIGHTED AVERAGE
     ConvNeXt standalone test AUC is 0.9880 — already higher than the
     0.9840 LogReg ensemble. Try w·ConvNeXt + (1-w)·LogReg with w tuned
     on val. If best w is 1.0, the meta-learner is hurting and you should
     ship ConvNeXt alone. If best w is in (0, 1), there's complementary
     signal worth keeping.

OUTPUT
------
  experiments/phase22_quick_checks/results.txt
    — comparison table of baseline + all three checks

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase22_quick_checks.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
CKPT_BASE = BASE_DIR / "experiments"
IN_DIR    = CKPT_BASE / "phase20_val_inference"
OUT_DIR   = CKPT_BASE / "phase22_quick_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX     = 0
DF_IDX      = 5
VASC_IDX    = 6
THRESHOLD_CLASSES = [VASC_IDX, DF_IDX, MEL_IDX]   # rarest first → highest priority

RANDOM_STATE = 42

# ── Helpers (mirrored from phase21) ───────────────────────────────────────────

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    p = probs.astype(np.float64)
    p = np.clip(p, 1e-10, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def expand_mel_binary_logits(binary_logits: np.ndarray) -> np.ndarray:
    """Same as Phase 21 — used only for the BASELINE recomputation."""
    n = binary_logits.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    out[:, MEL_IDX] = binary_logits[:, 1]
    for c in range(NUM_CLASSES):
        if c != MEL_IDX:
            out[:, c] = binary_logits[:, 0]
    return out


def tune_minority_thresholds(ensemble_probs: np.ndarray, labels: np.ndarray) -> dict:
    """F2-optimal threshold per minority class. Tune on val."""
    thresholds = {}
    for cls_idx in THRESHOLD_CLASSES:
        cls_probs = ensemble_probs[:, cls_idx]
        y_binary  = (labels == cls_idx).astype(int)
        n_pos     = int(y_binary.sum())
        if n_pos == 0:
            thresholds[cls_idx] = 0.5
            continue
        pos_mean = float(cls_probs[y_binary == 1].mean())
        t_max    = min(0.60, pos_mean + 0.15)
        best_t, best_f2 = 0.0, 0.0
        for t in np.linspace(0.01, t_max, 300):
            preds = (cls_probs > t).astype(int)
            score = fbeta_score(y_binary, preds, beta=2, zero_division=0)
            if score > best_f2:
                best_f2, best_t = score, t
        thresholds[cls_idx] = best_t
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    preds = np.argmax(probs, axis=1).copy()
    for cls_idx in reversed(THRESHOLD_CLASSES):
        if cls_idx in thresholds:
            mask = probs[:, cls_idx] > thresholds[cls_idx]
            preds[mask] = cls_idx
    return preds


def evaluate_full(
    val_probs:   np.ndarray,
    test_probs:  np.ndarray,
    val_labels:  np.ndarray,
    test_labels: np.ndarray,
    name:        str,
) -> dict:
    """Compute val/test AUC + per-class AUC + mel metrics with val-tuned thresholds."""
    val_probs  = normalize_probs(val_probs)
    test_probs = normalize_probs(test_probs)

    val_auc  = roc_auc_score(val_labels,  val_probs,  multi_class='ovr', average='macro')
    test_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    mel_auc  = roc_auc_score(
        (test_labels == MEL_IDX).astype(int), test_probs[:, MEL_IDX]
    )

    # Per-class test AUC
    per_class_auc = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class_auc[cls] = roc_auc_score(
            (test_labels == i).astype(int), test_probs[:, i]
        )

    # Threshold-based metrics on test, thresholds tuned on val
    thresholds   = tune_minority_thresholds(val_probs, val_labels)
    final_preds  = apply_thresholds(test_probs, thresholds)
    mel_mask     = test_labels == MEL_IDX
    mel_recall   = float((final_preds[mel_mask] == MEL_IDX).mean())
    n_pred_mel   = int((final_preds == MEL_IDX).sum())
    mel_precision = float(
        ((final_preds == MEL_IDX) & (test_labels == MEL_IDX)).sum()
    ) / max(n_pred_mel, 1)
    accuracy = (final_preds == test_labels).mean()

    print(f"\n  [{name}]")
    print(f"    Val AUC      : {val_auc:.4f}")
    print(f"    Test AUC     : {test_auc:.4f}")
    print(f"    Mel AUC      : {mel_auc:.4f}")
    print(f"    Mel recall   : {mel_recall:.4f}")
    print(f"    Mel precision: {mel_precision:.4f}")
    print(f"    Accuracy     : {accuracy:.4f}")

    return {
        "name":          name,
        "val_auc":       val_auc,
        "test_auc":      test_auc,
        "mel_auc":       mel_auc,
        "mel_recall":    mel_recall,
        "mel_precision": mel_precision,
        "accuracy":      accuracy,
        "per_class_auc": per_class_auc,
        "thresholds":    thresholds,
        "final_preds":   final_preds,
        "test_probs":    test_probs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PHASE 22: QUICK CHECKS  (calibration, cleaner mel, ConvNeXt blend)")
    print("=" * 70)

    # ── Load Phase 20 outputs ─────────────────────────────────────────────────
    with open(IN_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"\nManifest models: {manifest['models']}")

    val_labels  = np.load(IN_DIR / "val_labels.npy")
    test_labels = np.load(IN_DIR / "test_labels.npy")

    # Load all probs/logits
    probs  = {"val": {}, "test": {}}
    logits = {"val": {}, "test": {}}
    for key in manifest["models"]:
        for split in ("val", "test"):
            probs[split][key]  = np.load(IN_DIR / f"{key}_{split}_probs.npy")
            logits[split][key] = np.load(IN_DIR / f"{key}_{split}_logits.npy")

    print("\nLoaded artifacts:")
    for k in manifest["models"]:
        print(f"  {k:12s}  val_probs {probs['val'][k].shape}  "
              f"val_logits {logits['val'][k].shape}")

    model_keys = list(manifest["models"])

    # ──────────────────────────────────────────────────────────────────────────
    # BASELINE — recompute Phase 21's LogReg result
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  BASELINE: LogReg on 42 expanded-logit features (Phase 21)")
    print("═" * 70)

    # Replicate Phase 21's expansion of mel_binary logits (N, 2) → (N, 7)
    l_val_baseline  = []
    l_test_baseline = []
    for k in model_keys:
        if k == "mel_binary":
            l_val_baseline.append(expand_mel_binary_logits(logits["val"][k]))
            l_test_baseline.append(expand_mel_binary_logits(logits["test"][k]))
        else:
            l_val_baseline.append(logits["val"][k])
            l_test_baseline.append(logits["test"][k])

    feat_val_baseline  = np.hstack(l_val_baseline)
    feat_test_baseline = np.hstack(l_test_baseline)
    print(f"  Feature shape: val={feat_val_baseline.shape}, test={feat_test_baseline.shape}")

    logreg_baseline = LogisticRegression(
        solver='lbfgs', max_iter=2000, C=1.0, random_state=RANDOM_STATE,
    )
    logreg_baseline.fit(feat_val_baseline, val_labels)
    baseline_val_probs  = logreg_baseline.predict_proba(feat_val_baseline)
    baseline_test_probs = logreg_baseline.predict_proba(feat_test_baseline)

    res_baseline = evaluate_full(
        baseline_val_probs, baseline_test_probs,
        val_labels, test_labels,
        "BASELINE: LogReg (Phase 21)",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CHECK 1 — Per-class isotonic calibration on baseline LogReg
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  CHECK 1: Per-class isotonic calibration of LogReg outputs")
    print("═" * 70)

    val_probs_bl  = normalize_probs(baseline_val_probs)
    test_probs_bl = normalize_probs(baseline_test_probs)

    cal_val  = np.zeros_like(val_probs_bl)
    cal_test = np.zeros_like(test_probs_bl)
    for c in range(NUM_CLASSES):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        y_binary_val = (val_labels == c).astype(float)
        iso.fit(val_probs_bl[:, c], y_binary_val)
        cal_val[:, c]  = iso.transform(val_probs_bl[:, c])
        cal_test[:, c] = iso.transform(test_probs_bl[:, c])

    res_check1 = evaluate_full(
        cal_val, cal_test, val_labels, test_labels,
        "CHECK 1: Calibrated LogReg (per-class isotonic)",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CHECK 2 — LogReg with cleaner mel-binary feature (just P(mel))
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  CHECK 2: LogReg with cleaner mel-binary expansion")
    print("═" * 70)

    other_keys = [k for k in model_keys if k != "mel_binary"]
    feat_val_clean_parts  = [logits["val"][k]  for k in other_keys]
    feat_test_clean_parts = [logits["test"][k] for k in other_keys]

    if "mel_binary" in model_keys:
        # Use the raw (N, 2) probabilities — take column 1 = P(mel) as one feature
        p_mel_val  = probs["val"]["mel_binary"][:, 1].reshape(-1, 1)
        p_mel_test = probs["test"]["mel_binary"][:, 1].reshape(-1, 1)
        feat_val_clean_parts.append(p_mel_val)
        feat_test_clean_parts.append(p_mel_test)

    feat_val_clean  = np.hstack(feat_val_clean_parts)
    feat_test_clean = np.hstack(feat_test_clean_parts)
    print(f"  Feature shape: val={feat_val_clean.shape}, test={feat_test_clean.shape}  "
          f"(was 42, now {feat_val_clean.shape[1]})")

    logreg_clean = LogisticRegression(
        solver='lbfgs', max_iter=2000, C=1.0, random_state=RANDOM_STATE,
    )
    logreg_clean.fit(feat_val_clean, val_labels)
    clean_val_probs  = logreg_clean.predict_proba(feat_val_clean)
    clean_test_probs = logreg_clean.predict_proba(feat_test_clean)

    res_check2 = evaluate_full(
        clean_val_probs, clean_test_probs,
        val_labels, test_labels,
        "CHECK 2: LogReg w/ cleaner mel-binary",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CHECK 3 — ConvNeXt + LogReg(baseline) weighted average
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  CHECK 3: ConvNeXt + LogReg(baseline) weighted average")
    print("═" * 70)

    if "convnext" not in model_keys:
        print("  ConvNeXt not in manifest — skipping CHECK 3.")
        res_check3 = None
        res_convnext_alone = None
    else:
        cn_val  = normalize_probs(probs["val"]["convnext"])
        cn_test = normalize_probs(probs["test"]["convnext"])
        lr_val  = normalize_probs(baseline_val_probs)
        lr_test = normalize_probs(baseline_test_probs)

        # First, ConvNeXt alone — establish the upper bound
        # (val_probs is just convnext val for honest val AUC)
        res_convnext_alone = evaluate_full(
            cn_val, cn_test, val_labels, test_labels,
            "         ConvNeXt alone (w=1.0)",
        )

        # Scan w from 0.0 to 1.0
        best_w        = 0.0
        best_val_auc  = -np.inf
        for w in np.arange(0.0, 1.001, 0.05):
            blend_val = normalize_probs(w * cn_val + (1 - w) * lr_val)
            v_auc     = roc_auc_score(
                val_labels, blend_val, multi_class='ovr', average='macro'
            )
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_w       = w

        print(f"\n  Best blend weight (val): w={best_w:.2f}  "
              f"(ConvNeXt={best_w:.2f}, LogReg={1-best_w:.2f})")
        print(f"  Best val AUC: {best_val_auc:.4f}")

        blend_val_best  = w_norm = normalize_probs(
            best_w * cn_val + (1 - best_w) * lr_val
        )
        blend_test_best = normalize_probs(
            best_w * cn_test + (1 - best_w) * lr_test
        )
        res_check3 = evaluate_full(
            blend_val_best, blend_test_best,
            val_labels, test_labels,
            f"CHECK 3: ConvNeXt({best_w:.2f}) + LogReg({1-best_w:.2f})",
        )
        res_check3["best_w"] = best_w

    # ──────────────────────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  COMPARISON TABLE")
    print("═" * 70)

    rows = [res_baseline, res_check1, res_check2]
    if res_convnext_alone is not None:
        rows.append(res_convnext_alone)
    if res_check3 is not None:
        rows.append(res_check3)

    header = (
        f"{'Variant':45s}  "
        f"{'Val AUC':>8s}  {'Test AUC':>9s}  "
        f"{'Mel AUC':>8s}  {'MelRcl':>7s}  {'MelPrc':>7s}  {'Acc':>6s}"
    )
    print("\n" + header)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r['name']:45s}  "
            f"{r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}  "
            f"{r['mel_auc']:>8.4f}  {r['mel_recall']:>7.4f}  "
            f"{r['mel_precision']:>7.4f}  {r['accuracy']:>6.4f}"
        )

    # Identify the winner by TEST AUC for transparency
    winner_by_test = max(rows, key=lambda r: r["test_auc"])
    print(f"\n  Best by Test AUC : {winner_by_test['name']}  "
          f"({winner_by_test['test_auc']:.4f})")

    # And by VAL AUC for honest selection
    winner_by_val = max(rows, key=lambda r: r["val_auc"])
    print(f"  Best by Val AUC  : {winner_by_val['name']}  "
          f"(val={winner_by_val['val_auc']:.4f}, "
          f"test={winner_by_val['test_auc']:.4f})  ← honest choice")

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────────────────────────
    np.save(OUT_DIR / "calibrated_test_probs.npy",  res_check1["test_probs"])
    np.save(OUT_DIR / "clean_logreg_test_probs.npy", res_check2["test_probs"])
    if res_check3 is not None:
        np.save(OUT_DIR / "convnext_blend_test_probs.npy", res_check3["test_probs"])
    np.save(OUT_DIR / "test_labels.npy", test_labels)

    with open(OUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("=== PHASE 22 QUICK CHECKS ===\n\n")
        f.write("All variants use Phase 20 .npy artifacts. CPU only.\n")
        f.write("Threshold-based metrics use F2-optimal thresholds tuned on VAL.\n\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            f.write(
                f"{r['name']:45s}  "
                f"{r['val_auc']:>8.4f}  {r['test_auc']:>9.4f}  "
                f"{r['mel_auc']:>8.4f}  {r['mel_recall']:>7.4f}  "
                f"{r['mel_precision']:>7.4f}  {r['accuracy']:>6.4f}\n"
            )

        f.write(f"\nBest by Test AUC : {winner_by_test['name']}  "
                f"({winner_by_test['test_auc']:.4f})\n")
        f.write(f"Best by Val AUC  : {winner_by_val['name']}  "
                f"(val={winner_by_val['val_auc']:.4f}, "
                f"test={winner_by_val['test_auc']:.4f})  <- honest choice\n\n")

        # Per-class AUC for the val-AUC winner
        f.write(f"Per-class TEST AUC for best (by val): {winner_by_val['name']}\n")
        for cls, a in winner_by_val["per_class_auc"].items():
            mark = "  <-- melanoma" if cls == "mel" else ""
            f.write(f"  {cls:8s}: {a:.4f}{mark}\n")

        f.write(f"\nThresholds (val-tuned) for {winner_by_val['name']}:\n")
        for cls_idx, t in winner_by_val["thresholds"].items():
            f.write(f"  {CLASS_NAMES[cls_idx]:8s}: {t:.4f}\n")

        f.write(f"\nClassification Report (TEST, thresholds applied):\n")
        f.write(classification_report(
            test_labels, winner_by_val["final_preds"],
            target_names=CLASS_NAMES,
        ))

        if res_check3 is not None:
            f.write(f"\nCHECK 3 best blend weight: w={res_check3['best_w']:.2f}\n")
            f.write(f"  ConvNeXt contribution: {res_check3['best_w']:.2f}\n")
            f.write(f"  LogReg contribution  : {1 - res_check3['best_w']:.2f}\n")
            if res_check3["best_w"] >= 0.95:
                f.write("  → ConvNeXt dominates; meta-learner adds little. "
                        "Consider shipping ConvNeXt alone.\n")
            elif res_check3["best_w"] <= 0.05:
                f.write("  → LogReg dominates; ConvNeXt adds little.\n")
            else:
                f.write("  → Genuine complementary signal between the two.\n")

    print(f"\n[SUCCESS] Saved to: {OUT_DIR}\\results.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()