"""
phase21_final_ensemble.py
--------------------------
Final ensemble step. Loads the val + test probs/logits saved by Phase 20,
trains multiple meta-learners on VAL, evaluates on TEST, picks the best
combination, applies multi-class thresholds, reports honest numbers.

WHY MULTIPLE META-LEARNERS
---------------------------
Different meta-learners exploit different structure:
  - SLSQP per-class weights:     simple linear combination, low capacity
                                 (21 params for 3 models × 7 classes)
  - Logistic regression on logits: learns cross-class corrections, e.g.
                                 "if mel-binary says mel and EfficientNet
                                 says nv, trust mel-binary more"
  - XGBoost on probs:            captures non-linear interactions
                                 (300 trees × ~30 features = high capacity)

We run all three, pick the one with best val AUC, and ALSO try a
"meta-meta-ensemble" that averages all three meta-learners' outputs.

LOGIT vs PROBABILITY AVERAGING
-------------------------------
We try BOTH and pick whichever has higher val AUC for SLSQP:
  - probability averaging:  current Phase 16/17 approach
  - logit averaging:        softmax(mean(logits)) — preserves more signal
                            because softmax compression is reversed before
                            averaging

Empirically logit averaging helps 0.001–0.003 AUC on imbalanced data, but
not always, so we measure rather than assume.

MEL BINARY INTEGRATION
----------------------
The mel binary specialist outputs (N, 2) — P(non-mel), P(mel).
We append P(mel) as a 7-class feature where:
  feature_mel = P(mel)
  feature_other_classes = (1 - P(mel)) / 6  # diffused over non-mel classes

This converts the binary signal into a 7-vector compatible with the
ensemble while preserving the mel-specialist signal cleanly.

PER-CLASS THRESHOLDS
--------------------
For mel, df, and vasc (the three rarest classes), we tune F2-optimal
decision thresholds on VAL probs and apply them to TEST predictions.
Priority order: vasc > df > mel > argmax (rarest wins conflicts).

OUTPUTS (saved to experiments/phase21_final_ensemble/)
-------------------------------------------------------
  ensemble_test_probs.npy       # (1503, 7) final test probabilities
  test_labels.npy               # (1503,) ground truth
  meta_learner_comparison.txt   # AUCs for each meta-learner
  results.txt                   # final macro AUC + per-class report

Usage:
    cd D:\\ML\\skin-lesion-vit\\src
    python phase21_final_ensemble.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score, classification_report, fbeta_score,
)
from sklearn.linear_model import LogisticRegression
HAVE_XGB = False   # force disable — XGBoost overfits on 1502 val samples
# XGBoost is optional; fall back gracefully if not installed
"""
try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
"""
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\ML\skin-lesion-vit")
CKPT_BASE = BASE_DIR / "experiments"
IN_DIR    = CKPT_BASE / "phase20_val_inference"
OUT_DIR   = CKPT_BASE / "phase21_final_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX     = 0
DF_IDX      = 5
VASC_IDX    = 6
THRESHOLD_CLASSES = [VASC_IDX, DF_IDX, MEL_IDX]   # rarest first → highest priority

N_SLSQP_RESTARTS = 20    # more restarts than Phase 17 since we're at the
                         #  final step and time-cheap
RANDOM_STATE     = 42

_RNG = np.random.default_rng(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def load_phase20_outputs():
    """
    Load every {model}_{split}_probs.npy and _logits.npy that exists.
    Returns dicts: probs_dict, logits_dict, labels_dict
    """
    if not (IN_DIR / "manifest.json").exists():
        raise FileNotFoundError(
            f"manifest.json not found in {IN_DIR}. Run phase20_val_set_inference.py first."
        )

    with open(str(IN_DIR / "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Manifest models: {manifest['models']}")

    val_labels  = np.load(str(IN_DIR / "val_labels.npy"))
    test_labels = np.load(str(IN_DIR / "test_labels.npy"))
    print(f"Val labels:  {val_labels.shape}  mel positives: {(val_labels == MEL_IDX).sum()}")
    print(f"Test labels: {test_labels.shape} mel positives: {(test_labels == MEL_IDX).sum()}")

    probs_dict  = {"val": {}, "test": {}}
    logits_dict = {"val": {}, "test": {}}

    for key in manifest["models"]:
        for split in ("val", "test"):
            p_path = IN_DIR / f"{key}_{split}_probs.npy"
            l_path = IN_DIR / f"{key}_{split}_logits.npy"
            if not p_path.exists() or not l_path.exists():
                print(f"  WARNING: missing {key}/{split} — skipping model in ensemble")
                continue
            probs_dict[split][key]  = np.load(str(p_path))
            logits_dict[split][key] = np.load(str(l_path))

    return probs_dict, logits_dict, val_labels, test_labels, manifest


def expand_mel_binary(binary_probs: np.ndarray) -> np.ndarray:
    """
    Convert (N, 2) mel-binary probabilities to (N, 7) so it fits in the ensemble.

    Strategy:
      column[mel]        = P(mel)
      columns[other]    = (1 - P(mel)) / 6   # diffused mass across 6 non-mel classes

    This preserves the binary signal exactly on the mel column while making
    the array shape-compatible with the 7-class ensemble math. The meta-learner
    will figure out that mel-binary's non-mel columns are uninformative.
    """
    n = binary_probs.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    p_mel = binary_probs[:, 1].astype(np.float64)
    out[:, MEL_IDX] = p_mel
    diffused = (1.0 - p_mel) / (NUM_CLASSES - 1)
    for c in range(NUM_CLASSES):
        if c != MEL_IDX:
            out[:, c] = diffused
    return out


def expand_mel_binary_logits(binary_logits: np.ndarray) -> np.ndarray:
    """
    For logits we don't have the same trick. Take the mel logit as-is for the
    mel column and fill other columns with the non-mel logit.
    This is approximate — the meta-learner will weight accordingly.
    """
    n = binary_logits.shape[0]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    out[:, MEL_IDX] = binary_logits[:, 1]
    for c in range(NUM_CLASSES):
        if c != MEL_IDX:
            out[:, c] = binary_logits[:, 0]
    return out


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    p = probs.astype(np.float64)
    p = np.clip(p, 1e-10, 1.0)
    return p / p.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. META-LEARNERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_slsqp_per_class_weights(
    probs_list_val:  list,
    labels_val:      np.ndarray,
    n_restarts:      int = N_SLSQP_RESTARTS,
) -> tuple:
    """
    SLSQP per-class weight optimization.

    Variables: w[m, c] for each model m, class c
    Constraints: sum over m of w[m, c] = 1 for each c
    Bounds: 0 ≤ w[m, c] ≤ 1
    Objective: maximize macro-AUC of normalized weighted sum on val labels
    """
    num_models = len(probs_list_val)

    def objective(flat_w):
        w   = flat_w.reshape(num_models, NUM_CLASSES)
        ens = normalize_probs(
            sum(probs_list_val[m] * w[m] for m in range(num_models))
        )
        return -roc_auc_score(labels_val, ens, multi_class='ovr', average='macro')

    constraints = [
        {'type': 'eq',
         'fun': lambda w, c=c: np.sum(
             w.reshape(num_models, NUM_CLASSES)[:, c]
         ) - 1}
        for c in range(NUM_CLASSES)
    ]
    bounds = [(0.0, 1.0)] * (num_models * NUM_CLASSES)

    best_auc = -np.inf
    best_w   = None
    print(f"  SLSQP: {n_restarts} restarts, {num_models} models × {NUM_CLASSES} classes...")

    for i in range(n_restarts):
        if i == 0:
            init_w = np.ones((num_models, NUM_CLASSES)) / num_models
        else:
            init_w = _RNG.dirichlet(np.ones(num_models), size=NUM_CLASSES).T

        res = minimize(
            objective, init_w.flatten(),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10},
        )
        auc = -res.fun
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_w   = res.x.reshape(num_models, NUM_CLASSES)

    if best_w is None:
        raise RuntimeError("SLSQP: all restarts failed.")

    print(f"  SLSQP best val AUC: {best_auc:.4f}")
    return best_w, best_auc


def apply_slsqp_weights(probs_list: list, weights: np.ndarray) -> np.ndarray:
    """Apply SLSQP per-class weights to produce ensemble probs."""
    num_models = len(probs_list)
    ens = sum(probs_list[m] * weights[m] for m in range(num_models))
    return normalize_probs(ens)


def fit_logistic_regression(
    feature_matrix_val: np.ndarray,
    labels_val:         np.ndarray,
    C: float = 1.0,
) -> LogisticRegression:
    """
    Multinomial logistic regression on stacked logits/probs from all models.
    L2 regularization (C=1.0) prevents overfitting on small val set.

    Note: multi_class='multinomial' parameter was removed in sklearn 1.7.
    The lbfgs solver auto-uses multinomial for multi-class problems.
    """
    print(f"  LogReg: features={feature_matrix_val.shape}, C={C}")
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=C,
        random_state=RANDOM_STATE,
    )
    clf.fit(feature_matrix_val, labels_val)
    val_pred = clf.predict_proba(feature_matrix_val)
    val_auc  = roc_auc_score(labels_val, val_pred, multi_class='ovr', average='macro')
    print(f"  LogReg val AUC: {val_auc:.4f}")
    return clf, val_auc


def fit_xgboost(
    feature_matrix_val: np.ndarray,
    labels_val:         np.ndarray,
):
    """
    XGBoost meta-learner on stacked features.
    Trained ONLY on val — test is held out, so this is leak-free.
    """
    if not HAVE_XGB:
        return None, None
    print(f"  XGBoost: features={feature_matrix_val.shape}")
    clf = xgb.XGBClassifier(
        n_estimators       = 300,
        max_depth          = 4,
        learning_rate      = 0.05,
        objective          = 'multi:softprob',
        eval_metric        = 'mlogloss',
        num_class          = NUM_CLASSES,
        random_state       = RANDOM_STATE,
        tree_method        = 'hist',
        # subsample helps regularize on small data
        subsample          = 0.8,
        colsample_bytree   = 0.8,
    )
    clf.fit(feature_matrix_val, labels_val)
    val_pred = clf.predict_proba(feature_matrix_val)
    val_auc  = roc_auc_score(labels_val, val_pred, multi_class='ovr', average='macro')
    print(f"  XGBoost val AUC: {val_auc:.4f}  (likely overfit — see test AUC for truth)")
    return clf, val_auc


# ══════════════════════════════════════════════════════════════════════════════
# 3. THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

def tune_minority_thresholds(
    ensemble_probs: np.ndarray,
    labels:         np.ndarray,
) -> dict:
    """F2-optimal threshold per minority class. Tune on val, apply to test."""
    thresholds = {}
    print("\nTuning minority thresholds (F2, β=2) on val:")
    for cls_idx in THRESHOLD_CLASSES:
        cls_name  = CLASS_NAMES[cls_idx]
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
        recall = ((cls_probs > best_t) & (y_binary == 1)).sum() / n_pos
        print(f"  {cls_name:8s}: t={best_t:.4f}  F2={best_f2:.4f}  "
              f"recall={recall:.3f}  (n_pos={n_pos})")

    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """Apply class thresholds rarest-last so rare classes win conflicts."""
    preds = np.argmax(probs, axis=1).copy()
    for cls_idx in reversed(THRESHOLD_CLASSES):
        if cls_idx in thresholds:
            mask = probs[:, cls_idx] > thresholds[cls_idx]
            preds[mask] = cls_idx
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 4. AUC HELPER
# ══════════════════════════════════════════════════════════════════════════════

def report_aucs(probs: np.ndarray, labels: np.ndarray, tag: str) -> tuple:
    """Print and return (macro_auc, per_class_auc dict)."""
    macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    per_class = {}
    print(f"\n  [{tag}] Macro AUC: {macro:.4f}")
    for i, cls in enumerate(CLASS_NAMES):
        binary = (labels == i).astype(int)
        a = roc_auc_score(binary, probs[:, i])
        per_class[cls] = a
        marker = "  <-- melanoma" if cls == "mel" else ""
        print(f"    {cls:8s}: {a:.4f}{marker}")
    return macro, per_class


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PHASE 21: FINAL ENSEMBLE  (meta-learners on val → test eval)")
    print("=" * 70)

    # ── Load Phase 20 outputs ─────────────────────────────────────────────────
    probs_dict, logits_dict, val_labels, test_labels, manifest = load_phase20_outputs()
    n_val  = len(val_labels)
    n_test = len(test_labels)

    # ── Expand mel-binary (if present) into 7-class shape ─────────────────────
    model_keys = list(probs_dict["val"].keys())   # only models we actually loaded
    print(f"\nExpanding mel-binary to 7-class shape if present...")
    if "mel_binary" in model_keys:
        for split in ("val", "test"):
            probs_dict[split]["mel_binary"] = expand_mel_binary(
                probs_dict[split]["mel_binary"]
            )
            logits_dict[split]["mel_binary"] = expand_mel_binary_logits(
                logits_dict[split]["mel_binary"]
            )
        print(f"  mel_binary expanded: shapes now (N, 7)")

    # ── Sanity check: standalone val AUCs ─────────────────────────────────────
    print("\n── Standalone VAL AUCs (sanity check) ──")
    for k in model_keys:
        p = probs_dict["val"][k]
        if p.shape[1] == NUM_CLASSES:
            auc = roc_auc_score(val_labels, p, multi_class='ovr', average='macro')
            mel = roc_auc_score((val_labels == MEL_IDX).astype(int), p[:, MEL_IDX])
            print(f"  {k:12s}  Macro AUC: {auc:.4f}  Mel AUC: {mel:.4f}")

    # ── Build aligned probs lists (model_keys order is fixed) ─────────────────
    p_val  = [probs_dict["val"][k]  for k in model_keys]
    p_test = [probs_dict["test"][k] for k in model_keys]
    l_val  = [logits_dict["val"][k]  for k in model_keys]
    l_test = [logits_dict["test"][k] for k in model_keys]

    # Pre-computed: softmax(mean(logits)) per model — equivalent to logit-avg TTA
    # NOTE: Phase 20 already averaged logits across TTA views. Here we have
    #       the per-model averaged logits. To compare logit vs prob averaging
    #       at the ENSEMBLE level, we use these already-averaged logits.
    def softmax_logits(logits_arr: np.ndarray) -> np.ndarray:
        exp = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    # Per-model softmax-of-logits (for logit-space ensemble comparison)
    p_val_from_logits  = [softmax_logits(l) for l in l_val]
    p_test_from_logits = [softmax_logits(l) for l in l_test]

    # ──────────────────────────────────────────────────────────────────────────
    # META-LEARNER #1: SLSQP — probability averaging
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  META-LEARNER 1: SLSQP per-class weights (probability averaging)")
    print("═" * 70)
    slsqp_w_p, slsqp_val_auc_p = fit_slsqp_per_class_weights(p_val, val_labels)
    slsqp_test_p = apply_slsqp_weights(p_test, slsqp_w_p)
    slsqp_test_auc_p, _ = report_aucs(slsqp_test_p, test_labels, "SLSQP-prob test")

    # ──────────────────────────────────────────────────────────────────────────
    # META-LEARNER #2: SLSQP — logit averaging
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  META-LEARNER 2: SLSQP per-class weights (logit averaging)")
    print("═" * 70)
    slsqp_w_l, slsqp_val_auc_l = fit_slsqp_per_class_weights(
        p_val_from_logits, val_labels,
    )
    slsqp_test_l = apply_slsqp_weights(p_test_from_logits, slsqp_w_l)
    slsqp_test_auc_l, _ = report_aucs(slsqp_test_l, test_labels, "SLSQP-logit test")

    # Pick the better SLSQP variant (compare on val to avoid test peeking)
    if slsqp_val_auc_l > slsqp_val_auc_p:
        slsqp_test    = slsqp_test_l
        slsqp_variant = "logit"
        slsqp_test_auc = slsqp_test_auc_l
        print(f"\n  SLSQP variant choice: LOGIT averaging "
              f"(val {slsqp_val_auc_l:.4f} > {slsqp_val_auc_p:.4f})")
    else:
        slsqp_test    = slsqp_test_p
        slsqp_variant = "prob"
        slsqp_test_auc = slsqp_test_auc_p
        print(f"\n  SLSQP variant choice: PROB averaging "
              f"(val {slsqp_val_auc_p:.4f} ≥ {slsqp_val_auc_l:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # META-LEARNER #3: Logistic regression on stacked logits
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  META-LEARNER 3: Logistic regression on concatenated logits")
    print("═" * 70)
    feat_val_logits  = np.hstack(l_val)    # (n_val,  num_models * 7)
    feat_test_logits = np.hstack(l_test)
    print(f"  Feature shape: val={feat_val_logits.shape}, test={feat_test_logits.shape}")

    logreg, logreg_val_auc = fit_logistic_regression(
        feat_val_logits, val_labels, C=1.0,
    )
    logreg_test = logreg.predict_proba(feat_test_logits)
    logreg_test = normalize_probs(logreg_test)
    logreg_test_auc, _ = report_aucs(logreg_test, test_labels, "LogReg test")

    # ──────────────────────────────────────────────────────────────────────────
    # META-LEARNER #4: XGBoost on stacked probs (optional)
    # ──────────────────────────────────────────────────────────────────────────
    xgb_test = None
    xgb_test_auc = None
    if HAVE_XGB:
        print("\n" + "═" * 70)
        print("  META-LEARNER 4: XGBoost on concatenated probs")
        print("═" * 70)
        feat_val_probs  = np.hstack(p_val)
        feat_test_probs = np.hstack(p_test)
        xgb_clf, xgb_val_auc = fit_xgboost(feat_val_probs, val_labels)
        xgb_test = xgb_clf.predict_proba(feat_test_probs)
        xgb_test = normalize_probs(xgb_test)
        xgb_test_auc, _ = report_aucs(xgb_test, test_labels, "XGBoost test")
    else:
        print("\n[XGBoost not installed; skipping META-LEARNER 4]")

    # ──────────────────────────────────────────────────────────────────────────
    # META-META-ENSEMBLE: average all meta-learner outputs
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  META-META: average of all meta-learner test predictions")
    print("═" * 70)
    components = [slsqp_test, logreg_test]
    component_names = ["SLSQP", "LogReg"]
    if xgb_test is not None:
        components.append(xgb_test)
        component_names.append("XGBoost")

    meta_meta_test = normalize_probs(np.mean(components, axis=0))
    meta_meta_auc, _ = report_aucs(meta_meta_test, test_labels, "Meta-meta test")

    # ──────────────────────────────────────────────────────────────────────────
    # Pick the winning meta-learner BY VAL AUC (not test — no peeking)
    # ──────────────────────────────────────────────────────────────────────────
    candidates = {
        f"SLSQP-{slsqp_variant}": (
            slsqp_test, max(slsqp_val_auc_p, slsqp_val_auc_l), slsqp_test_auc,
        ),
        "LogReg":                  (logreg_test, logreg_val_auc, logreg_test_auc),
    }
    if xgb_test is not None:
        candidates["XGBoost"] = (xgb_test, xgb_val_auc, xgb_test_auc)

    # For meta-meta, we need a val AUC too. Compute it.
    val_components = []
    if slsqp_variant == "logit":
        val_components.append(apply_slsqp_weights(p_val_from_logits, slsqp_w_l))
    else:
        val_components.append(apply_slsqp_weights(p_val, slsqp_w_p))
    val_components.append(normalize_probs(logreg.predict_proba(feat_val_logits)))
    if HAVE_XGB and xgb_test is not None:
        val_components.append(normalize_probs(xgb_clf.predict_proba(np.hstack(p_val))))
    meta_meta_val = normalize_probs(np.mean(val_components, axis=0))
    meta_meta_val_auc = roc_auc_score(
        val_labels, meta_meta_val, multi_class='ovr', average='macro'
    )
    candidates["Meta-meta"] = (meta_meta_test, meta_meta_val_auc, meta_meta_auc)

    print("\n" + "─" * 70)
    print("CANDIDATE COMPARISON  (selection by VAL AUC)")
    print("─" * 70)
    print(f"  {'Meta-learner':16s} {'Val AUC':>10s}  {'Test AUC':>10s}")
    for name, (_, vauc, tauc) in candidates.items():
        print(f"  {name:16s} {vauc:>10.4f}  {tauc:>10.4f}")

    best_name = max(candidates.keys(), key=lambda k: candidates[k][1])
    best_test_probs, best_val_auc, best_test_auc = candidates[best_name]
    print(f"\n  → Selected: {best_name}  (val AUC {best_val_auc:.4f}, "
          f"test AUC {best_test_auc:.4f})")

    # Also save meta-meta as a bonus artifact
    final_test_probs = best_test_probs

    # ──────────────────────────────────────────────────────────────────────────
    # THRESHOLDS — tune on val, apply to test
    # ──────────────────────────────────────────────────────────────────────────
    # We need val probs from the winning meta-learner
    print("\n" + "═" * 70)
    print("  PER-CLASS THRESHOLDS (tune on val, apply to test)")
    print("═" * 70)

    if best_name.startswith("SLSQP"):
        if slsqp_variant == "logit":
            best_val_probs = apply_slsqp_weights(p_val_from_logits, slsqp_w_l)
        else:
            best_val_probs = apply_slsqp_weights(p_val, slsqp_w_p)
    elif best_name == "LogReg":
        best_val_probs = normalize_probs(logreg.predict_proba(feat_val_logits))
    elif best_name == "XGBoost":
        best_val_probs = normalize_probs(xgb_clf.predict_proba(np.hstack(p_val)))
    else:  # Meta-meta
        best_val_probs = meta_meta_val

    thresholds  = tune_minority_thresholds(best_val_probs, val_labels)
    final_preds = apply_thresholds(final_test_probs, thresholds)
    report      = classification_report(
        test_labels, final_preds, target_names=CLASS_NAMES,
    )

    print("\nFinal classification report (TEST, thresholds applied):")
    print(report)

    # ──────────────────────────────────────────────────────────────────────────
    # Final mel-specific summary
    # ──────────────────────────────────────────────────────────────────────────
    mel_auc       = roc_auc_score(
        (test_labels == MEL_IDX).astype(int),
        final_test_probs[:, MEL_IDX],
    )
    mel_mask      = test_labels == MEL_IDX
    mel_recall    = float((final_preds[mel_mask] == MEL_IDX).mean())
    n_pred_mel    = int((final_preds == MEL_IDX).sum())
    mel_precision = float(
        ((final_preds == MEL_IDX) & (test_labels == MEL_IDX)).sum()
    ) / max(n_pred_mel, 1)

    print(f"\n── Final Melanoma Metrics (TEST) ──")
    print(f"  Mel AUC       : {mel_auc:.4f}")
    print(f"  Mel recall    : {mel_recall:.4f}")
    print(f"  Mel precision : {mel_precision:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ──────────────────────────────────────────────────────────────────────────
    np.save(str(OUT_DIR / "ensemble_test_probs.npy"), final_test_probs)
    np.save(str(OUT_DIR / "test_labels.npy"),         test_labels)
    np.save(str(OUT_DIR / "final_preds.npy"),         final_preds)
    np.save(str(OUT_DIR / "slsqp_weights.npy"),
            slsqp_w_l if slsqp_variant == "logit" else slsqp_w_p)

    # Save meta-meta as bonus
    np.save(str(OUT_DIR / "meta_meta_test_probs.npy"), meta_meta_test)

    with open(str(OUT_DIR / "meta_learner_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("META-LEARNER COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write("Selected by val AUC (no test peeking).\n")
        f.write(f"Final selection: {best_name}\n\n")
        f.write(f"{'Meta-learner':16s} {'Val AUC':>10s}  {'Test AUC':>10s}\n")
        f.write("-" * 40 + "\n")
        for name, (_, vauc, tauc) in candidates.items():
            star = "  *" if name == best_name else ""
            f.write(f"{name:16s} {vauc:>10.4f}  {tauc:>10.4f}{star}\n")

    with open(str(OUT_DIR / "results.txt"), "w", encoding="utf-8") as f:
        f.write("=== PHASE 21: FINAL ENSEMBLE RESULTS ===\n\n")
        f.write(f"Models used  : {model_keys}\n")
        f.write(f"Selected     : {best_name}\n")
        f.write(f"Val AUC      : {best_val_auc:.4f}\n")
        f.write(f"Test AUC     : {best_test_auc:.4f}\n\n")

        f.write("All meta-learner test AUCs:\n")
        for name, (_, vauc, tauc) in candidates.items():
            f.write(f"  {name:16s}  val={vauc:.4f}  test={tauc:.4f}\n")
        f.write("\n")

        f.write("Thresholds applied (val-tuned, F2-optimal):\n")
        for cls_idx in THRESHOLD_CLASSES:
            t = thresholds.get(cls_idx, 0.5)
            f.write(f"  {CLASS_NAMES[cls_idx]:8s}: {t:.4f}\n")
        f.write("\n")

        f.write("Per-class TEST AUC:\n")
        for i, cls in enumerate(CLASS_NAMES):
            binary = (test_labels == i).astype(int)
            a = roc_auc_score(binary, final_test_probs[:, i])
            marker = "  <-- melanoma" if cls == "mel" else ""
            f.write(f"  {cls:8s}: {a:.4f}{marker}\n")
        f.write("\n")

        f.write(f"Melanoma summary:\n")
        f.write(f"  AUC       : {mel_auc:.4f}\n")
        f.write(f"  Recall    : {mel_recall:.4f}\n")
        f.write(f"  Precision : {mel_precision:.4f}\n\n")

        f.write("Classification Report (thresholds applied):\n")
        f.write(report)

    print(f"\n{'═' * 70}")
    print(f"[SUCCESS] Artifacts saved to: {OUT_DIR}")
    print(f"  ensemble_test_probs.npy   shape={final_test_probs.shape}")
    print(f"  meta_meta_test_probs.npy  shape={meta_meta_test.shape}")
    print(f"  results.txt               final summary")
    print(f"  meta_learner_comparison.txt")
    print(f"{'═' * 70}")
    print(f"FINAL TEST AUC: {best_test_auc:.4f}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
