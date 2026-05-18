"""
src/cross_dataset_ensemble.py
-----------------------------
Per-dataset SLSQP ensemble across external datasets.

For each external dataset (ISIC 2019, DDI, Fitzpatrick17k), fit ensemble
weights that maximize:
  (a) Macro OvR AUC (overall accuracy)
  (b) Dark-skin Malignant AUC (fairness, DDI/Fitz17k only)
and compare to:
  (c) The HAM10000-optimal weights from src.ensemble (the "global" weights)

Output: experiments/cross_dataset_ensemble/<dataset>/report.txt with the
full tradeoff matrix per model set.

Usage:
    python -m src.cross_dataset_ensemble \
        --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_arrays(dataset: str, model_keys: list[str], label_mode: str,
                ckpt_base: Path, probs_subdir: str = "cross_dataset",
                model_suffix: str = "",
               ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None]:
    """
    Load probs + labels (+ fitz_buckets if present) for the given dataset.

    Args:
        probs_subdir: 'cross_dataset' (default, no-TENT baseline) or
                      'cross_dataset_tent' to read TENT-adapted probs.
        model_suffix: append to each model key when looking up dirs
                      (used to pick a specific TENT hyperparam variant).

    Returns (probs_list, labels, buckets_or_None).
    """
    probs_list = []
    labels = None
    buckets = None
    probs_fname = f"probs_{label_mode}.npy"

    for k in model_keys:
        d = ckpt_base / probs_subdir / dataset / (k + model_suffix)
        p = d / probs_fname
        # Graceful fallback: if the suffixed variant (e.g. __best TENT) is
        # missing for this model — by design, when TENT hurt it — fall back
        # to the no-TENT baseline under cross_dataset/. "Use the best
        # available representation per model" is exactly the right semantics.
        if not p.exists() and model_suffix:
            fb = ckpt_base / "cross_dataset" / dataset / k
            fbp = fb / probs_fname
            if fbp.exists():
                print(f"  [fallback] {k}{model_suffix} missing for {dataset} "
                      f"-> using no-TENT baseline {fb}")
                d, p = fb, fbp
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run src.cross_dataset for {k}/{dataset}.")
        probs = np.load(p)
        probs_list.append(probs)
        if labels is None:
            labels = np.load(d / "labels.npy")
            bp = d / "fitz_buckets.npy"
            if bp.exists():
                buckets = np.load(bp, allow_pickle=True)
    return probs_list, labels, buckets


def auc_macro_ovr(probs: np.ndarray, labels: np.ndarray) -> float:
    """Macro OvR AUC. Handles cases where a class has 0 positives by dropping it."""
    try:
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Some class may have 0 positives — compute per-class manually and average present
        n_classes = probs.shape[1]
        aucs = []
        for c in range(n_classes):
            y = (labels == c).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            aucs.append(roc_auc_score(y, probs[:, c]))
        return float(np.mean(aucs)) if aucs else float("nan")


def auc_dark_mal(probs: np.ndarray, labels: np.ndarray, buckets: np.ndarray) -> float:
    """Malignant AUC on dark-skin subset (universal taxonomy: malignant=class 0)."""
    mask = (buckets == "dark")
    if mask.sum() == 0:
        return float("nan")
    y = (labels[mask] == 0).astype(int)  # malignant idx
    if y.sum() == 0 or y.sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y, probs[mask, 0]))


def light_dark_gap(probs: np.ndarray, labels: np.ndarray, buckets: np.ndarray) -> float:
    """Light AUC - Dark AUC on malignant class."""
    out = {}
    for grp in ["light", "dark"]:
        mask = (buckets == grp)
        if mask.sum() == 0:
            out[grp] = float("nan")
            continue
        y = (labels[mask] == 0).astype(int)
        if y.sum() == 0 or y.sum() == mask.sum():
            out[grp] = float("nan")
            continue
        out[grp] = float(roc_auc_score(y, probs[mask, 0]))
    return out["light"] - out["dark"]


def slsqp_weights(probs_list: list[np.ndarray], objective_fn) -> np.ndarray:
    """
    Multi-restart SLSQP over the simplex. objective_fn(weights) -> AUC (higher better).

    Restarts from uniform + each one-hot corner pulled toward uniform — mirrors
    src/ensemble.py's strategy. AUC is non-smooth, so a single restart from
    uniform almost always gets stuck on the plateau.
    """
    M = len(probs_list)
    bounds = [(0.01, 0.99)] * M
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    init_points = [
        np.ones(M) / M,  # uniform
        *(np.eye(M)[i] * 0.7 + np.ones(M) * 0.3 / M for i in range(M)),
    ]

    best_w, best_score = None, -np.inf
    for x0 in init_points:
        x0 = np.clip(x0, 0.01, 0.99); x0 = x0 / x0.sum()
        try:
            res = minimize(lambda w: -objective_fn(w / w.sum()), x0,
                           method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-7, "maxiter": 200})
            w = np.clip(res.x, 0.01, 0.99); w = w / w.sum()
            score = objective_fn(w)
            if score > best_score:
                best_score, best_w = score, w
        except Exception:
            continue
    return best_w if best_w is not None else (np.ones(M) / M)


def weighted_probs(probs_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Linear combination of probability arrays."""
    return np.sum([w * p for w, p in zip(weights, probs_list)], axis=0)


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse_dataset(dataset: str, model_keys: list[str], ckpt_base: Path,
                    ham10k_weights: np.ndarray,
                    probs_subdir: str = "cross_dataset",
                    model_suffix: str = "",
                   ) -> dict:
    """
    For one external dataset, return:
      - per-model standalone metrics
      - SLSQP weights optimising accuracy
      - SLSQP weights optimising dark-skin Mal AUC (if buckets available)
      - HAM10000-global weights applied here (for comparison)
    """
    label_mode = "ham10k" if dataset == "isic2019" else "universal"
    probs_list, labels, buckets = load_arrays(dataset, model_keys, label_mode, ckpt_base,
                                              probs_subdir=probs_subdir, model_suffix=model_suffix)

    # Per-model standalone
    standalone = {}
    for k, p in zip(model_keys, probs_list):
        d = {
            "macro_ovr_auc": auc_macro_ovr(p, labels),
        }
        if buckets is not None:
            d["dark_mal_auc"] = auc_dark_mal(p, labels, buckets)
            d["light_dark_gap"] = light_dark_gap(p, labels, buckets)
        standalone[k] = d

    # (a) Per-dataset SLSQP for macro OvR AUC
    obj_macro = lambda w: auc_macro_ovr(weighted_probs(probs_list, w), labels)
    w_macro = slsqp_weights(probs_list, obj_macro)
    macro_blend = weighted_probs(probs_list, w_macro)
    macro_result = {
        "weights": dict(zip(model_keys, [float(x) for x in w_macro])),
        "macro_ovr_auc": auc_macro_ovr(macro_blend, labels),
    }
    if buckets is not None:
        macro_result["dark_mal_auc"] = auc_dark_mal(macro_blend, labels, buckets)
        macro_result["light_dark_gap"] = light_dark_gap(macro_blend, labels, buckets)

    # (b) Per-dataset SLSQP for dark-skin Mal AUC (if buckets exist)
    fair_result = None
    if buckets is not None:
        obj_fair = lambda w: auc_dark_mal(weighted_probs(probs_list, w), labels, buckets)
        w_fair = slsqp_weights(probs_list, obj_fair)
        fair_blend = weighted_probs(probs_list, w_fair)
        fair_result = {
            "weights": dict(zip(model_keys, [float(x) for x in w_fair])),
            "macro_ovr_auc": auc_macro_ovr(fair_blend, labels),
            "dark_mal_auc": auc_dark_mal(fair_blend, labels, buckets),
            "light_dark_gap": light_dark_gap(fair_blend, labels, buckets),
        }

    # (c) HAM10000-global weights applied here
    if ham10k_weights is not None and label_mode == "universal":
        # HAM10000 weights are 7-class; for universal-3 we have to refit.
        global_result = None
    elif ham10k_weights is not None:
        global_blend = weighted_probs(probs_list, ham10k_weights)
        global_result = {
            "weights": dict(zip(model_keys, [float(x) for x in ham10k_weights])),
            "macro_ovr_auc": auc_macro_ovr(global_blend, labels),
        }
        if buckets is not None:
            global_result["dark_mal_auc"] = auc_dark_mal(global_blend, labels, buckets)
            global_result["light_dark_gap"] = light_dark_gap(global_blend, labels, buckets)
    else:
        global_result = None

    return {
        "dataset": dataset,
        "label_mode": label_mode,
        "n_samples": int(len(labels)),
        "standalone": standalone,
        "tuned_for_accuracy": macro_result,
        "tuned_for_fairness": fair_result,
        "ham10k_global_weights": global_result,
    }


def get_ham10k_weights(model_keys: list[str], ckpt_base: Path) -> np.ndarray | None:
    """
    Fit SLSQP ensemble weights on HAM10000 TTA test probs (mirrors src.ensemble logic).
    Returns weights in the same order as model_keys, or None if any TTA file is missing.
    """
    probs = []
    labels = None
    for k in model_keys:
        p = ckpt_base / k / "tta_test_probs.npy"
        if not p.exists():
            print(f"  [warn] {p} missing — skipping HAM10000-global comparison")
            return None
        probs.append(np.load(p))
        if labels is None:
            labels = np.load(ckpt_base / k / "tta_test_labels.npy")
    obj = lambda w: auc_macro_ovr(weighted_probs(probs, w), labels)
    return slsqp_weights(probs, obj)


def render_report(results: list[dict], model_keys: list[str], out_path: Path) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("Cross-dataset SLSQP ensemble (B4)")
    lines.append("=" * 80)
    lines.append(f"Models: {model_keys}")
    lines.append("")

    for r in results:
        ds = r["dataset"]
        lines.append("-" * 80)
        lines.append(f"  {ds.upper()}  (n={r['n_samples']}, label_mode={r['label_mode']})")
        lines.append("-" * 80)

        lines.append("Standalone per-model:")
        for k in model_keys:
            s = r["standalone"][k]
            line = f"  {k:30s}  macro_OvR={s['macro_ovr_auc']:.4f}"
            if "dark_mal_auc" in s:
                line += f"  dark_mal={s['dark_mal_auc']:.4f}  gap={s['light_dark_gap']:+.4f}"
            lines.append(line)
        lines.append("")

        lines.append("Tuned for accuracy (max macro OvR AUC on this dataset):")
        m = r["tuned_for_accuracy"]
        lines.append("  weights:")
        for k in model_keys:
            lines.append(f"    {k:30s}  {m['weights'][k]:.3f}")
        line = f"  macro_OvR={m['macro_ovr_auc']:.4f}"
        if "dark_mal_auc" in m:
            line += f"  dark_mal={m['dark_mal_auc']:.4f}  gap={m['light_dark_gap']:+.4f}"
        lines.append(line)
        lines.append("")

        if r["tuned_for_fairness"] is not None:
            f = r["tuned_for_fairness"]
            lines.append("Tuned for fairness (max dark-skin Mal AUC):")
            lines.append("  weights:")
            for k in model_keys:
                lines.append(f"    {k:30s}  {f['weights'][k]:.3f}")
            lines.append(f"  macro_OvR={f['macro_ovr_auc']:.4f}  dark_mal={f['dark_mal_auc']:.4f}  gap={f['light_dark_gap']:+.4f}")
            lines.append("")

        if r["ham10k_global_weights"] is not None:
            g = r["ham10k_global_weights"]
            lines.append("HAM10000-optimal weights applied here:")
            lines.append("  weights:")
            for k in model_keys:
                lines.append(f"    {k:30s}  {g['weights'][k]:.3f}")
            line = f"  macro_OvR={g['macro_ovr_auc']:.4f}"
            if "dark_mal_auc" in g:
                line += f"  dark_mal={g['dark_mal_auc']:.4f}  gap={g['light_dark_gap']:+.4f}"
            lines.append(line)
            lines.append("")
    lines.append("=" * 80)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=["isic2019", "ddi", "fitz17k"])
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--probs_subdir", default="cross_dataset",
                        choices=["cross_dataset", "cross_dataset_tent"],
                        help="Which probs to ensemble: baseline or TENT-adapted.")
    parser.add_argument("--model_suffix", default="",
                        help="Suffix on per-model dirs (e.g. '__lr1e-4' for TENT variants).")
    parser.add_argument("--out_suffix", default="",
                        help="Suffix on output report dir (e.g. '_tent_lr1e-4').")
    args = parser.parse_args()

    ckpt_base = Path(args.ckpt_base)

    print("Fitting HAM10000-global SLSQP weights for comparison...")
    ham_w = get_ham10k_weights(args.models, ckpt_base)
    if ham_w is not None:
        for k, w in zip(args.models, ham_w):
            print(f"  {k:30s}  {w:.3f}")

    results = []
    for ds in args.datasets:
        print(f"\n--- {ds} (probs_subdir={args.probs_subdir}, suffix='{args.model_suffix}') ---")
        r = analyse_dataset(ds, args.models, ckpt_base, ham_w,
                            probs_subdir=args.probs_subdir,
                            model_suffix=args.model_suffix)
        results.append(r)

    out_dir = ckpt_base / ("cross_dataset_ensemble" + args.out_suffix)
    out_path = out_dir / "report.txt"
    render_report(results, args.models, out_path)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({"models": args.models, "results": results,
                                     "probs_subdir": args.probs_subdir,
                                     "model_suffix": args.model_suffix}, indent=2),
                         encoding="utf-8")
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
