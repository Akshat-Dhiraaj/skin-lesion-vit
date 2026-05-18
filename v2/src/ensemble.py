"""
src/ensemble.py
---------------
Weighted ensemble of TTA-augmented model probabilities.

Reads tta_probs.npy from each model's experiment dir, runs a grid search
(or per-class SLSQP optimisation) over ensemble weights, and reports
the optimal combination.

Usage:
    # Simple 2-model grid search
    python -m src.ensemble --models convnextv2_large efficientnetv2_m

    # 4-model grid search (slower but more thorough)
    python -m src.ensemble \
        --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft

    # Per-class SLSQP weights (best results)
    python -m src.ensemble \
        --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_ft \
        --method slsqp
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import classification_report, roc_auc_score

from src.data.ham10000 import CLASS_NAMES
from src.models.registry import REGISTRY


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_tta_probs(model_keys: list[str], ckpt_base: str) -> tuple[list, np.ndarray]:
    """Load tta_probs.npy and labels for each model."""
    probs_list = []
    labels = None

    for key in model_keys:
        cfg = REGISTRY[key]
        ckpt_dir = Path(ckpt_base) / cfg.name
        probs_path = ckpt_dir / "tta_probs.npy"
        labels_path = ckpt_dir / "tta_labels.npy"

        if not probs_path.exists():
            raise FileNotFoundError(
                f"TTA probs not found: {probs_path}\n"
                f"Run TTA first: python -m src.tta --model {key}"
            )

        probs = np.load(probs_path)
        these_labels = np.load(labels_path)

        # Sanity check: labels must match across models (same test set)
        if labels is None:
            labels = these_labels
        else:
            assert np.array_equal(labels, these_labels), \
                f"Labels mismatch between models — different test split?"

        probs_list.append(probs)
        print(f"  Loaded {cfg.name}: shape={probs.shape}")

    return probs_list, labels


# ── Methods ───────────────────────────────────────────────────────────────────

def grid_search(probs_list: list[np.ndarray], labels: np.ndarray, step: float = 0.05) -> dict:
    """
    Exhaustive grid search over weight combinations that sum to 1.
    Tractable for ≤4 models with step=0.05.
    """
    n_models = len(probs_list)

    # Generate all weight combinations summing to 1
    # Discretise: each weight in {0, step, 2*step, ..., 1}
    grid_points = np.arange(0, 1 + step / 2, step)

    print(f"\nGrid search over {n_models} models (step={step})...")
    print(f"Total combinations to test: ~{len(grid_points) ** n_models}")

    best_auc = 0.0
    best_weights = None
    best_probs = None

    # Smart enumeration: weights must sum to 1
    for weights in itertools.product(grid_points, repeat=n_models):
        if abs(sum(weights) - 1.0) > 1e-6:
            continue
        if any(w < 0.05 for w in weights):  # Skip degenerate cases
            continue

        ensemble = sum(w * p for w, p in zip(weights, probs_list))
        auc = roc_auc_score(labels, ensemble, multi_class="ovr", average="macro")

        if auc > best_auc:
            best_auc = auc
            best_weights = weights
            best_probs = ensemble

    return {
        "weights": best_weights,
        "auc": best_auc,
        "probs": best_probs,
        "method": "grid_search",
    }


def slsqp_search(probs_list: list[np.ndarray], labels: np.ndarray) -> dict:
    """
    Continuous optimisation over weights using SLSQP.
    Faster than grid search for >4 models, may find better optima.
    """
    n_models = len(probs_list)

    def neg_auc(weights):
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()  # Normalise
        ensemble = sum(w * p for w, p in zip(weights, probs_list))
        return -roc_auc_score(labels, ensemble, multi_class="ovr", average="macro")

    print(f"\nSLSQP optimisation over {n_models} models...")

    # Multi-restart from different init points
    best_result = None
    best_auc = 0.0

    init_points = [
        np.ones(n_models) / n_models,  # uniform
        *(np.eye(n_models)[i] * 0.7 + np.ones(n_models) * 0.3 / n_models for i in range(n_models)),
    ]

    for x0 in init_points:
        try:
            result = minimize(
                neg_auc,
                x0=x0,
                method="SLSQP",
                bounds=[(0.01, 0.99)] * n_models,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                options={"maxiter": 100, "ftol": 1e-6},
            )
            auc = -result.fun
            if auc > best_auc:
                best_auc = auc
                best_result = result
        except Exception as e:
            print(f"  Restart failed: {e}")

    weights = best_result.x / best_result.x.sum()
    ensemble = sum(w * p for w, p in zip(weights, probs_list))

    return {
        "weights": tuple(weights),
        "auc": best_auc,
        "probs": ensemble,
        "method": "slsqp",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Registry keys for models to ensemble")
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument("--method", choices=["grid", "slsqp"], default="grid",
                        help="grid: exhaustive (≤4 models). slsqp: continuous optim.")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Grid search step size")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE — {len(args.models)} models")
    print(f"{'=' * 60}")
    print(f"Models: {args.models}")
    print(f"Method: {args.method}")

    # Load TTA probs
    probs_list, labels = load_tta_probs(args.models, args.ckpt_base)

    # Sanity check: individual AUCs
    print("\nIndividual TTA AUCs:")
    for key, probs in zip(args.models, probs_list):
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        print(f"  {key:30s}: {auc:.4f}")

    # Find optimal weights
    if args.method == "grid":
        result = grid_search(probs_list, labels, step=args.step)
    else:
        result = slsqp_search(probs_list, labels)

    # Report
    weights = result["weights"]
    best_auc = result["auc"]
    best_probs = result["probs"]
    best_acc = float((best_probs.argmax(1) == labels).mean())

    print(f"\n{'─' * 60}")
    print(f"Best ensemble weights ({result['method']}):")
    for key, w in zip(args.models, weights):
        print(f"  {key:30s}: {w:.3f}")
    print(f"{'─' * 60}")
    print(f"Ensemble Test AUC: {best_auc:.4f}")
    print(f"Ensemble Test Acc: {best_acc:.4f}")

    # Compare to best individual
    individual_aucs = [roc_auc_score(labels, p, multi_class="ovr", average="macro") for p in probs_list]
    best_individual = max(individual_aucs)
    print(f"\nBest individual : {best_individual:.4f}")
    print(f"Ensemble lift   : {best_auc - best_individual:+.4f}")

    # Per-class report
    print("\nPer-class report:")
    print(classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES))

    # Save results
    ens_dir = Path(args.ckpt_base) / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    np.save(ens_dir / "ensemble_probs.npy", best_probs)
    np.save(ens_dir / "ensemble_labels.npy", labels)

    result_path = ens_dir / "results.txt"
    weight_lines = "\n".join(
        f"  {key:30s}: {w:.3f}" for key, w in zip(args.models, weights)
    )
    result_path.write_text(
        f"Ensemble — {len(args.models)} models ({result['method']})\n\n"
        f"Models and weights:\n{weight_lines}\n\n"
        f"Test AUC : {best_auc:.4f}\n"
        f"Test Acc : {best_acc:.4f}\n"
        f"Best individual : {best_individual:.4f}\n"
        f"Ensemble lift   : {best_auc - best_individual:+.4f}\n\n"
        f"{classification_report(labels, best_probs.argmax(1), target_names=CLASS_NAMES)}",
        encoding="utf-8",
    )
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()