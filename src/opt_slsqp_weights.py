import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def objective(weights, probs, labels):
    weights = weights.reshape(len(probs), -1)
    ensemble_probs = np.zeros_like(probs[0])
    for i in range(len(probs)):
        ensemble_probs += probs[i] * weights[i]
    return -roc_auc_score(labels, ensemble_probs, multi_class='ovr', average='macro')


def run_slsqp():
    base = "experiments/mega_tta_sync/"
    p1     = np.load(base + "eff_probs.npy")
    p2     = np.load(base + "dino_probs.npy")
    labels = np.load(base + "labels.npy")

    # FIX — Per-class weights were optimized directly on the full test-set
    # labels and the reported AUC was measured on that same set, which
    # inflates the number (the optimizer literally fits to the answers).
    # Weights must be tuned on a held-out validation partition and then
    # evaluated on a separate test partition.
    indices = np.arange(len(labels))
    val_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=42
    )

    probs_val  = [p1[val_idx],  p2[val_idx]]
    probs_test = [p1[test_idx], p2[test_idx]]
    y_val      = labels[val_idx]
    y_test     = labels[test_idx]

    num_models  = 2
    num_classes = 7

    initial_weights = np.ones((num_models, num_classes)) / num_models
    bounds = [(0, 1)] * (num_models * num_classes)

    cons = [
        {'type': 'eq',
         'fun': lambda w, c=c: np.sum(w.reshape(num_models, num_classes)[:, c]) - 1}
        for c in range(num_classes)
    ]

    # Optimize on val split only
    res = minimize(
        objective, initial_weights.flatten(),
        args=(probs_val, y_val),
        method='SLSQP', bounds=bounds, constraints=cons
    )

    opt_w = res.x.reshape(num_models, num_classes)

    # Report on the held-out test split
    ens_test = sum(probs_test[i] * opt_w[i] for i in range(num_models))
    test_auc = roc_auc_score(y_test, ens_test, multi_class='ovr', average='macro')

    print(f"SLSQP Val AUC  : {-res.fun:.4f}")
    print(f"SLSQP Test AUC : {test_auc:.4f}")
    np.save(base + "slsqp_weights.npy", res.x)


if __name__ == "__main__":
    run_slsqp()