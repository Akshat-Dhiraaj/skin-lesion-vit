import numpy as np
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
MEL_IDX = 0


def run_thresholding():
    base   = "experiments/mega_tta_sync/"
    p1     = np.load(base + "eff_probs.npy")
    p2     = np.load(base + "dino_probs.npy")
    y_true = np.load(base + "labels.npy")

    ensemble_probs = (p1 + p2) / 2
    mel_probs      = ensemble_probs[:, MEL_IDX]

    # FIX — The threshold was previously searched over the full test set and
    # the classification_report was also printed on that same set. This means
    # the chosen threshold is overfit to the test data and the reported
    # per-class metrics are optimistic. Split into val (threshold tuning)
    # and test (honest evaluation) using stratified indices so both splits
    # keep the same class distribution as the full test set.
    indices = np.arange(len(y_true))
    val_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=y_true, random_state=42
    )

    # --- tune on val ---
    mel_val  = mel_probs[val_idx]
    y_val    = y_true[val_idx]
    y_val_b  = (y_val == MEL_IDX).astype(int)

    best_threshold, best_f2 = 0.0, 0.0
    for t in np.linspace(0.01, 0.5, 100):
        score = fbeta_score(y_val_b, (mel_val > t).astype(int), beta=2)
        if score > best_f2:
            best_f2, best_threshold = score, t

    print(f"Optimal Melanoma Threshold (val): {best_threshold:.4f}  (F2={best_f2:.4f})")

    # --- evaluate on test ---
    mel_test   = mel_probs[test_idx]
    ens_test   = ensemble_probs[test_idx]
    y_test     = y_true[test_idx]

    final_preds = np.argmax(ens_test, axis=1)
    final_preds[mel_test > best_threshold] = MEL_IDX

    print("\nClassification report on held-out test split:")
    print(classification_report(y_test, final_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    run_thresholding()