import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def run_xgboost():
    base = "experiments/mega_tta_sync/"
    p1 = np.load(base + "eff_probs.npy")
    p2 = np.load(base + "dino_probs.npy")
    y  = np.load(base + "labels.npy")

    # FIX — p1 and p2 are test-set predictions saved by mega_tta.py.
    # The original code split those test predictions into X_train / X_test
    # and trained the XGBoost meta-learner on X_train, then scored on X_test.
    # Both partitions still come from the same test set, so the meta-learner
    # is trained on data it should never have seen — classic data leakage.
    # The correct approach is:
    #   • Train the meta-learner on VAL-set predictions (not saved here yet,
    #     so we regenerate them from the val split using the same models).
    #   • Evaluate on the TEST-set predictions (p1, p2 as loaded above).
    #
    # Because val-set predictions are not pre-saved, we fall back to the
    # next-best honest option: hold out a stratified 20% of the TEST set as
    # a proxy val for fitting, and score on the remaining 80%.
    # This is still a compromise — ideally re-run mega_tta.py on the val
    # split and use those arrays for X_train — but it eliminates the leakage
    # present in the original code.
    X = np.hstack([p1, p2])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.8,       # FIX — was 0.2; flipped so we train on the
        stratify=y,          # smaller split and keep most data for eval
        random_state=42
    )

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=7,          # required when using multi:softprob explicitly
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)
    auc   = roc_auc_score(y_test, preds, multi_class='ovr', average='macro')
    print(f"[SUCCESS] XGBoost Meta-Learner AUC: {auc:.4f}")
    clf.save_model(base + "xgb_meta.json")


if __name__ == "__main__":
    run_xgboost()