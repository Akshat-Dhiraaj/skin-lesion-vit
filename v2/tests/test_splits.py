"""
tests/test_splits.py
--------------------
CI tests for src/data/splits.py. Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    assert_no_group_leakage,
    report_split_stats,
    split_dataset,
)


@pytest.fixture
def synthetic_df():
    """50 patients, 1-5 lesions each, 1-3 images per lesion, 7 imbalanced classes."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(50):
        n_lesions = int(rng.integers(1, 6))
        for lidx in range(n_lesions):
            lesion_id = f"p{pid}_l{lidx}"
            n_imgs = int(rng.integers(1, 4))
            cls = int(rng.choice(7, p=[0.55, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]))
            for iidx in range(n_imgs):
                rows.append({
                    "image_id":   f"p{pid}_l{lidx}_i{iidx}",
                    "patient_id": f"p{pid}",
                    "lesion_id":  lesion_id,
                    "label":      cls,
                })
    return pd.DataFrame(rows)


def test_no_lesion_leak(synthetic_df):
    train, val, test = split_dataset(synthetic_df, "label", "lesion_id")
    assert_no_group_leakage(train, val, test, "lesion_id")


def test_no_patient_leak(synthetic_df):
    train, val, test = split_dataset(synthetic_df, "label", "patient_id")
    assert_no_group_leakage(train, val, test, "patient_id")


def test_split_sizes_in_tolerance(synthetic_df):
    train, val, test = split_dataset(
        synthetic_df, "label", "patient_id", val_frac=0.15, test_frac=0.15,
    )
    n = len(synthetic_df)
    # Group-aware splitting can't hit exact fractions; allow generous slack.
    assert 0.55 * n <= len(train) <= 0.85 * n
    assert 0.05 * n <= len(val)   <= 0.30 * n
    assert 0.05 * n <= len(test)  <= 0.30 * n


def test_all_classes_in_train(synthetic_df):
    train, val, test = split_dataset(synthetic_df, "label", "patient_id")
    assert set(train["label"].unique()) == set(synthetic_df["label"].unique())


def test_reproducibility(synthetic_df):
    a1, b1, c1 = split_dataset(synthetic_df, "label", "patient_id", random_state=42)
    a2, b2, c2 = split_dataset(synthetic_df, "label", "patient_id", random_state=42)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(c1, c2)


def test_different_seeds_differ(synthetic_df):
    a1, _, _ = split_dataset(synthetic_df, "label", "patient_id", random_state=42)
    a2, _, _ = split_dataset(synthetic_df, "label", "patient_id", random_state=7)
    assert not a1.equals(a2)


def test_invalid_fractions(synthetic_df):
    with pytest.raises(ValueError):
        split_dataset(synthetic_df, "label", "patient_id", val_frac=0.6, test_frac=0.6)
    with pytest.raises(ValueError):
        split_dataset(synthetic_df, "label", "patient_id", val_frac=0.0, test_frac=0.15)


def test_nan_in_group_col_rejected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "patient_id"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        split_dataset(df, "label", "patient_id")


def test_missing_column_rejected(synthetic_df):
    with pytest.raises(ValueError, match="not in DataFrame"):
        split_dataset(synthetic_df, "label", "nonexistent_col")


def test_report_stats_shape(synthetic_df):
    train, val, test = split_dataset(synthetic_df, "label", "patient_id")
    stats = report_split_stats(train, val, test, "label", "patient_id")
    assert len(stats) == 3
    assert {"split", "n_images", "n_groups"}.issubset(stats.columns)
    # Sum of per-class counts across splits == total per class
    for cls in synthetic_df["label"].unique():
        col = f"n_{cls}"
        assert stats[col].sum() == (synthetic_df["label"] == cls).sum()

# ── HAM10000 loader tests (added Phase 2) ─────────────────────────────────────

from src.data.ham10000 import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    HAM10000Dataset,
    get_weighted_sampler,
    NUM_CLASSES,
)


def test_class_registry_consistent():
    assert len(CLASS_NAMES) == NUM_CLASSES == 7
    assert all(CLASS_TO_IDX[c] == i for i, c in enumerate(CLASS_NAMES))


def test_ham10000_dataset_len():
    """Dataset length matches DataFrame length."""
    rng = np.random.default_rng(0)
    n = 20
    dummy_df = pd.DataFrame({
        "image_path": ["__dummy__"] * n,
        "label": rng.integers(0, 7, n),
    })
    ds = HAM10000Dataset(dummy_df, transform=None)
    assert len(ds) == n


def test_weighted_sampler_length():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": rng.integers(0, 7, 100)})
    sampler = get_weighted_sampler(df)
    assert sampler.num_samples == 100


# ── Registry tests (added Phase 4) ────────────────────────────────────────────
import torch
from src.models.registry import (
    REGISTRY,
    ModelConfig,
    build_model,
    build_optimizer,
    count_params,
)


def test_registry_keys_exist():
    expected = {
        "convnextv2_large", "dinov2_registers_probe", "dinov2_registers_ft",
        "pandem_probe", "pandem_ft", "siglip2_probe", "efficientnetv2_m",
        "pandem_probe_224", "pandem_ft_224",
        "pandem_real_probe", "pandem_real_ft",
    }
    assert expected.issubset(set(REGISTRY.keys()))


def test_registry_entries_are_model_configs():
    for key, cfg in REGISTRY.items():
        assert isinstance(cfg, ModelConfig), f"{key} is not a ModelConfig"


def test_registry_valid_model_types():
    valid = {"timm", "dinov2", "pandem", "pandem_real", "siglip2"}
    for key, cfg in REGISTRY.items():
        assert cfg.model_type in valid, f"{key} has invalid model_type: {cfg.model_type}"


def test_build_model_timm_tiny():
    """Build a tiny timm model (no large download) to test the timm path."""
    cfg = ModelConfig(
        name="test_resnet18",
        model_type="timm",
        model_id="resnet18",
        freeze_backbone=False,
        num_classes=7,
    )
    model = build_model(cfg)
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_build_model_timm_frozen():
    cfg = ModelConfig(
        name="test_resnet18_frozen",
        model_type="timm",
        model_id="resnet18",
        freeze_backbone=True,
        num_classes=7,
    )
    model = build_model(cfg)
    trainable, total = count_params(model)
    assert trainable < total  # some params frozen


def test_build_optimizer_discriminative_lr():
    cfg = ModelConfig(
        name="test_resnet18",
        model_type="timm",
        model_id="resnet18",
        freeze_backbone=False,
        lr_head=1e-3,
        lr_backbone=1e-5,
    )
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    lrs = {g["lr"] for g in opt.param_groups}
    assert 1e-3 in lrs
    assert 1e-5 in lrs


def test_unknown_registry_key_raises():
    import pytest
    with pytest.raises(KeyError):
        build_model("nonexistent_model")


def test_unknown_model_type_raises():
    cfg = ModelConfig(
        name="bad", model_type="unknown_type",
        model_id="x", freeze_backbone=False,
    )
    with pytest.raises(ValueError):
        build_model(cfg)


# ── Engine tests (added Phase 5) ──────────────────────────────────────────────

from src.engine import DEVICE, evaluate, train_one_epoch


class _TinyModel(torch.nn.Module):
    """Minimal model for engine tests — no downloads needed."""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, 7)
    def forward(self, x):
        return self.fc(x.flatten(1))


def _tiny_loader(n=32, img_size=32, batch=8):
    from torch.utils.data import DataLoader, TensorDataset
    imgs = torch.randn(n, 3, img_size, img_size)
    lbls = torch.randint(0, 7, (n,))
    return DataLoader(TensorDataset(imgs, lbls), batch_size=batch)


def test_evaluate_output_shape():
    model = _TinyModel().to(DEVICE)
    loader = _tiny_loader()
    result = evaluate(model, loader)
    assert set(result.keys()) == {"auc", "acc", "probs", "labels"}
    assert result["probs"].shape == (32, 7)
    assert result["labels"].shape == (32,)
    assert 0.0 <= result["auc"] <= 1.0
    assert 0.0 <= result["acc"] <= 1.0


def test_train_one_epoch_returns_float():
    model = _TinyModel().to(DEVICE)
    loader = _tiny_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_one_epoch(model, loader, opt, None, scaler, accum_steps=2)
    assert isinstance(loss, float)
    assert loss > 0