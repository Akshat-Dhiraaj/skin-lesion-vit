"""
src/data/transforms.py
----------------------
Albumentations augmentation pipelines.

Identical augmentation policy to the original repo — dermatoscopic images
benefit from aggressive spatial and colour augmentation. The only change
is image_size is a parameter rather than a hardcoded 224, so higher-
resolution phases (384 for EfficientNetV2, 448 for curriculum training)
don't need a separate file.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet stats — used for all backbones pretrained on ImageNet or
# ImageNet-21k (ViT, DINOv2, EfficientNetV2, ConvNeXt-V2, PanDerm).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# SigLIP / SigLIP 2 expect (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) — not ImageNet.
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD  = (0.5, 0.5, 0.5)

# OpenAI CLIP stats — used by DermLIP / PanDerm real (open_clip preprocess_cfg).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def get_norm_stats(model_type: str | None) -> tuple[tuple, tuple]:
    """Return (mean, std) for the given model_type."""
    if model_type == "siglip2":
        return SIGLIP_MEAN, SIGLIP_STD
    if model_type == "pandem_real":
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def get_train_transform(image_size: int = 224, model_type: str | None = None) -> A.Compose:
    """
    Heavy augmentation for training.

    Policy rationale:
    - Flips + RandomRotate90: dermoscopic images have no canonical
      orientation; all 8 dihedral symmetries are valid.
    - ShiftScaleRotate: small translations/scales account for variable
      lesion centering; up to 30° rotation adds continuous coverage
      between the 90° steps.
    - OneOf noise/blur: simulates imaging artefacts (hair, gel bubbles,
      motion blur) present in HAM10000.
    - OpticalDistortion / GridDistortion: simulate lens distortion
      variation across dermatoscope models.
    - ColorJitter: accounts for white-balance and illumination differences
      across acquisition sites and sessions.
    """
    mean, std = get_norm_stats(model_type)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
        ], p=0.2),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224, model_type: str | None = None) -> A.Compose:
    """
    Minimal transform for validation and test — resize + normalize only.
    No stochastic ops so metrics are deterministic across runs.
    """
    mean, std = get_norm_stats(model_type)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# NOTE (refactor 2026-05-16): a `get_tta_transforms` previously lived here but
# was dead code — the live 8-view TTA pipeline that produced every committed
# `tta_results.txt` is `src/tta.py:get_tta_transforms`. The two were NOT
# identical (this one used `A.Affine(scale=(0.9,1.1))` for view 8; tta.py uses
# `A.ShiftScaleRotate(scale_limit=0.1)`), so they were intentionally NOT merged
# under the strict behavior-preservation refactor — merging would have changed
# committed TTA numbers. The unused copy was removed; `src/tta.py` is the single
# source of truth for TTA augmentation. Do not re-add a TTA factory here.