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


def get_tta_transforms(image_size: int = 224, model_type: str | None = None) -> list[A.Compose]:
    """
    8 deterministic TTA views: original + 3 rotations + 3 flips + zoom.
    Average softmax probabilities across all views at inference time.
    """
    mean, std = get_norm_stats(model_type)
    norm = A.Normalize(mean=mean, std=std)
    resize = A.Resize(image_size, image_size)
    tail = [norm, ToTensorV2()]
    return [
        # 1. Original
        A.Compose([resize] + tail),
        # 2. H-flip
        A.Compose([resize, A.HorizontalFlip(p=1.0)] + tail),
        # 3. V-flip
        A.Compose([resize, A.VerticalFlip(p=1.0)] + tail),
        # 4. Rotate 90
        A.Compose([resize, A.Rotate(limit=(90, 90), p=1.0)] + tail),
        # 5. Rotate 180
        A.Compose([resize, A.Rotate(limit=(180, 180), p=1.0)] + tail),
        # 6. Rotate 270
        A.Compose([resize, A.Rotate(limit=(270, 270), p=1.0)] + tail),
        # 7. H-flip + V-flip
        A.Compose([resize, A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + tail),
        # 8. Slight zoom
        A.Compose([resize, A.Affine(scale=(0.9, 1.1), p=1.0)] + tail),
    ]