"""
model.py
--------
ViT-Base fine-tuning setup for skin lesion classification.
Supports partial freezing (train only last N transformer blocks + head).
Also includes ResNet50 baseline for comparison.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification
import timm

NUM_CLASSES = 7


def get_vit_model(num_classes=NUM_CLASSES, freeze_until_block=8):
    """
    Load ViT-Base-patch16-224 pretrained on ImageNet-21k.
    Freezes all layers except the last (12 - freeze_until_block) transformer blocks + head.

    Args:
        num_classes      : number of output classes (7 for HAM10000)
        freeze_until_block: freeze blocks 0..freeze_until_block-1, train the rest

    Returns:
        model ready for fine-tuning
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # replaces the pretrained head
    )

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N transformer blocks
    for i in range(freeze_until_block, 12):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True

    # Always unfreeze the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ViT-Base: {trainable:,} / {total:,} parameters trainable ({trainable/total*100:.1f}%)")

    return model


def get_resnet_baseline(num_classes=NUM_CLASSES, freeze_backbone=True):
    """
    ResNet50 baseline.
    freeze_backbone=True  → only train the final FC layer (feature extractor mode)
    freeze_backbone=False → train all layers (full fine-tuning)
    """
    model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:  # freeze everything except final layer
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet50: {trainable:,} / {total:,} parameters trainable ({trainable/total*100:.1f}%)")

    return model


def get_optimizer(model, lr_head=1e-3, lr_backbone=1e-5):
    """
    Discriminative learning rates:
    - Higher LR for the classification head (randomly initialized)
    - Lower LR for pretrained backbone layers (already good features)
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=1e-4)

    return optimizer


def get_scheduler(optimizer, num_epochs, steps_per_epoch):
    """
    OneCycleLR — best scheduler for fine-tuning transformers.
    Warms up then decays, avoids destroying pretrained weights early.
    """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-3],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,  # 10% warmup
    )
    return scheduler