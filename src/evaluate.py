"""
evaluate.py
-----------
Evaluation utilities:
  - Per-class AUC, confusion matrix, classification report
  - GradCAM visualizations to show where the model looks
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.cuda.amp import autocast

from dataset import CLASS_NAMES


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference and collect all predictions and labels."""
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        with autocast():
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def print_metrics(all_probs, all_labels):
    """Print overall and per-class metrics."""
    preds = all_probs.argmax(axis=1)

    overall_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"\nOverall AUC (macro OvR): {overall_auc:.4f}")

    # Per-class AUC
    print("\nPer-class AUC:")
    for i, cls in enumerate(CLASS_NAMES):
        binary_labels = (all_labels == i).astype(int)
        cls_auc = roc_auc_score(binary_labels, all_probs[:, i])
        print(f"  {cls:8s}: {cls_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=CLASS_NAMES))


def plot_confusion_matrix(all_probs, all_labels, save_path=None):
    """Plot normalized confusion matrix."""
    preds = all_probs.argmax(axis=1)
    cm = confusion_matrix(all_labels, preds, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


def plot_roc_curves(all_probs, all_labels, save_path=None):
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(CLASS_NAMES):
        binary_labels = (all_labels == i).astype(int)
        RocCurveDisplay.from_predictions(binary_labels, all_probs[:, i], name=cls, ax=ax)

    ax.set_title('ROC Curves — One vs Rest')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_gradcam(model, images, labels, device, num_samples=4, save_path=None):
    """
    GradCAM visualization — shows which pixels influenced the model's decision.
    Critical for medical imaging: confirms the model looks at the lesion, not artifacts.
    """
    # For ViT, target the last attention layer
    # For ResNet, target the last conv layer
    try:
        target_layer = model.vit.encoder.layer[-1].layernorm_before
    except AttributeError:
        target_layer = model.layer4[-1]  # ResNet fallback

    cam = GradCAM(model=model, target_layers=[target_layer])

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))

    for i in range(min(num_samples, len(images))):
        image = images[i].unsqueeze(0).to(device)
        label = labels[i]

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=image, targets=targets)[0]

        # Denormalize image for display
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Original: {CLASS_NAMES[label]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title('GradCAM')
        axes[i, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()