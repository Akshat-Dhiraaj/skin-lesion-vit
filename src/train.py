"""
train.py
--------
Training loop for skin lesion classification.
Features:
  - Mixed precision (AMP) for 8GB VRAM
  - Gradient accumulation for effective larger batch size
  - MLflow experiment tracking
  - Checkpoint saving (best val AUC)
"""

import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

from dataset import load_metadata, split_dataset, get_dataloaders
from transforms import get_train_transform, get_val_transform
from model import get_vit_model, get_optimizer, get_scheduler

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir'          : 'data/raw',
    'checkpoint_dir'    : 'checkpoints',
    'batch_size'        : 16,        # real batch size (VRAM safe for ViT on 8GB)
    'accum_steps'       : 4,         # effective batch = 16 * 4 = 64
    'num_epochs'        : 20,
    'freeze_until_block': 8,         # train last 4 transformer blocks + head
    'num_classes'       : 7,
    'device'            : 'cuda' if torch.cuda.is_available() else 'cpu',
    'experiment_name'   : 'skin-lesion-vit',
}
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc='Train')):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = nn.CrossEntropyLoss()(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Eval'):
        images = images.to(device)
        with autocast():
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # One-vs-rest AUC for multiclass
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    acc = (all_probs.argmax(axis=1) == all_labels).mean()

    return {'auc': auc, 'acc': acc}


def train():
    device = CONFIG['device']
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    df = load_metadata(CONFIG['data_dir'])
    train_df, val_df, test_df = split_dataset(df)
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df,
        get_train_transform(), get_val_transform(),
        batch_size=CONFIG['batch_size']
    )

    # Model
    model = get_vit_model(
        num_classes=CONFIG['num_classes'],
        freeze_until_block=CONFIG['freeze_until_block']
    ).to(device)

    # Enable gradient checkpointing to save VRAM
    model.vit.encoder.gradient_checkpointing = True

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, CONFIG['num_epochs'], len(train_loader) // CONFIG['accum_steps'])
    scaler    = GradScaler()

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    best_auc = 0.0

    mlflow.set_experiment(CONFIG['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, CONFIG['accum_steps']
            )
            val_metrics = evaluate(model, val_loader, device, CONFIG['num_classes'])

            print(f"Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['acc']:.4f}")

            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_auc'   : val_metrics['auc'],
                'val_acc'   : val_metrics['acc'],
            }, step=epoch)

            # Save best model
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                ckpt_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ New best AUC: {best_auc:.4f} — saved to {ckpt_path}")

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    train()