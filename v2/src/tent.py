"""
src/tent.py
-----------
TENT: Fully Test-Time Adaptation by Entropy Minimization
(Wang et al., ICLR 2021).

Adapts the affine parameters of normalisation layers (BatchNorm /
LayerNorm) at test time by minimising the prediction entropy on each
batch. No labels, no training set. Designed for domain shift.

Apply this to a trained HAM10000 model when evaluating on Fitz17k / DDI.

Usage:
    from src.tent import wrap_for_tent, tent_evaluate
    adapted_model = wrap_for_tent(model)
    probs, labels = tent_evaluate(adapted_model, loader)
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tqdm import tqdm

NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)


def wrap_for_tent(model: nn.Module, copy_first: bool = True) -> nn.Module:
    """
    Freeze everything except normalisation-layer affines.

    For BatchNorm: also set track_running_stats=False so each test batch
    estimates its own statistics. For LayerNorm: only affine params adapt.
    """
    if copy_first:
        model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, NORM_TYPES):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution per sample. Shape: (B,)."""
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)


def tent_evaluate(
    model: nn.Module,
    loader,
    lr: float = 1e-3,
    n_steps: int = 1,
    device: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run TENT on the loader. Each batch is used for n_steps adaptation
    steps, then a final forward yields the adapted prediction.

    Args:
        model:    wrap_for_tent-wrapped model.
        loader:   yields (image, label) batches.
        lr:       learning rate for Adam over the BN/LN affines.
        n_steps:  number of adaptation steps per batch (default 1; more
                  can overfit to a particular batch).
        device:   target device. Default: cuda if available.

    Returns:
        probs:    (N, n_classes) softmax probabilities AFTER adaptation.
        labels:   (N,) ground-truth labels.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Keep BN/LN in train mode so they update affines under autograd
    model.train()
    # But silence dropout & sample-dependent ops where possible
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable params after wrap_for_tent — model has no normalisation layers?")
    optimizer = torch.optim.Adam(params, lr=lr)

    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc="TENT", leave=False):
        x = x.to(device)
        for _ in range(n_steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = softmax_entropy(logits).mean()
            loss.backward()
            optimizer.step()
        # Final forward (no grad) — read off the adapted prediction
        with torch.no_grad():
            logits = model(x)
            probs = logits.softmax(dim=-1)
        all_probs.append(probs.detach().cpu())
        all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)
