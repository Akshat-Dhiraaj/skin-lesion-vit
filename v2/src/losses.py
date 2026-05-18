"""
src/losses.py
-------------
Loss functions used outside the standard CrossEntropyLoss path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification (Lin et al., 2017).

    L = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else 1-p.

    Args:
        alpha: weight for the positive class (default 0.25 per Lin et al.)
        gamma: focusing parameter (default 2.0)
        reduction: 'mean' / 'sum' / 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,) or (B, 1)
        # targets: (B,) {0,1}
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)               # (B,1) -> (B,) single-logit head
        targets = targets.float()
        # per-sample BCE; reduction handled at the end after focal weighting
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        # p_t = model's predicted prob of the *true* class (p if y=1 else 1-p)
        p_t = p * targets + (1 - p) * (1 - targets)
        # α-balance term. ⚠️ TRAP (OA1): with a 50/50 balanced sampler you want
        # alpha ≈ 0.5. Lin et al.'s default alpha=0.25 *down-weights* positives
        # (correct for 1:1000 detection anchors, WRONG here) — using 0.25 here
        # collapsed the ConvNeXt mel head to 0 recall. See 02_MODELS_AND_TECHNIQUES.md.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # focal term (1-p_t)^gamma down-weights easy (already-confident) samples
        # so the gradient focuses on the hard, near-the-boundary ones.
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
