"""Weighted cross-entropy + Dice loss, one explicit ignore index (plan section 4.3/4.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.sarfuseseg.config import CE_LOSS_WEIGHT, DICE_LOSS_WEIGHT, IGNORE_INDEX, N_CLASSES


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, n_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """Multi-class soft Dice loss, ignoring pixels equal to ``ignore_index``."""
    probs = F.softmax(logits, dim=1)
    valid = (target != ignore_index).unsqueeze(1)  # (B,1,H,W)
    target_clamped = target.clone()
    target_clamped[target == ignore_index] = 0
    target_onehot = F.one_hot(target_clamped, num_classes=n_classes).permute(0, 3, 1, 2).float()

    probs = probs * valid
    target_onehot = target_onehot * valid

    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dim=dims)
    union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float] | None = None,
        ignore_index: int = IGNORE_INDEX,
        n_classes: int = N_CLASSES,
        ce_weight: float = CE_LOSS_WEIGHT,
        dice_weight: float = DICE_LOSS_WEIGHT,
    ):
        super().__init__()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = dice_loss(logits, target, self.ignore_index, self.n_classes)
        return self.ce_weight * ce + self.dice_weight * dice
