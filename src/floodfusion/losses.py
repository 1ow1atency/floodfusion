"""Cross-entropy + soft Dice on the water class; pixels with target -1 are ignored by both.

* CE: mean over valid pixels (``ignore_index=-1``). If a batch has no valid pixel the
  CE term is 0 (PyTorch would return NaN for 0/0).
* Dice: soft Dice of the water probability over all valid pixels of the batch, pooled
  (not per image, which is unstable for chips without water):
      dice = (2 * sum(p * t) + smooth) / (sum(p) + sum(t) + smooth),  loss = 1 - dice
  with p = softmax(logits)[:, 1] and t = (target == 1), both multiplied by the valid
  mask, so ignored pixels contribute nothing and get zero gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chips import IGNORE_INDEX


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    valid = (target != IGNORE_INDEX).float()
    p = torch.softmax(logits.float(), dim=1)[:, 1] * valid
    t = (target == 1).float() * valid
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not (target != IGNORE_INDEX).any():
        return logits.sum() * 0.0  # keeps the graph, no NaN
    return F.cross_entropy(logits.float(), target, ignore_index=IGNORE_INDEX)


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.ce_weight, self.dice_weight, self.smooth = ce_weight, dice_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4 or logits.shape[1] != 2:
            raise ValueError(f"logits must be (N, 2, H, W), got {tuple(logits.shape)}")
        if target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(f"target {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
        return (self.ce_weight * masked_cross_entropy(logits, target)
                + self.dice_weight * soft_dice_loss(logits, target, self.smooth))
