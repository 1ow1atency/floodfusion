"""Segmentation models.

ImageNet weights with != 3 input channels (smp ``patch_first_conv``, smp 0.6): only the
first convolution changes. Input channel i gets a copy of the pretrained RGB filter
``i % 3`` (R, G, B, R, G, ...), and the whole kernel is scaled by 3 / C so the first
layer's output keeps roughly the magnitude it had on RGB. 2 channels: VV <- R, VH <- G,
x1.5. 13 or 15 channels: the RGB filters repeat across bands, x3/13 or x3/15. Everything
after the first convolution keeps its ImageNet weights unchanged.
"""

from __future__ import annotations

import torch.nn as nn

NUM_CLASSES = 2  # 0 = not water, 1 = water; outputs logits (N, 2, H, W)


def build_unet(in_channels: int, encoder: str = "resnet34", encoder_weights: str | None = "imagenet") -> nn.Module:
    """U-Net with ``in_channels`` inputs and 2 output logits. ``encoder_weights=None``
    gives random initialisation (used in tests, no download)."""
    import segmentation_models_pytorch as smp  # heavy import, only when a model is built

    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=NUM_CLASSES)
