"""Joint augmentation for Sen1Floods11 sample dicts ({"id", "s1", "s2", "label"}).

Training: random crop, random horizontal/vertical flips and a random multiple of 90
degrees rotation, applied identically to every image tensor and the label (the same
crop window and the same geometric op). Pure geometry: values are never changed, so
normalisation and the -1 ignore label are unaffected. Randomness comes from torch's RNG,
which is seeded by seed_everything and per DataLoader worker by PyTorch.

Validation/test: no transform (full 512 x 512 chips).
"""

from __future__ import annotations

import torch

DEFAULT_CROP = 256


def _spatial_keys(sample: dict) -> list[str]:
    return [k for k, v in sample.items() if isinstance(v, torch.Tensor) and v.dim() >= 2]


class TrainTransform:
    def __init__(self, crop_size: int | None = DEFAULT_CROP, flips: bool = True, rot90: bool = True):
        if crop_size is not None and crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {crop_size}")
        self.crop_size = crop_size
        self.flips = flips
        self.rot90 = rot90

    def __call__(self, sample: dict) -> dict:
        keys = _spatial_keys(sample)
        if "label" not in keys:
            raise KeyError("sample needs a 'label' tensor")
        h, w = sample["label"].shape[-2:]
        for k in keys:
            if sample[k].shape[-2:] != (h, w):
                raise ValueError(f"{k} has spatial shape {tuple(sample[k].shape[-2:])}, label has {(h, w)}")
        out = dict(sample)

        if self.crop_size is not None:
            c = self.crop_size
            if c > h or c > w:
                raise ValueError(f"crop_size {c} larger than chip {(h, w)}")
            top = int(torch.randint(0, h - c + 1, (1,)))
            left = int(torch.randint(0, w - c + 1, (1,)))
            for k in keys:
                out[k] = out[k][..., top : top + c, left : left + c]

        ops = []
        if self.flips and torch.rand(()) < 0.5:
            ops.append(lambda x: torch.flip(x, dims=(-1,)))
        if self.flips and torch.rand(()) < 0.5:
            ops.append(lambda x: torch.flip(x, dims=(-2,)))
        k_rot = int(torch.randint(0, 4, (1,))) if self.rot90 else 0
        if k_rot:
            ops.append(lambda x: torch.rot90(x, k_rot, dims=(-2, -1)))
        for k in keys:
            x = out[k]
            for op in ops:
                x = op(x)
            out[k] = x.contiguous()
        return out
