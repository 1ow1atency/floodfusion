"""Joint augmentation for Sen1Floods11 sample dicts ({"id", "s1", "s2", "label"}).

Training: random crop, random horizontal/vertical flips and a random multiple of 90
degrees rotation, applied identically to every image tensor and the label (the same
crop window and the same geometric op). Pure geometry: values are never changed, so
normalisation and the -1 ignore label are unaffected. Randomness comes from torch's RNG,
which is seeded by seed_everything and per DataLoader worker by PyTorch.

Validation/test: no transform (full 512 x 512 chips).

Robustness augmentation (training only, off unless enabled in the config's ``augment``):
  * SyntheticClouds: the dataset's ``s2_augment`` hook; adds synthetic clouds to the raw
    S2 (DN) before clipping and normalisation (floodfusion.clouds_synth).
  * ModalityDropout: after normalisation, drops S1 or S2 (never both) for the whole sample
    by setting it to 0 (= the train-split band mean) and records ``modality_mask``
    (S1 present, S2 present). FusionNet uses the mask; the U-Net simply sees the zeros.
Both draw from torch's RNG, like the geometric transform, so runs (and resumes) are
reproducible from the seed.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from .clouds_synth import THIN_P, cloudify

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


class Compose:
    def __init__(self, transforms: Sequence[Callable[[dict], dict]]):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class SyntheticClouds:
    """With probability ``p``, clouds over a coverage drawn uniformly from ``coverage``
    (min, max). Called by the dataset on raw S2 (13, H, W) in DN: ``f(s2, chip_id)``."""

    def __init__(self, p: float, coverage: Sequence[float], thin_p: float = THIN_P):
        lo, hi = (float(c) for c in coverage)
        if not 0.0 <= p <= 1.0 or not 0.0 <= lo <= hi <= 1.0 or not 0.0 <= thin_p <= 1.0:
            raise ValueError(f"need 0 <= p, thin_p <= 1 and 0 <= coverage min <= max <= 1, "
                             f"got p={p}, coverage={coverage}, thin_p={thin_p}")
        self.p, self.coverage, self.thin_p = float(p), (lo, hi), float(thin_p)

    def __call__(self, s2: np.ndarray, chip_id: str | None = None) -> np.ndarray:
        if float(torch.rand(())) >= self.p:
            return s2
        lo, hi = self.coverage
        coverage = lo + (hi - lo) * float(torch.rand(()))
        seed = int(torch.randint(0, 2**31 - 1, ()))
        return cloudify(s2, coverage, seed, thin_p=self.thin_p)


class ModalityDropout:
    """Drop S2 with probability ``p_drop_s2`` or S1 with ``p_drop_s1`` (one uniform draw,
    so never both), by zeroing it; adds ``modality_mask`` = (S1 present, S2 present)."""

    def __init__(self, p_drop_s1: float, p_drop_s2: float):
        if min(p_drop_s1, p_drop_s2) < 0 or p_drop_s1 + p_drop_s2 > 1:
            raise ValueError(f"need p_drop_s1, p_drop_s2 >= 0 with sum <= 1, got {p_drop_s1}, {p_drop_s2}")
        self.p_drop_s1, self.p_drop_s2 = float(p_drop_s1), float(p_drop_s2)

    def __call__(self, sample: dict) -> dict:
        if "s1" not in sample or "s2" not in sample:
            raise KeyError("modality dropout needs both s1 and s2 in the sample")
        u = float(torch.rand(()))
        drop = "s2" if u < self.p_drop_s2 else "s1" if u < self.p_drop_s2 + self.p_drop_s1 else None
        out = dict(sample)
        if drop is not None:
            out[drop] = torch.zeros_like(sample[drop])  # 0 = train band mean after normalisation
        out["modality_mask"] = torch.tensor([drop != "s1", drop != "s2"])
        return out
