"""Shared fixtures: a tiny synthetic Sen1Floods11 tree (never the real dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

SYN = 64  # chip size
SYN_SPLITS = {"train": [f"Tr_{i}" for i in range(6)], "valid": ["Va_0", "Va_1"],
              "test": ["Te_0", "Te_1"], "bolivia": ["Bo_0"]}


def write_tif(path: Path, arr: np.ndarray, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c, dtype=arr.dtype,
                       crs="EPSG:4326", transform=Affine(1e-4, 0, 0, 0, -1e-4, 0), nodata=nodata) as dst:
        dst.write(arr)


def _chip(rng: np.random.Generator):
    y = np.zeros((SYN, SYN), np.int16)
    r, c = rng.integers(0, SYN - 24, size=2)
    y[r : r + 24, c : c + 24] = 1                       # a water block
    y[:2] = -1                                          # some no-data rows
    water = y == 1
    s1 = np.stack([np.where(water, -20.0, -8.0), np.where(water, -28.0, -15.0)]) + rng.normal(0, 1, (2, SYN, SYN))
    s1 = s1.astype(np.float32)
    s1[:, -1, :4] = np.nan
    s2 = (rng.integers(500, 3000, (13, SYN, SYN)) - 400 * water).astype(np.int16)
    s2[:, -2:, -2:] = 0                                 # all-band nodata
    return s1, s2, y[None]


@pytest.fixture
def synth_root(tmp_path: Path) -> Path:
    root = tmp_path / "sen1floods11"
    rng = np.random.default_rng(0)
    for split, ids in SYN_SPLITS.items():
        for cid in ids:
            s1, s2, y = _chip(rng)
            write_tif(root / "S1Hand" / f"{cid}_S1Hand.tif", s1, nodata=np.nan)
            write_tif(root / "S2Hand" / f"{cid}_S2Hand.tif", s2, nodata=0)
            write_tif(root / "LabelHand" / f"{cid}_LabelHand.tif", y)
        (root / "splits").mkdir(parents=True, exist_ok=True)
        (root / "splits" / f"flood_{split}_data.csv").write_bytes(
            "".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\r\n" for c in ids).encode())
    stats = {
        "s1": {"mean": [-10.0, -17.0], "std": [4.0, 4.5], "clip_min": [-30.0, -40.0], "clip_max": [0.0, -5.0]},
        "s2": {"mean": [1500.0] * 13, "std": [700.0] * 13, "clip_min": [0.0] * 13, "clip_max": [5000.0] * 13},
        "meta": {"split": "train"},
    }
    (tmp_path / "band_stats.json").write_text(json.dumps(stats))
    return root


@pytest.fixture
def synth_cfg(synth_root: Path, tmp_path: Path) -> dict:
    """Resolved training config for a tiny, fast CPU run on the synthetic data."""
    from floodfusion.training import resolve_config

    cfg = {
        "name": "tiny", "seed": 0, "output_dir": str(tmp_path / "runs"),
        "data": {"root": str(synth_root), "stats": str(tmp_path / "band_stats.json"), "modalities": ["s1", "s2"],
                 "crop_size": 32, "num_workers": 0},
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"batch_size": 2, "val_batch_size": 2, "epochs": 2, "lr": 1e-3, "weight_decay": 1e-2},
        "wandb": {"enabled": False},
    }
    return resolve_config(cfg)
