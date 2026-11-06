"""Tests for floodfusion.viz and scripts/visualize_sample.py on tiny synthetic chips."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from floodfusion.data import S2_BANDS
from floodfusion.viz import (
    FALSE_COLOUR, NODATA_RGB, TRUE_COLOUR, composite, plot_chip, plot_s1_histograms,
    s1_class_histograms, water_fraction,
)

H = W = 16
REPO = Path(__file__).resolve().parents[1]
STATS = {"s1": {"bands": ["VV", "VH"], "clip_min": [-25.0, -30.0], "clip_max": [-3.0, -10.0]}}


def _write_tif(path: Path, arr: np.ndarray, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:4326",
                       transform=Affine(1e-4, 0, 0, 0, -1e-4, 0), nodata=nodata) as dst:
        dst.write(arr)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two chips. Left half water (label 1, VV=-20, VH=-28), right half land (0, -8, -15).
    Row 0: label -1. Row 1: S1 NaN. C1 has one VV water pixel at -60 dB (below HIST_EDGES)."""
    for cid in ("C0", "C1"):
        y = np.zeros((H, W), np.int16)
        y[:, : W // 2] = 1
        y[0] = -1
        s1 = np.stack([np.where(y == 1, -20.0, -8.0), np.where(y == 1, -28.0, -15.0)]).astype(np.float32)
        s1[:, 1] = np.nan
        if cid == "C1":
            s1[0, 5, 0] = -60.0
        s2 = np.stack([np.full((H, W), 1000 + 100 * b) for b in range(13)]).astype(np.int16)
        s2[:, 2, :4] = 0  # all-band nodata
        _write_tif(tmp_path / "S1Hand" / f"{cid}_S1Hand.tif", s1, nodata=np.nan)
        _write_tif(tmp_path / "S2Hand" / f"{cid}_S2Hand.tif", s2, nodata=0)
        _write_tif(tmp_path / "LabelHand" / f"{cid}_LabelHand.tif", y[None])
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "flood_train_data.csv").write_bytes(
        b"C0_S1Hand.tif,C0_LabelHand.tif\r\nC1_S1Hand.tif,C1_LabelHand.tif\r\n")
    (tmp_path / "stats.json").write_text(json.dumps(STATS))
    return tmp_path


def test_composite_band_order_scale_and_nodata():
    s2 = np.stack([np.full((4, 4), 100.0 * (b + 1)) for b in range(13)]).astype(np.float32)
    s2[:, 0, 0] = np.nan
    rgb = composite(s2, TRUE_COLOUR, scale=1000.0)
    assert rgb.shape == (4, 4, 3) and rgb.min() >= 0 and rgb.max() <= 1
    expected = [100.0 * (S2_BANDS.index(b) + 1) / 1000.0 for b in TRUE_COLOUR]
    assert np.allclose(rgb[1, 1], expected)
    assert tuple(rgb[0, 0]) == NODATA_RGB
    assert np.allclose(composite(s2, FALSE_COLOUR, scale=100.0)[1, 1], 1.0)  # clipped to 1
    with pytest.raises(ValueError):
        composite(s2[:3], TRUE_COLOUR, 1000.0)


def test_water_fraction_ignores_no_data():
    y = np.array([[1, 0, -1, -1], [1, 1, 0, -1]])
    assert water_fraction(y) == pytest.approx(3 / 5)
    assert water_fraction(np.full((2, 2), -1)) is None


def test_s1_histograms_count_only_labelled_valid_pixels(root: Path):
    h = s1_class_histograms(root, ["C0", "C1"])
    assert h["n_chips"] == 2
    per_chip = (H - 2) * (W // 2)  # rows 0 (label -1) and 1 (NaN) excluded, half the columns
    for band in ("VV", "VH"):
        for cls in ("water", "land"):
            d = h[band][cls]
            assert d["total"] == 2 * per_chip
            assert d["counts"].sum() + d["below"] + d["above"] == d["total"]
    assert h["VV"]["water"]["below"] == 1  # the -60 dB pixel is counted, not dropped
    assert h["VV"]["land"]["below"] == 0


def test_plots_are_written(root: Path, tmp_path: Path):
    p = plot_chip(root, "C0", STATS, tmp_path / "figs" / "c0.png")
    q = plot_s1_histograms(s1_class_histograms(root, ["C0"]), STATS, tmp_path / "figs" / "h.png", "t")
    for f in (p, q):
        assert f.is_file() and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_script_end_to_end(root: Path, tmp_path: Path):
    out = tmp_path / "figures"
    cmd = [sys.executable, str(REPO / "scripts" / "visualize_sample.py"), "--root", str(root),
           "--split", "train", "--index", "1", "--n", "5", "--stats", str(root / "stats.json"),
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert sorted(p.name for p in out.iterdir()) == ["train_001_C1.png", "train_s1_hist.png"]
    assert "only 1 chips available" in r.stdout
    subprocess.run(cmd[:-1] + [str(tmp_path / "f2"), "--no-hist"], check=True, capture_output=True)
    assert [p.name for p in (tmp_path / "f2").iterdir()] == ["train_001_C1.png"]
    i = cmd.index("--index") + 1
    bad = subprocess.run(cmd[:i] + ["7"] + cmd[i + 1:], capture_output=True, text=True)
    assert bad.returncode != 0 and "--index must be in [0, 1]" in bad.stderr
