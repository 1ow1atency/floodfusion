"""Tests for floodfusion.clouds with fake masks / a fake detector (s2cloudless is not run)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from floodfusion.clouds import (
    CLOUD_BIN_LABELS, CSV_COLUMNS, bin_counts, cloud_bin, cloud_fractions, cloud_mask,
    compute_cloud_cover, format_bin_table, pick_examples, read_cloud_csv, s2_reflectance,
    write_cloud_csv,
)

H = W = 8


# --- fraction calculation with a fake mask ----------------------------------

def test_cloud_fraction_counts_only_valid_pixels():
    valid = np.ones((4, 4), bool)
    valid[0] = False                      # 12 valid pixels
    mask = np.zeros((4, 4), bool)
    mask[0, :2] = True                    # cloud on invalid pixels: must be ignored
    mask[1, :3] = True                    # 3 cloudy valid pixels
    cloud, valid_frac = cloud_fractions(mask, valid)
    assert cloud == pytest.approx(3 / 12)
    assert valid_frac == pytest.approx(12 / 16)


def test_cloud_fraction_edge_cases():
    assert cloud_fractions(np.ones((2, 2), bool), np.ones((2, 2), bool)) == (1.0, 1.0)
    assert cloud_fractions(np.zeros((2, 2), bool), np.ones((2, 2), bool)) == (0.0, 1.0)
    cloud, valid_frac = cloud_fractions(np.ones((2, 2), bool), np.zeros((2, 2), bool))
    assert math.isnan(cloud) and valid_frac == 0.0
    with pytest.raises(ValueError):
        cloud_fractions(np.ones((2, 2), bool), np.ones((3, 3), bool))


def test_reflectance_scaling_and_valid_mask():
    s2 = np.full((13, 3, 3), 2500.0, np.float32)
    s2[:, 0, 0] = np.nan          # all-band nodata (as produced by mark_missing)
    s2[4, 1, 1] = 0.0             # genuine 0 in one band: still valid
    refl, valid = s2_reflectance(s2)
    assert refl.shape == (3, 3, 13) and refl.dtype == np.float32
    assert np.allclose(refl[2, 2], 0.25)
    assert (refl[0, 0] == 0).all() and refl[1, 1, 4] == 0
    assert valid.sum() == 8 and not valid[0, 0] and valid[1, 1]


class FakeDetector:
    """Probability = B2 reflectance; mask = prob > 0.3 (no smoothing)."""

    def __init__(self):
        self.seen_probs = None

    def get_cloud_probability_maps(self, data):
        assert data.ndim == 4 and data.shape[-1] == 13
        return data[..., 1].copy()

    def get_mask_from_prob(self, cloud_probs):
        self.seen_probs = cloud_probs.copy()
        return (cloud_probs > 0.3).astype(np.uint8)


def test_cloud_mask_zeroes_nodata_probability_and_restricts_to_valid():
    s2 = np.full((13, 4, 4), 1000.0, np.float32)
    s2[1, :2, :] = 5000.0          # B2 bright in the top half -> "cloud"
    s2[:, 0, 0] = np.nan           # nodata inside the bright area
    det = FakeDetector()
    mask, valid = cloud_mask(det, s2)
    assert det.seen_probs[0, 0, 0] == 0.0       # nodata probability zeroed before post-processing
    assert not mask[0, 0] and mask.sum() == 7   # 8 bright pixels minus the nodata one
    assert cloud_fractions(mask, valid) == (pytest.approx(7 / 15), pytest.approx(15 / 16))


# --- bins, CSV, example selection --------------------------------------------

def test_cloud_bins_edges():
    assert [cloud_bin(f) for f in (0.0, 0.099, 0.1, 0.299, 0.3, 0.6, 1.0)] == [
        "0-10%", "0-10%", "10-30%", "10-30%", "30-60%", "60-100%", "60-100%"]
    assert cloud_bin(math.nan) is None
    with pytest.raises(ValueError):
        cloud_bin(1.2)


def _rows():
    return [
        {"chip_id": "a", "split": "train", "cloud_fraction": 0.05, "valid_s2_fraction": 1.0},
        {"chip_id": "b", "split": "train", "cloud_fraction": 0.95, "valid_s2_fraction": 0.5},
        {"chip_id": "c", "split": "train", "cloud_fraction": 0.40, "valid_s2_fraction": 1.0},
        {"chip_id": "d", "split": "bolivia", "cloud_fraction": math.nan, "valid_s2_fraction": 0.0},
        {"chip_id": "e", "split": "test", "cloud_fraction": 0.1, "valid_s2_fraction": 1.0},
    ]


def test_bin_counts_and_table():
    c = bin_counts(_rows())
    assert c["train"] == {"0-10%": 1, "10-30%": 0, "30-60%": 1, "60-100%": 1, "no valid S2": 0, "total": 3}
    assert c["bolivia"]["no valid S2"] == 1 and c["test"]["10-30%"] == 1 and c["valid"]["total"] == 0
    table = format_bin_table(_rows())
    assert all(label in table for label in CLOUD_BIN_LABELS) and "bolivia" in table


def test_csv_roundtrip(tmp_path: Path):
    path = write_cloud_csv(_rows(), tmp_path / "x" / "cc.csv")
    assert path.read_text().splitlines()[0] == ",".join(CSV_COLUMNS)
    back = read_cloud_csv(path)
    assert [r["chip_id"] for r in back] == list("abcde")
    assert math.isnan(back[3]["cloud_fraction"]) and back[1]["cloud_fraction"] == pytest.approx(0.95)


def test_pick_examples():
    ex = pick_examples(_rows(), "train", n_cloudy=2, n_clear=2)
    assert [r["chip_id"] for r in ex] == ["b", "c", "a"]   # no chip repeated, NaN never picked


# --- whole pipeline on fake chips, fake detector ---------------------------------

def _write_tif(path: Path, arr: np.ndarray, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:4326",
                       transform=Affine(1e-4, 0, 0, 0, -1e-4, 0), nodata=nodata) as dst:
        dst.write(arr)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    # chip k: B2 bright (cloud) in the first k rows; chip "N" has a nodata half.
    chips = {"train": ["T0", "T2", "T8"], "valid": ["V4"], "test": [], "bolivia": ["N"]}
    for split, ids in chips.items():
        for cid in ids:
            s2 = np.full((13, H, W), 1000, np.int16)
            k = 4 if cid == "N" else int(cid[1:])
            s2[1, :k] = 5000
            if cid == "N":
                s2[:, :, : W // 2] = 0   # left half nodata -> excluded
            _write_tif(path=tmp_path / "S2Hand" / f"{cid}_S2Hand.tif", arr=s2, nodata=0)
            _write_tif(tmp_path / "S1Hand" / f"{cid}_S1Hand.tif", np.zeros((2, H, W), np.float32))
            _write_tif(tmp_path / "LabelHand" / f"{cid}_LabelHand.tif", np.zeros((1, H, W), np.int16))
        rows = "".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\r\n" for c in ids)
        (tmp_path / "splits").mkdir(exist_ok=True)
        (tmp_path / "splits" / f"flood_{split}_data.csv").write_bytes(rows.encode())
    return tmp_path


def test_compute_cloud_cover_all_splits(root: Path):
    rows = compute_cloud_cover(root, detector=FakeDetector(), progress=False)
    got = {r["chip_id"]: (r["split"], r["cloud_fraction"], r["valid_s2_fraction"]) for r in rows}
    assert got["T0"] == ("train", 0.0, 1.0)
    assert got["T2"] == ("train", pytest.approx(2 / 8), 1.0)
    assert got["T8"] == ("train", 1.0, 1.0)
    assert got["V4"] == ("valid", pytest.approx(0.5), 1.0)
    assert got["N"] == ("bolivia", pytest.approx(0.5), pytest.approx(0.5))  # 16 cloudy / 32 valid
    assert {r["split"] for r in rows} == {"train", "valid", "bolivia"}


def test_rasters_stats_clouds_viz_do_not_import_torch():
    # torch's bundled OpenMP + LightGBM's (Homebrew) OpenMP in one process abort on macOS.
    code = ("import sys, floodfusion.chips, floodfusion.clouds, floodfusion.clouds_synth, floodfusion.stats, floodfusion.viz; "
            "sys.exit(int('torch' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
