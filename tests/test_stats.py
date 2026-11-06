"""Tests for floodfusion.stats and scripts/compute_stats.py on tiny synthetic chips."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from floodfusion.data import S2_BANDS, Sen1Floods11
from floodfusion.stats import compute_band_stats, compute_label_stats

H = W = 32
N = H * W
TRAIN = ("T0", "T1")
REPO = Path(__file__).resolve().parents[1]

# Invalid-pixel layout (flat pixel indices), identical in both train chips.
S1_NAN = np.arange(0, 100)           # NaN in both S1 bands
VV_INF, VH_NEGINF = 200, 201         # +inf in VV, -inf in VH
S2_ALLZERO = np.arange(900, N)       # all 13 S2 bands == 0 -> missing
S2_PARTIAL_ZERO = (3, 500)           # (band, pixel): genuine 0 in one band -> valid
LO, HI = 1.0, 99.0


def _write_tif(path: Path, arr: np.ndarray, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:4326",
                       transform=Affine(1e-4, 0, 0, 0, -1e-4, 0), nodata=nodata) as dst:
        dst.write(arr)


def _s1_flat(k: int) -> np.ndarray:
    a = np.stack([np.arange(N) * 0.1 - 50 + 30 * k, np.arange(N) * -0.02 - 5 - 7 * k]).astype(np.float32)
    a[:, S1_NAN] = np.nan
    a[0, VV_INF] = np.inf
    a[1, VH_NEGINF] = -np.inf
    return a


def _s2_flat(k: int) -> np.ndarray:
    a = np.stack([np.arange(N) % 500 + 1 + 10 * b + 1000 * k for b in range(13)]).astype(np.int16)
    a[:, S2_ALLZERO] = 0
    a[S2_PARTIAL_ZERO] = 0
    return a


def _label(k: int) -> np.ndarray:
    y = np.zeros(N, np.int16)
    y[:50] = -1
    y[50 : 150 + 100 * k] = 1  # T0: 100 water, T1: 200 water
    return y.reshape(1, H, W)


def _expected_valid(mod: str, b: int) -> np.ndarray:
    """Valid values of band b over both train chips, built from the known layout."""
    keep = np.ones(N, bool)
    if mod == "s1":
        keep[S1_NAN] = False
        keep[VV_INF if b == 0 else VH_NEGINF] = False
        return np.concatenate([_s1_flat(k)[b][keep] for k in range(2)]).astype(np.float64)
    keep[S2_ALLZERO] = False
    return np.concatenate([_s2_flat(k)[b][keep] for k in range(2)]).astype(np.float64)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for k, cid in enumerate(TRAIN):
        _write_tif(tmp_path / "S1Hand" / f"{cid}_S1Hand.tif", _s1_flat(k).reshape(2, H, W), nodata=np.nan)
        _write_tif(tmp_path / "S2Hand" / f"{cid}_S2Hand.tif", _s2_flat(k).reshape(13, H, W), nodata=0)
        _write_tif(tmp_path / "LabelHand" / f"{cid}_LabelHand.tif", _label(k))
    splits = tmp_path / "splits"
    splits.mkdir()
    row = lambda c: f"{c}_S1Hand.tif,{c}_LabelHand.tif\r\n"  # noqa: E731
    (splits / "flood_train_data.csv").write_bytes((row("T0") + row("T1") + "\r\n").encode())
    (splits / "flood_valid_data.csv").write_bytes(row("T0").encode())
    (splits / "flood_test_data.csv").write_bytes(row("Ghost").encode())  # missing files
    (splits / "flood_bolivia_data.csv").write_bytes(row("T1").encode())
    return tmp_path


def test_fixture_is_discriminating():
    # Including the invalid pixels would visibly change the stats, so the tests below
    # can tell whether they were skipped.
    s2_all = np.concatenate([_s2_flat(k)[0] for k in range(2)]).astype(np.float64)
    assert np.percentile(s2_all, LO) == 0 != np.percentile(_expected_valid("s2", 0), LO)
    assert s2_all.mean() < _expected_valid("s2", 0).mean() - 50


def test_band_stats_use_only_valid_pixels(root: Path):
    stats = compute_band_stats(root, seed=0, verbose=False)
    assert stats["meta"]["percentile_method"] == "exact"
    assert stats["meta"]["split"] == "train" and stats["meta"]["n_chips"] == 2
    for mod, bands in (("s1", ("VV", "VH")), ("s2", S2_BANDS)):
        s = stats[mod]
        assert s["bands"] == list(bands)
        for b in range(len(bands)):
            valid = _expected_valid(mod, b)
            lo, hi = np.percentile(valid, [LO, HI])
            clipped = np.clip(valid, lo, hi)
            assert s["clip_min"][b] == pytest.approx(lo, rel=1e-6)
            assert s["clip_max"][b] == pytest.approx(hi, rel=1e-6)
            assert s["mean"][b] == pytest.approx(clipped.mean(), rel=1e-6)
            assert s["std"][b] == pytest.approx(clipped.std(), rel=1e-5)
            assert s["n_valid"][b] == s["n_sampled"][b] == valid.size
    # The genuine single-band zero counts as valid (one more pixel than other bands).
    b3 = S2_BANDS.index("B3")
    assert stats["s2"]["n_valid"][b3] == 2 * (N - S2_ALLZERO.size)
    assert stats["s2"]["clip_min"][b3] < stats["s2"]["clip_min"][b3 + 1]


def test_subsample_is_seeded_and_close(root: Path):
    a = compute_band_stats(root, seed=1, max_pixels_per_band=800, verbose=False)
    b = compute_band_stats(root, seed=1, max_pixels_per_band=800, verbose=False)
    c = compute_band_stats(root, seed=2, max_pixels_per_band=800, verbose=False)
    exact = compute_band_stats(root, verbose=False)
    assert a["meta"]["percentile_method"] == "subsample"
    assert a["meta"]["sample_fraction"] == pytest.approx(800 / (2 * N))
    assert a == b and a["s2"]["clip_min"] != c["s2"]["clip_min"]
    for mod in ("s1", "s2"):
        assert all(ns < nv for ns, nv in zip(a[mod]["n_sampled"], a[mod]["n_valid"]))
        assert a[mod]["n_valid"] == exact[mod]["n_valid"]  # pass 2 always uses every valid pixel
        rng = np.array(exact[mod]["clip_max"]) - np.array(exact[mod]["clip_min"])
        assert np.allclose(a[mod]["clip_min"], exact[mod]["clip_min"], atol=0.05 * rng.max())
        assert np.allclose(a[mod]["mean"], exact[mod]["mean"], atol=0.05 * rng.max())


def test_stats_are_accepted_by_dataset(root: Path):
    stats = compute_band_stats(root, verbose=False)
    s = Sen1Floods11(root, "train", stats=json.loads(json.dumps(stats)))[0]
    assert all(np.isfinite(s[m].numpy()).all() for m in ("s1", "s2"))


def test_label_stats(root: Path):
    ls = compute_label_stats(root, allow_missing=True)
    labelled = N - 50
    assert ls["train"] == {"n_chips": 2, "labelled_pixels": 2 * labelled, "water_pixels": 300,
                           "water_fraction": pytest.approx(300 / (2 * labelled))}
    assert ls["valid"]["water_pixels"] == 100 and ls["bolivia"]["water_pixels"] == 200
    assert ls["test"] == {"n_chips": 0, "labelled_pixels": 0, "water_pixels": 0, "water_fraction": None}
    with pytest.raises(FileNotFoundError):
        compute_label_stats(root)  # strict by default: 'test' lists a missing chip


def test_script_writes_both_files(root: Path, tmp_path: Path):
    out = tmp_path / "out" / "band_stats.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "compute_stats.py"), "--root", str(root),
                        "--out", str(out), "--allow-missing", "--seed", "3"],
                       capture_output=True, text=True, check=True)
    assert json.loads(out.read_text())["meta"]["seed"] == 3
    assert set(json.loads((out.parent / "label_stats.json").read_text())) == {
        "meta", "train", "valid", "test", "bolivia"}
    assert "B8A" in r.stdout and "water_frac" in r.stdout
