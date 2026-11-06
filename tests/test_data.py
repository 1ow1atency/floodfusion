"""Tests for floodfusion.data using tiny synthetic GeoTIFFs (never the real dataset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

from floodfusion.data import IGNORE_INDEX, S1_BANDS, S2_BANDS, Sen1Floods11

H = W = 32
CHIPS = ("Fake_1", "Fake_2", "Fake_3")  # Fake_3 has no S2 file
MISSING_S2 = "Fake_3"


def _write_tif(path: Path, arr: np.ndarray, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=count, dtype=arr.dtype,
        crs="EPSG:4326", transform=Affine(1e-4, 0, 0, 0, -1e-4, 0), nodata=nodata,
    ) as dst:
        dst.write(arr)


def _s1(k: int) -> np.ndarray:
    # Band b is filled with -(10*b + k) dB; a NaN block in the top-left corner.
    a = np.stack([np.full((H, W), -(10.0 * b + k), np.float32) for b in range(len(S1_BANDS))])
    a[:, :4, :4] = np.nan
    return a


def _s2(k: int) -> np.ndarray:
    # Band b is filled with 100*b + k, so selected bands are identifiable by value.
    return np.stack([np.full((H, W), 100 * b + k, np.int16) for b in range(len(S2_BANDS))])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for k, cid in enumerate(CHIPS, start=1):
        _write_tif(tmp_path / "S1Hand" / f"{cid}_S1Hand.tif", _s1(k), nodata=np.nan)
        if cid != MISSING_S2:
            _write_tif(tmp_path / "S2Hand" / f"{cid}_S2Hand.tif", _s2(k), nodata=0)
        if k == 1:
            # Float label with NaN plus explicit -1 (exercises NaN -> IGNORE_INDEX).
            y = np.zeros((1, H, W), np.float32)
            y[0, 10:20, 10:20] = 1
            y[0, 0, :] = np.nan
            y[0, -1, :] = -1
        else:
            # int16 label like the real data.
            y = np.zeros((1, H, W), np.int16)
            y[0, :, :8] = 1
            y[0, :2, :2] = -1
        _write_tif(tmp_path / "LabelHand" / f"{cid}_LabelHand.tif", y)

    # No header, CRLF line endings, blank rows (incl. trailing).
    rows = [f"{cid}_S1Hand.tif,{cid}_LabelHand.tif" for cid in CHIPS]
    text = rows[0] + "\r\n\r\n" + "\r\n".join(rows[1:]) + "\r\n\r\n"
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "flood_train_data.csv").write_bytes(text.encode())
    return tmp_path


@pytest.fixture
def ds(root: Path) -> Sen1Floods11:
    return Sen1Floods11(root, "train", allow_missing=True)


def test_missing_file_raises_by_default(root: Path):
    with pytest.raises(FileNotFoundError) as exc:
        Sen1Floods11(root, "train")
    msg = str(exc.value)
    assert f"{MISSING_S2}_S2Hand.tif" in msg
    assert "1 of 3 chips" in msg


def test_missing_error_lists_at_most_10(root: Path):
    ghosts = [f"Ghost_{i}_S1Hand.tif,x\r\n" for i in range(12)]
    (root / "splits" / "flood_valid_data.csv").write_text("".join(ghosts), newline="")
    with pytest.raises(FileNotFoundError) as exc:
        Sen1Floods11(root, "valid")
    msg = str(exc.value)
    # 12 chips x 3 files = 36 missing; only 10 listed.
    assert msg.count(".tif") == 10
    assert "and 26 more" in msg


def test_allow_missing_drops_and_reports(root: Path, capsys):
    ds = Sen1Floods11(root, "train", allow_missing=True)
    assert len(ds) == 2
    assert ds.ids == ["Fake_1", "Fake_2"]  # blank CSV rows skipped, order kept
    assert "kept 2 of 3" in capsys.readouterr().out


def test_shapes_and_dtypes(ds: Sen1Floods11):
    s = ds[0]
    assert set(s) == {"id", "s1", "s2", "label"}
    assert s["id"] == "Fake_1"
    assert s["s1"].shape == (2, H, W) and s["s1"].dtype == torch.float32
    assert s["s2"].shape == (13, H, W) and s["s2"].dtype == torch.float32
    assert s["label"].shape == (H, W) and s["label"].dtype == torch.int64


def test_single_modality(root: Path):
    s = Sen1Floods11(root, "train", modalities=("s1",), allow_missing=True)[0]
    assert set(s) == {"id", "s1", "label"}
    s = Sen1Floods11(root, "train", modalities=("s2",), allow_missing=True)[0]
    assert set(s) == {"id", "s2", "label"}


def test_s2_band_selection(root: Path):
    bands = ["B12", "B8A", "B2"]  # out-of-order on purpose
    s = Sen1Floods11(root, "train", s2_bands=bands, allow_missing=True)[1]  # Fake_2, k=2
    assert s["s2"].shape == (3, H, W)
    expected = [100 * S2_BANDS.index(b) + 2 for b in bands]
    assert [float(s["s2"][c, 0, 0]) for c in range(3)] == expected


def test_unknown_band_and_split_raise(root: Path):
    with pytest.raises(ValueError, match="Unknown band"):
        Sen1Floods11(root, "train", s2_bands=["B13"], allow_missing=True)
    with pytest.raises(ValueError, match="split"):
        Sen1Floods11(root, "training")


def test_nan_inputs_become_zero_without_stats(ds: Sen1Floods11):
    s1 = ds[0]["s1"]
    assert torch.isfinite(s1).all()
    assert (s1[:, :4, :4] == 0).all()
    assert s1[0, 10, 10] == -1.0 and s1[1, 10, 10] == -11.0  # untouched elsewhere


def _stats(s1_mean=(-5.0, -15.0), s1_std=(2.0, 4.0), s2_mean=None, s2_std=None,
           s1_clip=(-1e4, 1e4), s2_clip=(-1e5, 1e5)) -> dict:
    """Stats in the full format; default clip limits are wide enough never to bite."""
    def block(mean, std, clip):
        n = len(mean)
        lo, hi = clip
        lo = list(lo) if isinstance(lo, (list, tuple)) else [lo] * n
        hi = list(hi) if isinstance(hi, (list, tuple)) else [hi] * n
        return {"mean": list(mean), "std": list(std), "clip_min": lo, "clip_max": hi}

    s2_mean = [100.0 * b for b in range(13)] if s2_mean is None else s2_mean
    s2_std = [0.5] * 13 if s2_std is None else s2_std
    return {"s1": block(s1_mean, s1_std, s1_clip), "s2": block(s2_mean, s2_std, s2_clip)}


def test_normalisation_then_nan_to_zero(root: Path):
    stats = _stats()
    ds = Sen1Floods11(root, "train", s2_bands=["B3", "B11"], stats=stats, allow_missing=True)
    s = ds[0]  # Fake_1, k=1: VV=-1, VH=-11, band b = 100*b + 1
    assert torch.isfinite(s["s1"]).all() and torch.isfinite(s["s2"]).all()
    assert torch.allclose(s["s1"][:, 10, 10], torch.tensor([(-1 + 5) / 2, (-11 + 15) / 4]))
    assert (s["s1"][:, :4, :4] == 0).all()  # NaN -> 0 after normalisation
    # Selected bands use their own stats: (100*b + 1 - 100*b) / 0.5 = 2.
    assert torch.allclose(s["s2"][:, 5, 5], torch.tensor([2.0, 2.0]))


def test_bad_stats_length_raises(root: Path):
    stats = _stats(s1_mean=[0.0], s1_std=[1.0])
    with pytest.raises(ValueError, match="2 values"):
        Sen1Floods11(root, "train", stats=stats, allow_missing=True)


# --- clipping ---------------------------------------------------------------

def test_clip_before_normalise(root: Path):
    # Fake_1: VV=-1, VH=-11 dB; S2 band b = 100*b + 1.
    stats = _stats(s1_clip=([-50.0, -8.0], [-3.0, 0.0]),       # VV capped at -3, VH floored at -8
                   s2_clip=([0.0] * 13, [50.0] + [1e5] * 12))  # B1 (=1) untouched; B2 (=101) not capped
    stats["s2"]["clip_max"][2] = 150.0                         # B3 (=201) capped at 150
    s = Sen1Floods11(root, "train", s2_bands=["B1", "B3"], stats=stats, allow_missing=True)[0]
    assert torch.allclose(s["s1"][:, 10, 10], torch.tensor([(-3 + 5) / 2, (-8 + 15) / 4]))
    assert torch.allclose(s["s2"][:, 10, 10], torch.tensor([(1 - 0) / 0.5, (150 - 200) / 0.5]))
    # NaN is not clipped to a limit: it stays missing and becomes 0.
    assert (s["s1"][:, :4, :4] == 0).all()


@pytest.mark.parametrize("lo, hi", [(5.0, 5.0), (6.0, 5.0)])
def test_clip_min_must_be_below_clip_max(root: Path, lo, hi):
    stats = _stats(s1_clip=([-50.0, lo], [10.0, hi]))
    with pytest.raises(ValueError, match="clip_min must be < clip_max.*VH"):
        Sen1Floods11(root, "train", stats=stats, allow_missing=True)


def test_missing_clip_key_raises(root: Path):
    stats = _stats()
    del stats["s2"]["clip_max"]
    with pytest.raises(KeyError, match="clip_max"):
        Sen1Floods11(root, "train", stats=stats, allow_missing=True)


# --- S2 nodata (all 13 bands == 0) ------------------------------------------

ZERO_CHIP = "Zero_1"
B3 = S2_BANDS.index("B3")


@pytest.fixture
def zero_root(root: Path) -> Path:
    """Adds a 'test' split with one chip: all-zero S2 block at [:3, :3] and a pixel
    at (5, 5) where only B3 is a genuine 0."""
    s2 = _s2(1)
    s2[:, :3, :3] = 0
    s2[B3, 5, 5] = 0
    _write_tif(root / "S1Hand" / f"{ZERO_CHIP}_S1Hand.tif", _s1(1), nodata=np.nan)
    _write_tif(root / "S2Hand" / f"{ZERO_CHIP}_S2Hand.tif", s2, nodata=0)
    _write_tif(root / "LabelHand" / f"{ZERO_CHIP}_LabelHand.tif", np.zeros((1, H, W), np.int16))
    (root / "splits" / "flood_test_data.csv").write_bytes(
        f"{ZERO_CHIP}_S1Hand.tif,{ZERO_CHIP}_LabelHand.tif\r\n".encode())
    return root


def test_s2_partial_zero_pixel_unchanged_without_stats(zero_root: Path):
    x = Sen1Floods11(zero_root, "test")[0]["s2"]
    expected = torch.tensor([100.0 * b + 1 for b in range(13)])
    expected[B3] = 0.0
    assert torch.equal(x[:, 5, 5], expected)
    assert (x[:, :3, :3] == 0).all()


def test_s2_all_zero_pixels_are_missing_after_normalisation(zero_root: Path):
    s2_mean = [100.0 * b + 50 for b in range(13)]  # non-zero means, so a raw 0 would not map to 0
    ds = Sen1Floods11(zero_root, "test", stats=_stats(s2_mean=s2_mean))
    x = ds[0]["s2"]
    assert (x[:, :3, :3] == 0).all()  # missing -> band mean -> 0
    # Partial-zero pixel is normalised like real data, B3 included.
    expected = (torch.tensor([100.0 * b + 1 for b in range(13)]) - torch.tensor(s2_mean)) / 0.5
    expected[B3] = (0 - s2_mean[B3]) / 0.5
    assert torch.allclose(x[:, 5, 5], expected)
    assert x[B3, 5, 5] != 0


def test_s2_nodata_decided_before_band_selection(zero_root: Path):
    # Selecting only B3: at (5, 5) it is 0, but the other 12 bands are not, so it is
    # real data. A rule applied after selection would wrongly mark it missing.
    s2_mean = [100.0 * b + 50 for b in range(13)]
    x = Sen1Floods11(zero_root, "test", s2_bands=["B3"], stats=_stats(s2_mean=s2_mean))[0]["s2"]
    assert torch.isclose(x[0, 5, 5], torch.tensor((0 - s2_mean[B3]) / 0.5))
    assert (x[0, :3, :3] == 0).all()


def test_label_ignore_index(ds: Sen1Floods11):
    y = ds[0]["label"]  # float label: NaN in row 0, -1 in last row
    assert IGNORE_INDEX == -1
    assert (y[0] == IGNORE_INDEX).all()
    assert (y[-1] == IGNORE_INDEX).all()
    assert (y[10:20, 10:20] == 1).all()
    assert set(torch.unique(y).tolist()) == {-1, 0, 1}

    y2 = ds[1]["label"]  # int16 label
    assert (y2[:2, :2] == IGNORE_INDEX).all()
    assert set(torch.unique(y2).tolist()) == {-1, 0, 1}


def test_transform_applied(root: Path):
    def flip(s):
        return {**s, "label": torch.flip(s["label"], dims=[0]), "flipped": True}

    s = Sen1Floods11(root, "train", transform=flip, allow_missing=True)[0]
    assert s["flipped"] is True
    assert (s["label"][0] == IGNORE_INDEX).all()  # former last row (-1)
