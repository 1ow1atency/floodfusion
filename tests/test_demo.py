"""Demo (floodfusion.demo + app/app.py): selection rule, packed chips, opaque clouds,
and an end-to-end smoke test with FAKE (random) weights. Never touches the real dataset."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from floodfusion.chips import IGNORE_INDEX, load_label, load_raw
from floodfusion.demo import (
    MANIFEST, chip_water_iou, load_demo_model, load_packed_chip, make_dev_assets,
    opaque_cloud_s2, pack_chip, predict, select_demo_chips,
)

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- selection

def _write_selection_csvs(tmp_path, rows):
    """rows: (chip_id, split, cloud, unreliable, valid_s2, water_px, land_px)."""
    cloud = ["chip_id,split,cloud_fraction,valid_s2_fraction,mask_unreliable"]
    audit = ["chip_id,split,t0.4_cloudy_water,t0.4_clear_water,t0.4_cloudy_land,t0.4_clear_land"]
    for cid, split, c, unrel, valid, water, land in rows:
        cloud.append(f"{cid},{split},{c},{valid},{unrel}")
        audit.append(f"{cid},{split},0,{water},0,{land}")
    (tmp_path / "cloud.csv").write_text("\n".join(cloud) + "\n")
    (tmp_path / "audit.csv").write_text("\n".join(audit) + "\n")
    return tmp_path / "cloud.csv", tmp_path / "audit.csv"


def test_select_demo_chips_rule(tmp_path):
    cloud_csv, audit_csv = _write_selection_csvs(tmp_path, [
        ("A_1", "test", 0.0, 0, 1.0, 50, 50),       # region A, 50% water
        ("A_2", "test", 0.005, 0, 1.0, 30, 70),     # region A, 30% water
        ("B_1", "test", 0.0, 0, 1.0, 40, 60),       # region B, 40% water
        ("C_1", "test", 0.5, 0, 1.0, 90, 10),       # too cloudy
        ("D_1", "test", 0.0, 1, 1.0, 90, 10),       # unreliable mask
        ("E_1", "test", 0.0, 0, 0.9, 90, 10),       # not fully valid S2
        ("F_1", "test", 0.0, 0, 1.0, 2, 98),        # too little water
        ("G_1", "train", 0.0, 0, 1.0, 90, 10),      # wrong split
        ("H_1", "test", "", 0, 1.0, 90, 10),        # no cloud fraction
    ])
    picked = select_demo_chips(cloud_csv, audit_csv, n=3)
    # round-robin by region (A wettest, then B), then rank 2 of A
    assert [c["chip_id"] for c in picked] == ["A_1", "B_1", "A_2"]
    assert picked[0]["water_fraction"] == 0.5 and picked[0]["region"] == "A"
    # deterministic
    assert picked == select_demo_chips(cloud_csv, audit_csv, n=3)


def test_select_demo_chips_on_repo_csvs():
    """The committed CSVs give 8 test chips from several regions, none from Bolivia."""
    picked = select_demo_chips(REPO / "docs/cloud_cover.csv",
                               REPO / "docs/results/cloud_audit/chip_audit.csv")
    assert len(picked) == 8
    assert len({c["region"] for c in picked}) >= 4
    assert all(c["region"] != "Bolivia" for c in picked)
    assert all(c["cloud_fraction"] <= 0.01 and c["water_fraction"] >= 0.05 for c in picked)


# ----------------------------------------------------------------------------- packed chips

def test_pack_roundtrip(synth_root, tmp_path):
    out = pack_chip(synth_root, "Te_0", tmp_path / "Te_0.npz")
    with np.load(out) as z:
        assert z["s2"].dtype == np.uint16  # raw DN packed losslessly
        assert z["label"].dtype == np.int8
    chip = load_packed_chip(out)
    assert chip["id"] == "Te_0"
    np.testing.assert_array_equal(chip["s1"], load_raw(synth_root, "Te_0", "s1"))
    np.testing.assert_array_equal(chip["s2"], load_raw(synth_root, "Te_0", "s2"))
    np.testing.assert_array_equal(chip["label"], load_label(synth_root, "Te_0"))
    assert chip["s1"].dtype == chip["s2"].dtype == np.float32
    assert np.isnan(chip["s2"][:, -1, -1]).all()  # all-band-0 nodata rule re-applied


# ----------------------------------------------------------------------------- clouds

def test_opaque_cloud_s2(synth_root):
    s2 = load_raw(synth_root, "Te_0", "s2")
    np.testing.assert_array_equal(opaque_cloud_s2(s2, 0.0, "Te_0"), s2)
    half = opaque_cloud_s2(s2, 0.5, "Te_0")
    np.testing.assert_array_equal(half, opaque_cloud_s2(s2, 0.5, "Te_0"))  # deterministic
    assert not np.array_equal(np.nan_to_num(half), np.nan_to_num(s2))
    assert np.isnan(half[:, -1, -1]).all()  # NaN stays NaN
    # the same seed for every coverage: a covered pixel stays covered as coverage grows
    nan_free = ~np.isnan(s2).any(axis=0)
    covered50 = (np.nan_to_num(half) != np.nan_to_num(s2)).any(axis=0) & nan_free
    full = opaque_cloud_s2(s2, 1.0, "Te_0")
    covered100 = (np.nan_to_num(full) != np.nan_to_num(s2)).any(axis=0) & nan_free
    assert (covered50 <= covered100).all() and covered100.mean() > 0.95
    # fully opaque: at 100% the output does not depend on the ground
    other = opaque_cloud_s2(np.full_like(s2, 9999.0), 1.0, "Te_0")
    np.testing.assert_allclose(full[:, nan_free], other[:, nan_free])


# ----------------------------------------------------------------------------- IoU

def test_chip_water_iou():
    label = np.array([[1, 1, 0, IGNORE_INDEX]])
    pred = np.array([[1, 0, 1, 1]], dtype=np.uint8)  # the ignored pixel must not count
    assert chip_water_iou(pred, label) == pytest.approx(1 / 3)
    assert chip_water_iou(np.array([[1, 1, 0, 0]], np.uint8), label) == 1.0
    assert math.isnan(chip_water_iou(np.zeros((1, 4), np.uint8), np.zeros((1, 4), np.int64)))


# ----------------------------------------------------------------------------- fake weights, end to end

@pytest.fixture(scope="module")
def dev_assets(tmp_path_factory):
    return make_dev_assets(tmp_path_factory.mktemp("dev"), size=64)


def test_predict_with_fake_weights(dev_assets):
    chips_dir, weights = dev_assets
    chip = load_packed_chip(chips_dir / "Dev_0.npz")
    for name, path in weights.items():
        dm = load_demo_model(path)
        assert dm.smoke, "dev checkpoints must be marked smoke"
        for cov in (0.0, 1.0):
            pred = predict(dm, chip["s1"], opaque_cloud_s2(chip["s2"], cov, chip["id"]))
            assert pred.shape == chip["label"].shape and pred.dtype == np.uint8
            assert set(np.unique(pred)) <= {0, 1}
            iou = chip_water_iou(pred, chip["label"])
            assert math.isnan(iou) or 0.0 <= iou <= 1.0
        # deterministic
        p = predict(dm, chip["s1"], chip["s2"])
        np.testing.assert_array_equal(p, predict(dm, chip["s1"], chip["s2"]))


def test_app_smoke(dev_assets):
    """Build the Gradio UI with fake weights and run one full view update."""
    pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("floodfusion_demo_app", REPO / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    chips_dir, weights = dev_assets
    assets = app.DemoAssets(chips_dir, weights)
    demo = app.build_demo(assets)
    assert demo is not None
    out = assets.view(assets.choice_label("Dev_0"), 60)
    optical, sar, label, pred_a, pred_b, iou_md = out
    for img in (optical, sar, label, pred_a, pred_b):
        assert img.dtype == np.uint8 and img.shape == (64, 64, 3)
    assert "IoU" in iou_md and "random weights" in iou_md.lower()
    # cached: a second call returns the identical panels
    np.testing.assert_array_equal(optical, assets.view(assets.choice_label("Dev_0"), 60)[0])
