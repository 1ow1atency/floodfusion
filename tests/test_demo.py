"""Demo (floodfusion.demo + app/app.py): selection rule, packed chips, opaque clouds,
and an end-to-end smoke test with FAKE (random) weights. Never touches the real dataset."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from floodfusion.chips import IGNORE_INDEX, load_label, load_raw
from floodfusion.demo import (
    MANIFEST, chip_water_iou, label_panel, load_demo_model, load_packed_chip, make_dev_assets,
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


# ----------------------------------------------------------------------------- static export

def test_export_static_demo(dev_assets, tmp_path):
    """The static export writes every referenced file and the IoUs it records are the ones
    ``predict`` gives for the same chip and cloud cover."""
    from floodfusion.demo import DATA_JSON, export_static_demo, load_demo_model

    chips_dir, weights = dev_assets
    out = tmp_path / "static"
    coverages = [0, 50, 100]
    data = export_static_demo(chips_dir, weights, out, coverages=coverages, log=lambda *a: None)

    assert json.loads((out / DATA_JSON).read_text()) == data
    assert data["coverages"] == coverages
    assert data["smoke"] is True  # dev checkpoints: the page must warn
    assert [m["key"] for m in data["models"]] == list(weights)
    assert len(data["chips"]) == 2

    models = {name: load_demo_model(p) for name, p in weights.items()}
    for c in data["chips"]:
        chip = load_packed_chip(chips_dir / f"{c['chip_id']}.npz")
        for rel in (c["sar"], c["label"]):
            assert (out / rel).is_file(), rel
        assert [f["coverage"] for f in c["frames"]] == coverages
        for f in c["frames"]:
            assert (out / f["optical"]).is_file()
            s2 = opaque_cloud_s2(chip["s2"], f["coverage"] / 100.0, c["chip_id"])
            for name, dm in models.items():
                assert (out / f["preds"][name]["image"]).is_file()
                want = chip_water_iou(predict(dm, chip["s1"], s2), chip["label"])
                got = f["preds"][name]["iou"]
                assert (got is None) == math.isnan(want)
                if got is not None:
                    assert got == pytest.approx(want, abs=5e-5)


def test_export_static_demo_image_sizes(dev_assets, tmp_path):
    """Panels keep the chip's shape; label maps are small indexed PNGs with exact colours."""
    from PIL import Image

    from floodfusion.demo import export_static_demo
    from floodfusion.viz import LABEL_COLOURS

    chips_dir, weights = dev_assets
    out = tmp_path / "static"
    data = export_static_demo(chips_dir, weights, out, coverages=[0], log=lambda *a: None)
    c = data["chips"][0]
    chip = load_packed_chip(chips_dir / f"{c['chip_id']}.npz")
    h, w = chip["label"].shape
    for rel in (c["sar"], c["frames"][0]["optical"]):
        assert Image.open(out / rel).size == (w, h)
    lab = Image.open(out / c["label"])
    assert lab.mode == "P" and lab.size == (w, h)
    # palette index i is the i-th sorted label value, so the page's legend colours are exact
    pal = lab.getpalette()
    for i, k in enumerate(sorted(LABEL_COLOURS)):
        want = [int(LABEL_COLOURS[k][j: j + 2], 16) for j in (1, 3, 5)]
        assert pal[3 * i: 3 * i + 3] == want
    np.testing.assert_array_equal(np.asarray(lab.convert("RGB")), label_panel(chip["label"]))


def test_static_page_matches_the_data_contract():
    """index.html must look up the element ids its own JS builds for each model key, and quote
    the same headline numbers as the Gradio app (the two front-ends are kept in step by hand)."""
    import re

    from floodfusion.demo import HEADLINE, WEIGHT_FILES

    html = (REPO / "static_demo" / "index.html").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    assert {"chip", "cover", "cover-value", "img-optical", "img-sar", "img-label"} <= ids
    for key in WEIGHT_FILES:
        assert {f"img-pred-{key}", f"iou-{key}", f"title-{key}"} <= ids, key

    intro = (REPO / "app" / "app.py").read_text().split('INTRO = """')[1].split('"""')[0]
    for group in ("clean", "opaque_100"):
        for key, value in HEADLINE[group].items():
            quoted = f"{value:.2f}"
            assert quoted in html, f"{quoted} ({group} {key}) missing from index.html"
            assert quoted in intro, f"{quoted} ({group} {key}) missing from the Gradio intro"
    for text in ("precomputed", "opaque", "test split"):
        assert text in html.lower(), text


def test_static_demo_export_is_complete():
    """Every image the committed data.json references exists, and nothing is left over."""
    from floodfusion.demo import DATA_JSON

    root = REPO / "static_demo"
    data = json.loads((root / DATA_JSON).read_text())
    assert data["smoke"] is False, "the committed export must come from the real checkpoints"
    refs = set()
    for c in data["chips"]:
        refs |= {c["sar"], c["label"]}
        for f in c["frames"]:
            refs.add(f["optical"])
            refs |= {p["image"] for p in f["preds"].values()}
    assert not [r for r in sorted(refs) if not (root / r).is_file()]
    on_disk = {str(p.relative_to(root)) for p in root.rglob("*") if p.suffix in (".webp", ".png")}
    assert on_disk == refs
