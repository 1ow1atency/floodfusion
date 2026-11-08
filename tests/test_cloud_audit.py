"""s2cloudless mask audit (floodfusion.cloud_audit): label fractions under cloud vs clear, pooling,
agreement with label -1, thresholds, chip selection. Hand-made arrays and a fake detector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.cloud_audit import (
    LABEL_KEYS, bin_counts_at, chip_audit, label_counts, label_shares, masks_at, nodata_agreement,
    pick_audit_chips, pooled_labels, read_rows, summary_markdown, write_rows,
)


class ThresholdDetector:
    """Probability = B2 reflectance; mask = prob > threshold (no smoothing, no dilation)."""

    def get_cloud_probability_maps(self, data):
        return data[..., 1].copy()

    def get_mask_from_prob(self, probs, threshold=0.4):
        return (probs > threshold).astype(np.uint8)


def test_label_counts_split_cloudy_and_clear():
    label = np.array([[-1, -1, 0, 1],
                      [-1, 0, 0, 1],
                      [1, 1, 0, -1]])
    cloud = np.array([[1, 1, 1, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1]], bool)
    valid = np.ones_like(cloud)
    valid[1, 0] = False  # no valid S2 there: counted on neither side
    c = label_counts(label, cloud, valid)
    assert c["cloudy"] == {"nodata": 2 + 1, "land": 1, "water": 1}  # (0,0),(0,1),(2,3) / (0,2) / (2,0)
    assert c["clear"] == {"nodata": 0, "land": 3, "water": 3}
    assert sum(c["cloudy"].values()) + sum(c["clear"].values()) == valid.sum()
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        label_counts(np.full((2, 2), 2), np.zeros((2, 2), bool), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="must match"):
        label_counts(label, cloud[:2], valid)


def test_label_shares_and_pooling_per_group():
    rows = [{"chip_id": "a", "split": "test", "t0.4_cloudy_nodata": 6, "t0.4_cloudy_land": 3, "t0.4_cloudy_water": 1,
             "t0.4_clear_nodata": 0, "t0.4_clear_land": 70, "t0.4_clear_water": 20},
            {"chip_id": "b", "split": "test", "t0.4_cloudy_nodata": 2, "t0.4_cloudy_land": 2, "t0.4_cloudy_water": 0,
             "t0.4_clear_nodata": 4, "t0.4_clear_land": 6, "t0.4_clear_water": 0},
            {"chip_id": "c", "split": "bolivia", "t0.4_cloudy_nodata": 0, "t0.4_cloudy_land": 0, "t0.4_cloudy_water": 0,
             "t0.4_clear_nodata": 0, "t0.4_clear_land": 1, "t0.4_clear_water": 0}]
    acc = pooled_labels(rows, 0.4, lambda r: r["split"])
    assert acc["test"]["chips"] == 2 and acc["test"]["cloudy"] == {"nodata": 8, "land": 5, "water": 1}
    s = label_shares(acc["test"])
    assert s["cloudy_nodata"] == pytest.approx(8 / 14) and s["cloudy_water"] == pytest.approx(1 / 14)
    assert s["clear_nodata"] == pytest.approx(4 / 100) and s["clear_land"] == pytest.approx(76 / 100)
    assert s["cloudy_of_valid"] == pytest.approx(14 / 114)
    assert s["scored_under_cloud"] == pytest.approx(6 / (6 + 96))  # labelled 0/1 pixels only
    sb = label_shares(acc["bolivia"])
    assert math.isnan(sb["cloudy_nodata"]) and sb["scored_under_cloud"] == 0.0  # no cloudy pixel: NaN, not 0
    assert set(pooled_labels(rows, 0.4, lambda r: None)) == set()


def test_nodata_agreement_uses_train_only():
    def row(split, cn, cl, kn):
        return {"chip_id": split, "split": split, "t0.5_cloudy_nodata": cn, "t0.5_cloudy_land": cl,
                "t0.5_cloudy_water": 0, "t0.5_clear_nodata": kn, "t0.5_clear_land": 50, "t0.5_clear_water": 5}

    rows = [row("train", 30, 10, 20), row("test", 1000, 0, 0)]
    a = nodata_agreement(rows, 0.5)
    assert a["chips"] == 1 and a["precision"] == pytest.approx(30 / 40) and a["recall"] == pytest.approx(30 / 50)
    assert a["f1"] == pytest.approx(2 * 30 / (60 + 10 + 20))
    assert nodata_agreement(rows, 0.5, split="valid")["chips"] == 0


def test_chip_audit_thresholds_with_fake_detector():
    s2 = np.full((13, 4, 4), 1000.0, np.float32)
    s2[1] = np.array([[3000, 3000, 5500, 5500],
                      [3000, 4500, 5500, 5500],
                      [1000, 4500, 7000, 7000],
                      [1000, 1000, 1000, 1000]], np.float32)  # B2 / 10000 = probability
    s2[2:4] = 0.0  # B3, B4 dark: mean visible reflectance = B2 / 3
    s2[:, 3, 3] = np.nan  # nodata: never cloud, never counted
    label = np.array([[-1, -1, -1, -1], [0, 0, -1, -1], [1, 0, -1, 0], [1, 0, 0, 0]])
    row = chip_audit(ThresholdDetector(), s2, label, (0.2, 0.4, 0.5, 0.6))
    assert row["valid_s2_fraction"] == pytest.approx(15 / 16)
    # mask: >0.4 -> 4500, 5500, 7000 pixels (8); >0.5 -> 5500, 7000 (6); >0.6 -> 7000 (2)
    assert [row[f"t{t:g}_cloud_fraction"] for t in (0.4, 0.5, 0.6)] == pytest.approx([8 / 15, 6 / 15, 2 / 15])
    assert (row["t0.4_cloudy_nodata"], row["t0.4_cloudy_land"], row["t0.4_cloudy_water"]) == (5, 3, 0)
    assert (row["t0.6_cloudy_nodata"], row["t0.6_cloudy_land"]) == (1, 1)
    for t in (0.4, 0.5, 0.6):
        n = sum(row[f"t{t:g}_{side}_{k}"] for side in ("cloudy", "clear") for k in LABEL_KEYS)
        assert n == 15
        assert row[f"t{t:g}_mask_lowprob"] == 0.0  # no smoothing in the fake: every mask pixel is above t
    assert row["t0.4_mask_dim"] == 0.0  # mean B2-B4 >= 0.15 for every pixel above 0.4
    assert row["t0.2_mask_dim"] == pytest.approx(3 / 11)  # the three 3000-DN pixels (0.10) are dim
    prob = np.zeros((2, 2), np.float32)
    assert not any(m.any() for m in masks_at(ThresholdDetector(), prob, np.ones((2, 2), bool)).values())


def test_pick_chips_bins_and_csv_roundtrip(tmp_path):
    fr = {"t1": 0.99, "t2": 0.95, "t3": 0.90, "m1": 0.47, "m2": 0.40, "c1": 0.0, "Paraguay_36146": 0.02}
    rows = [{"chip_id": c, "split": "train", "valid_s2_fraction": 1.0, "t0.4_cloud_fraction": f} for c, f in fr.items()]
    rows += [{"chip_id": "x", "split": "train", "valid_s2_fraction": 0.0, "t0.4_cloud_fraction": math.nan},
             {"chip_id": "v", "split": "test", "valid_s2_fraction": 1.0, "t0.4_cloud_fraction": 0.99}]
    cloudy, mid = pick_audit_chips(rows, n_cloudy=2, n_mid=2)
    assert cloudy == ["t1", "t2"] and mid == ["m1", "m2", "Paraguay_36146"]  # named chip added; test chips never
    counts = bin_counts_at(rows, 0.4)
    assert counts["test"]["60-100%"] == 1 and sum(counts["bolivia"].values()) == 0
    full = []
    for r in rows:
        r = dict(r)
        for side in ("cloudy", "clear"):
            for k in LABEL_KEYS:
                r[f"t0.4_{side}_{k}"] = 1
        r["t0.4_mask_lowprob"] = r["t0.4_mask_dim"] = math.nan if r["chip_id"] == "x" else 0.25
        full.append(r)
    back = read_rows(write_rows(full, tmp_path / "a.csv"))
    assert back[0]["t0.4_cloud_fraction"] == pytest.approx(0.99) and back[0]["t0.4_clear_water"] == 1
    assert math.isnan(back[7]["t0.4_cloud_fraction"])
    md = summary_markdown(back, (0.4,), reference={"t1": 0.99})
    assert "Label check at threshold 0.4" in md and "max abs difference 0.00e+00" in md and "F1" in md
