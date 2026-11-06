"""Tests for floodfusion.metrics with small hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from floodfusion.metrics import WaterConfusion, metrics_from_counts

nan = math.nan


def t(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def logits_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """(N, H, W) {0,1} -> (N, 2, H, W) logits whose argmax is ``labels``."""
    lab = labels.float()
    return torch.stack([1 - lab, lab], dim=1) * 3.0 - 1.0


# Chip "basic", 2x3:            target          pred
#                               [[ 1, 1, 0],    [[1, 0, 0],
#                                [ 0,-1, 1]]     [1, 1, 1]]
# (0,0) TP  (0,1) FN  (0,2) TN  (1,0) FP  (1,1) ignored (pred 1 not counted)  (1,2) TP
# TP=2 FP=1 FN=1 TN=1 -> IoU 2/4, precision 2/3, recall 2/3, F1 4/6, accuracy 3/5.
BASIC_T = t([[[1, 1, 0], [0, -1, 1]]])
BASIC_P = t([[[1, 0, 0], [1, 1, 1]]])
BASIC = {"tp": 2, "fp": 1, "fn": 1, "tn": 1, "n_valid": 5,
         "iou": 0.5, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "accuracy": 0.6}


def assert_metrics(got: dict, expected: dict):
    for k, v in expected.items():
        if isinstance(v, float) and math.isnan(v):
            assert math.isnan(got[k]), (k, got[k])
        else:
            assert got[k] == pytest.approx(v), (k, got[k], v)


def test_hand_computed_labels():
    m = WaterConfusion()
    m.update(BASIC_P, BASIC_T, ["basic"])
    assert_metrics(m.compute(), BASIC)
    assert_metrics(m.per_chip()["basic"], BASIC)


def test_logits_match_labels_and_tie_is_not_water():
    m = WaterConfusion()
    m.update(logits_from_labels(BASIC_P), BASIC_T)
    assert_metrics(m.compute(), BASIC)
    tie = torch.zeros(1, 2, 1, 2)          # equal logits -> argmax 0 -> not water
    m2 = WaterConfusion()
    m2.update(tie, t([[[1, 0]]]))
    assert_metrics(m2.compute(), {"tp": 0, "fn": 1, "tn": 1, "fp": 0})


def test_numpy_and_bool_inputs():
    m = WaterConfusion()
    m.update(BASIC_P.numpy().astype(bool), BASIC_T.numpy())
    assert_metrics(m.compute(), BASIC)


# Pooling vs averaging. Chip "small": target [1,0,0,0], pred [1,0,0,0] -> TP1 TN3, IoU 1.
# Chip "big":   target [1,1,1,1], pred [1,0,0,0] -> TP1 FN3, IoU 1/4.
# Pooled: TP2 FN3 TN3 -> IoU 2/5 = 0.4.  Mean of chip IoUs: (1 + 0.25)/2 = 0.625 (biased up).
SMALL_T, SMALL_P = t([[[1, 0, 0, 0]]]), t([[[1, 0, 0, 0]]])
BIG_T, BIG_P = t([[[1, 1, 1, 1]]]), t([[[1, 0, 0, 0]]])


def test_counts_are_pooled_across_batches_not_averaged():
    two_batches = WaterConfusion()
    two_batches.update(SMALL_P, SMALL_T, ["small"])
    two_batches.update(BIG_P, BIG_T, ["big"])
    one_batch = WaterConfusion()
    one_batch.update(torch.cat([SMALL_P, BIG_P]), torch.cat([SMALL_T, BIG_T]), ["small", "big"])
    for m in (two_batches, one_batch):
        assert_metrics(m.compute(), {"tp": 2, "fp": 0, "fn": 3, "tn": 3, "iou": 0.4,
                                     "precision": 1.0, "recall": 0.4, "accuracy": 5 / 8})
        assert m.mean_chip_iou() == pytest.approx(0.625)
        assert m.per_chip()["small"]["iou"] == 1.0 and m.per_chip()["big"]["iou"] == 0.25


def test_batch_with_only_ignored_pixels():
    m = WaterConfusion()
    m.update(t([[[1, 1], [0, 0]]]), t([[[-1, -1], [-1, -1]]]), ["empty"])
    # Nothing counted; every metric undefined, recorded per chip with zero counts.
    nan_all = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_valid": 0,
               "iou": nan, "f1": nan, "precision": nan, "recall": nan, "accuracy": nan}
    assert_metrics(m.compute(), nan_all)
    assert_metrics(m.per_chip()["empty"], nan_all)
    assert math.isnan(m.mean_chip_iou())
    # Adding it to real data changes nothing in the totals.
    m.update(BASIC_P, BASIC_T, ["basic"])
    assert_metrics(m.compute(), BASIC)
    assert m.mean_chip_iou() == pytest.approx(0.5)


def test_chip_without_water_is_nan_but_counts_go_into_totals():
    # Chip "dry": target [0,0,0,-1], pred [0,0,0,1] -> TN3 (last pixel ignored), IoU NaN.
    m = WaterConfusion()
    m.update(SMALL_P, SMALL_T, ["small"])            # TP1 TN3, IoU 1
    m.update(t([[[0, 0, 0, 1]]]), t([[[0, 0, 0, -1]]]), ["dry"])
    dry = m.per_chip()["dry"]
    assert_metrics(dry, {"tp": 0, "fp": 0, "fn": 0, "tn": 3, "iou": nan, "f1": nan,
                         "precision": nan, "recall": nan, "accuracy": 1.0})
    assert m.mean_chip_iou() == pytest.approx(1.0)  # "dry" excluded from the mean
    assert_metrics(m.compute(), {"tp": 1, "tn": 6, "n_valid": 7, "iou": 1.0, "accuracy": 1.0})


def test_false_alarm_and_missed_water_chips_have_iou_zero():
    m = WaterConfusion()
    m.update(t([[[1, 0]]]), t([[[0, 0]]]), ["false_alarm"])   # FP1 TN1
    m.update(t([[[0, 0]]]), t([[[1, 0]]]), ["missed"])        # FN1 TN1
    fa, miss = m.per_chip()["false_alarm"], m.per_chip()["missed"]
    assert fa["iou"] == 0.0 and math.isnan(fa["recall"]) and fa["precision"] == 0.0
    assert miss["iou"] == 0.0 and math.isnan(miss["precision"]) and miss["recall"] == 0.0
    assert m.mean_chip_iou() == 0.0                          # both included
    assert_metrics(m.compute(), {"tp": 0, "fp": 1, "fn": 1, "tn": 2, "iou": 0.0, "f1": 0.0})


def test_compute_for_subset_pools_counts():
    m = WaterConfusion()
    m.update(torch.cat([SMALL_P, BIG_P]), torch.cat([SMALL_T, BIG_T]), ["small", "big"])
    m.update(BASIC_P, BASIC_T, ["basic"])
    sub = m.compute_for(["big", "basic"])                      # TP3 FP1 FN4 TN1
    assert_metrics(sub, {"tp": 3, "fp": 1, "fn": 4, "tn": 1, "iou": 3 / 8, "n_chips": 2})
    assert_metrics(m.compute_for(["small", "big", "basic"]), m.compute())
    assert_metrics(m.compute_for([]), {"n_chips": 0, "iou": nan})
    with pytest.raises(KeyError):
        m.compute_for(["unknown"])
    with pytest.raises(ValueError):
        m.compute_for(["big", "big"])


def test_reset():
    m = WaterConfusion()
    m.update(BASIC_P, BASIC_T, ["basic"])
    m.reset()
    assert m.compute()["n_valid"] == 0 and m.per_chip() == {}
    m.update(BASIC_P, BASIC_T, ["basic"])  # same id allowed again after reset
    assert_metrics(m.compute(), BASIC)


@pytest.mark.parametrize("pred, target, ids, match", [
    (t([[[2, 0]]]), t([[[1, 0]]]), None, "predicted labels must be 0 or 1"),
    (torch.tensor([[[0.7, 0.2]]]), t([[[1, 0]]]), None, "integer or bool"),
    (torch.zeros(1, 3, 1, 2), t([[[1, 0]]]), None, r"\(N, 2, H, W\)"),
    (t([[[1, 0, 0]]]), t([[[1, 0]]]), None, "does not match"),
    (t([[[1, 0]]]), t([[[1, 5]]]), None, "target values"),
    (t([[1, 0]]), t([[1, 0]]), None, "target must have shape"),
    (t([[[1, 0]]]), t([[[1, 0]]]), ["a", "b"], "2 chip ids for a batch of 1"),
    (t([[[1]], [[0]]]), t([[[1]], [[0]]]), ["a", "a"], "already added"),
])
def test_invalid_inputs_raise(pred, target, ids, match):
    with pytest.raises(ValueError, match=match):
        WaterConfusion().update(pred, target, ids)


def test_duplicate_chip_across_batches_raises():
    m = WaterConfusion()
    m.update(BASIC_P, BASIC_T, ["basic"])
    with pytest.raises(ValueError, match="already added"):
        m.update(BASIC_P, BASIC_T, ["basic"])
    assert_metrics(m.compute(), BASIC)  # failed update did not change the totals


def test_metrics_from_counts_direct():
    assert_metrics(metrics_from_counts(0, 0, 0, 0), {"iou": nan, "accuracy": nan, "n_valid": 0})
    assert metrics_from_counts(5, 0, 0, 0)["iou"] == 1.0
    big = 10**12  # python ints: no overflow on large totals
    assert metrics_from_counts(big, big, 0, 0)["iou"] == 0.5
