"""Confusion-matrix metrics for binary water segmentation (primary metric: water IoU).

Counts (TP, FP, FN, TN for the water class) are accumulated over every pixel of every
batch, and metrics are computed once from the totals. Per-batch or per-chip IoUs are
never averaged into the headline number: IoU is a ratio, and averaging ratios gives a
chip with 10 water pixels the same weight as one with 100,000, makes the result depend
on batch size/order, and needs an arbitrary value for chips without water. Pooled
counts give the IoU of the whole split treated as one segmentation, independent of
batching.

Definitions and edge cases:
  * Pixels whose target is IGNORE_INDEX (-1) are ignored completely, whatever the
    prediction there.
  * Predictions: logits (N, 2, H, W) -> argmax over the class dim (a tie gives class 0,
    not water); or labels (N, H, W) with values in {0, 1} (integer or bool dtype).
  * A batch where every target is -1 adds nothing to the totals. Its chips are still
    recorded per chip, with zero counts and NaN metrics.
  * A metric whose denominator is 0 is NaN, never 0 or 1:
      iou       = TP / (TP + FP + FN)   NaN if no water in label and prediction
      precision = TP / (TP + FP)        NaN if no water predicted
      recall    = TP / (TP + FN)        NaN if no water in label
      f1        = 2TP / (2TP + FP + FN) NaN if no water in label and prediction
      accuracy  = (TP + TN) / valid     NaN if no valid pixel
    So a chip with no water in label and prediction has per-chip IoU NaN: it is excluded
    from per-chip means, but its counts (TN) still go into the global totals. A chip with
    water predicted but none in the label (or vice versa) has IoU 0 and is included.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from .chips import IGNORE_INDEX

COUNT_KEYS = ("tp", "fp", "fn", "tn")


def _div(num: int, den: int) -> float:
    return num / den if den else math.nan


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Water IoU, F1, precision, recall and accuracy from confusion counts (see module doc)."""
    return {
        "iou": _div(tp, tp + fp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "accuracy": _div(tp + tn, tp + fp + fn + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_valid": tp + fp + fn + tn,
    }


def _to_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def _predicted_labels(pred: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if pred.dim() == 4:
        if pred.shape[1] != 2:
            raise ValueError(f"logits must have shape (N, 2, H, W), got {tuple(pred.shape)}")
        if not pred.is_floating_point():
            raise ValueError("4-D predictions are treated as logits and must be floating point")
        labels = pred.argmax(dim=1)
    elif pred.dim() == 3:
        if pred.is_floating_point():
            raise ValueError("(N, H, W) predictions must be integer or bool labels in {0, 1}; "
                             "threshold probabilities or pass logits (N, 2, H, W)")
        labels = pred.long()
        bad = labels[(labels != 0) & (labels != 1)]
        if bad.numel():
            raise ValueError(f"predicted labels must be 0 or 1, found {bad.unique().tolist()}")
    else:
        raise ValueError(f"predictions must be (N, 2, H, W) logits or (N, H, W) labels, "
                         f"got {tuple(pred.shape)}")
    if labels.shape != target_shape:
        raise ValueError(f"prediction shape {tuple(labels.shape)} does not match target {tuple(target_shape)}")
    return labels


class WaterConfusion:
    """Accumulates water-class confusion counts over batches, globally and per chip.

    >>> m = WaterConfusion()
    >>> for batch in loader:                                  # doctest: +SKIP
    ...     m.update(model(x), batch["label"], batch["id"])
    >>> m.compute()["iou"]                                    # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self._chips: dict[str, dict[str, int]] = {}

    @torch.no_grad()
    def update(self, pred, target, chip_ids: Sequence[str] | None = None) -> None:
        """Add one batch. ``pred``: logits (N, 2, H, W) or labels (N, H, W); ``target``:
        (N, H, W) with values in {-1, 0, 1}; ``chip_ids``: optional N ids for per-chip
        results (each chip id may be added only once)."""
        target = _to_tensor(target)
        if target.dim() != 3:
            raise ValueError(f"target must have shape (N, H, W), got {tuple(target.shape)}")
        target = target.long()
        bad = target[(target != IGNORE_INDEX) & (target != 0) & (target != 1)]
        if bad.numel():
            raise ValueError(f"target values must be -1, 0 or 1, found {bad.unique().tolist()}")
        labels = _predicted_labels(_to_tensor(pred).to(target.device), target.shape)

        valid = target != IGNORE_INDEX
        t = (target == 1) & valid
        p = (labels == 1) & valid
        per_chip = torch.stack([(p & t), (p & ~t), (~p & t), (~p & ~t & valid)])  # tp, fp, fn, tn
        counts = per_chip.flatten(2).sum(dim=2).cpu().tolist()  # 4 lists of N ints

        if chip_ids is not None:
            chip_ids = list(chip_ids)
            if len(chip_ids) != target.shape[0]:
                raise ValueError(f"got {len(chip_ids)} chip ids for a batch of {target.shape[0]}")
            dup = [c for c in chip_ids if c in self._chips] + \
                  [c for i, c in enumerate(chip_ids) if c in chip_ids[:i]]
            if dup:
                raise ValueError(f"chip id(s) already added: {sorted(set(dup))}")
            for i, cid in enumerate(chip_ids):
                self._chips[cid] = {k: counts[j][i] for j, k in enumerate(COUNT_KEYS)}

        self.tp += sum(counts[0])
        self.fp += sum(counts[1])
        self.fn += sum(counts[2])
        self.tn += sum(counts[3])

    def compute(self) -> dict:
        """Metrics from the pooled counts of everything added so far."""
        return metrics_from_counts(self.tp, self.fp, self.fn, self.tn)

    def per_chip(self) -> dict[str, dict]:
        """{chip_id: metrics and counts of that chip}, in the order chips were added."""
        return {cid: metrics_from_counts(**c) for cid, c in self._chips.items()}

    def mean_chip_iou(self) -> float:
        """Mean of per-chip IoUs, excluding NaN chips (no water in label and prediction).
        Secondary diagnostic only; the primary metric is ``compute()["iou"]``."""
        ious = [r["iou"] for r in self.per_chip().values() if not math.isnan(r["iou"])]
        return sum(ious) / len(ious) if ious else math.nan

    def compute_for(self, chip_ids: Iterable[str]) -> dict:
        """Pooled metrics over a subset of chips (e.g. one cloud-cover bin). Needs chip
        ids in ``update``; unknown ids raise KeyError."""
        ids = list(chip_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chip ids would be counted twice")
        unknown = [c for c in ids if c not in self._chips]
        if unknown:
            raise KeyError(f"no per-chip results for {unknown[:10]}")
        totals = {k: sum(self._chips[c][k] for c in ids) for k in COUNT_KEYS}
        return {**metrics_from_counts(**totals), "n_chips": len(ids)}
