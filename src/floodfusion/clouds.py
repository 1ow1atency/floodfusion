"""Per-chip cloud fraction of the Sentinel-2 image with s2cloudless (evaluation only).

Method (fixed, so cloud-level results are reproducible):
  * Input: all 13 L1C bands of the chip, missing pixels from
    :func:`floodfusion.chips.mark_missing` (all 13 bands == 0, or non-finite).
    Valid S2 pixel = no band missing.
  * s2cloudless expects TOA reflectance = DN / 10000 for bands B1..B12 in our order
    (``all_bands=True`` picks B1,B2,B4,B5,B8,B8A,B9,B10,B11,B12 itself). Sen1Floods11
    imagery is from 2016-2019, before processing baseline 04.00 (Jan 2022), so no
    -1000 harmonisation offset is applied.
  * Parameters from the official s2cloudless example for 10 m data (our chips are 10 m):
    threshold 0.4, average_over 4, dilation_size 2.
  * Missing pixels get cloud probability 0 before averaging/dilation, and the final
    mask is restricted to valid pixels, so nodata never counts as cloud.
  * cloud_fraction = cloudy valid pixels / valid pixels (NaN if no valid pixel);
    valid_s2_fraction = valid pixels / all pixels.

This module never imports torch (see floodfusion.chips for why).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from tqdm.auto import tqdm

from .chips import S2_BANDS, SPLITS, list_split_chips, load_raw

REFLECTANCE_SCALE = 10000.0
CLOUD_THRESHOLD = 0.4
AVERAGE_OVER = 4
DILATION_SIZE = 2
CSV_COLUMNS = ("chip_id", "split", "cloud_fraction", "valid_s2_fraction")
# Cloud bins for evaluation: [0, 0.1), [0.1, 0.3), [0.3, 0.6), [0.6, 1.0].
CLOUD_BIN_EDGES: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0)
CLOUD_BIN_LABELS: tuple[str, ...] = ("0-10%", "10-30%", "30-60%", "60-100%")


class CloudDetector(Protocol):
    """The two s2cloudless.S2PixelCloudDetector methods we use (lets tests pass a fake)."""

    def get_cloud_probability_maps(self, data: np.ndarray) -> np.ndarray: ...
    def get_mask_from_prob(self, cloud_probs: np.ndarray) -> np.ndarray: ...


def make_detector() -> CloudDetector:
    """s2cloudless detector with the fixed parameters above."""
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError as e:  # includes LightGBM failing to load libomp on macOS
        raise ImportError(
            "s2cloudless is needed for cloud cover: pip install -e '.[clouds]' "
            "(on macOS also: brew install libomp)"
        ) from e
    return S2PixelCloudDetector(threshold=CLOUD_THRESHOLD, average_over=AVERAGE_OVER,
                                dilation_size=DILATION_SIZE, all_bands=True)


def s2_reflectance(s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From all-13-band S2 (C, H, W) with NaN = missing, return the s2cloudless input
    (H, W, 13) float32 reflectance with missing set to 0, and the valid mask (H, W)."""
    if s2.shape[0] != len(S2_BANDS):
        raise ValueError(f"expected all {len(S2_BANDS)} S2 bands, got {s2.shape[0]}")
    valid = ~np.isnan(s2).any(axis=0)
    refl = np.nan_to_num(s2, nan=0.0).transpose(1, 2, 0) / np.float32(REFLECTANCE_SCALE)
    return refl.astype(np.float32), valid


def cloud_mask(detector: CloudDetector, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean cloud mask (only ever True on valid pixels) and valid mask, both (H, W)."""
    refl, valid = s2_reflectance(s2)
    prob = np.asarray(detector.get_cloud_probability_maps(refl[None]), dtype=np.float32)
    prob[0][~valid] = 0.0
    mask = np.asarray(detector.get_mask_from_prob(prob))[0].astype(bool)
    return mask & valid, valid


def cloud_fractions(mask: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(cloud_fraction, valid_s2_fraction). Cloud pixels outside ``valid`` are ignored;
    cloud_fraction is NaN when there is no valid pixel."""
    if mask.shape != valid.shape:
        raise ValueError(f"mask {mask.shape} and valid {valid.shape} differ in shape")
    n_valid = int(valid.sum())
    cloud = float((mask & valid).sum()) / n_valid if n_valid else math.nan
    return cloud, n_valid / valid.size


def compute_cloud_cover(root: str | Path, *, detector: CloudDetector | None = None,
                        allow_missing: bool = False, progress: bool = True) -> list[dict]:
    """One row per chip of every split: chip_id, split, cloud_fraction, valid_s2_fraction."""
    detector = make_detector() if detector is None else detector
    rows = []
    for split in SPLITS:
        for cid in tqdm(list_split_chips(root, split, allow_missing), desc=f"clouds {split}",
                        disable=not progress):
            mask, valid = cloud_mask(detector, load_raw(root, cid, "s2"))
            cloud, valid_frac = cloud_fractions(mask, valid)
            rows.append({"chip_id": cid, "split": split, "cloud_fraction": cloud,
                         "valid_s2_fraction": valid_frac})
    return rows


def write_cloud_csv(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            cf = r["cloud_fraction"]
            w.writerow([r["chip_id"], r["split"], "" if math.isnan(cf) else f"{cf:.6f}",
                        f"{r['valid_s2_fraction']:.6f}"])
    return path


def read_cloud_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{"chip_id": r["chip_id"], "split": r["split"],
                 "cloud_fraction": float(r["cloud_fraction"]) if r["cloud_fraction"] else math.nan,
                 "valid_s2_fraction": float(r["valid_s2_fraction"])} for r in csv.DictReader(f)]


def cloud_bin(fraction: float) -> str | None:
    """Bin label for a cloud fraction; None for NaN (no valid S2 pixel)."""
    if math.isnan(fraction):
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"cloud fraction out of [0, 1]: {fraction}")
    for hi, label in zip(CLOUD_BIN_EDGES[1:], CLOUD_BIN_LABELS):
        if fraction < hi or hi == CLOUD_BIN_EDGES[-1]:
            return label
    raise AssertionError("unreachable")


def bin_counts(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    """{split: {bin label: n, 'no valid S2': n, 'total': n}} for every split."""
    out = {s: {**{b: 0 for b in CLOUD_BIN_LABELS}, "no valid S2": 0, "total": 0} for s in SPLITS}
    for r in rows:
        b = cloud_bin(r["cloud_fraction"])
        out[r["split"]][b or "no valid S2"] += 1
        out[r["split"]]["total"] += 1
    return out


def format_bin_table(rows: Iterable[dict]) -> str:
    counts = bin_counts(rows)
    cols = list(CLOUD_BIN_LABELS) + ["no valid S2", "total"]
    head = f"{'split':<9}" + "".join(f"{c:>13}" for c in cols)
    lines = [head, "-" * len(head)]
    for s in SPLITS:
        lines.append(f"{s:<9}" + "".join(f"{counts[s][c]:>13}" for c in cols))
    return "\n".join(lines)


def pick_examples(rows: Iterable[dict], split: str = "train", n_cloudy: int = 6,
                  n_clear: int = 3) -> list[dict]:
    """The n_cloudy cloudiest then n_clear clearest chips of ``split`` (ties by chip id)."""
    cand = sorted((r for r in rows if r["split"] == split and not math.isnan(r["cloud_fraction"])),
                  key=lambda r: (r["cloud_fraction"], r["chip_id"]))
    cloudy = cand[::-1][:n_cloudy]
    clear = [r for r in cand if r not in cloudy][:n_clear]
    return cloudy + clear
