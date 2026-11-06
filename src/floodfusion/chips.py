"""Torch-free access to Sen1Floods11 chips: paths, split lists, raster I/O, missing-value rules.

Kept free of torch on purpose: on macOS, torch bundles its own OpenMP runtime, and
LightGBM (used by s2cloudless in floodfusion.clouds) loads Homebrew's. Two OpenMP
runtimes in one process abort with "OMP: Error #15". Code that only needs rasters
(stats, cloud cover, figures) imports from here, not from floodfusion.data.

Directory layout expected under ``root``::

    S1Hand/<chip_id>_S1Hand.tif        2 bands (VV, VH), dB, may contain NaN
    S2Hand/<chip_id>_S2Hand.tif        13 bands (B1..B12 incl. B8A), L1C TOA
    LabelHand/<chip_id>_LabelHand.tif  1 band: 1 water, 0 not water, -1 no data
    splits/flood_{split}_data.csv      no header, first column <chip_id>_S1Hand.tif
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio

S1_BANDS: tuple[str, ...] = ("VV", "VH")
S2_BANDS: tuple[str, ...] = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
)
IGNORE_INDEX: int = -1

SPLITS: tuple[str, ...] = ("train", "valid", "test", "bolivia")
MAX_LISTED_MISSING: int = 10

_SUBDIRS = {"s1": ("S1Hand", "S1Hand"), "s2": ("S2Hand", "S2Hand"), "label": ("LabelHand", "LabelHand")}
_S1_SUFFIX = "_S1Hand.tif"


def chip_path(root: str | Path, chip_id: str, kind: str) -> Path:
    """Path of the ``kind`` ('s1', 's2' or 'label') GeoTIFF for ``chip_id``."""
    subdir, tag = _SUBDIRS[kind]
    return Path(root) / subdir / f"{chip_id}_{tag}.tif"


def read_split_ids(csv_path: str | Path) -> list[str]:
    """Chip ids listed in a split CSV (no header, CRLF-tolerant, blank rows skipped)."""
    ids: list[str] = []
    # newline="" lets the csv module handle \r\n; strip() catches any stray \r.
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            if not name.endswith(_S1_SUFFIX):
                raise ValueError(
                    f"{csv_path}: expected first column '<chip_id>{_S1_SUFFIX}', got {name!r}"
                )
            ids.append(name[: -len(_S1_SUFFIX)])
    return ids


def list_split_chips(root: str | Path, split: str, allow_missing: bool = False) -> list[str]:
    """Chip ids of ``split`` that have S1, S2 and label files.

    Every chip must have all three files, even if only one modality is used, so that
    SAR-only / optical-only / fusion runs (and stats, cloud cover) see identical chips.
    allow_missing=False raises listing up to MAX_LISTED_MISSING missing files;
    allow_missing=True drops such chips and prints how many were kept (local sample only).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root)
    csv_path = root / "splits" / f"flood_{split}_data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split file not found: {csv_path}")
    all_ids = read_split_ids(csv_path)

    missing: list[Path] = []
    kept: list[str] = []
    for cid in all_ids:
        chip_missing = [p for k in ("s1", "s2", "label") if not (p := chip_path(root, cid, k)).is_file()]
        if chip_missing:
            missing.extend(chip_missing)
        else:
            kept.append(cid)

    if missing and not allow_missing:
        listed = "\n  ".join(str(p) for p in missing[:MAX_LISTED_MISSING])
        more = len(missing) - MAX_LISTED_MISSING
        tail = f"\n  ... and {more} more" if more > 0 else ""
        raise FileNotFoundError(
            f"Sen1Floods11 split '{split}': {len(missing)} file(s) missing for "
            f"{len(all_ids) - len(kept)} of {len(all_ids)} chips:\n  {listed}{tail}\n"
            "Pass allow_missing=True only for a partial local sample."
        )
    if allow_missing:
        print(f"Sen1Floods11[{split}]: kept {len(kept)} of {len(all_ids)} chips "
              f"({len(all_ids) - len(kept)} dropped for missing files)")
    return kept


def read_float32(path: str | Path, indexes: Sequence[int] | None = None) -> np.ndarray:
    """Read bands (0-based ``indexes``, default all) as float32 of shape (C, H, W)."""
    with rasterio.open(path) as src:
        idx = list(range(1, src.count + 1)) if indexes is None else [i + 1 for i in indexes]
        return src.read(indexes=idx, out_dtype="float32")


def mark_missing(x: np.ndarray, modality: str) -> np.ndarray:
    """Set missing values to NaN in place and return ``x`` (all bands, shape (C, H, W)).

    Missing = any non-finite value; for S2 also every pixel where ALL bands are 0
    (the S2 nodata value). ±inf counts as missing so it is never clipped to a limit.
    Must be applied to the full band stack, before band selection.
    """
    if modality == "s2":
        if x.shape[0] != len(S2_BANDS):
            raise ValueError(f"S2 nodata rule needs all {len(S2_BANDS)} bands, got {x.shape[0]}")
        x[:, (x == 0).all(axis=0)] = np.nan
    elif modality != "s1":
        raise ValueError(f"modality must be 's1' or 's2', got {modality!r}")
    x[~np.isfinite(x)] = np.nan
    return x


def load_raw(root: str | Path, chip_id: str, modality: str) -> np.ndarray:
    """All bands of one chip's S1 or S2 as float32 (C, H, W), missing values as NaN,
    not clipped or normalised. Shared by the Dataset, stats, cloud cover and figures."""
    return mark_missing(read_float32(chip_path(root, chip_id, modality)), modality)


def load_label(root: str | Path, chip_id: str) -> np.ndarray:
    """Label of one chip as int64 (H, W); NaN -> IGNORE_INDEX."""
    y = read_float32(chip_path(root, chip_id, "label"))[0]
    y[np.isnan(y)] = IGNORE_INDEX
    return y.astype(np.int64)
