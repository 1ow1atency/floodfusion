"""Sen1Floods11 (v1.1, hand-labeled subset) as a PyTorch Dataset.

Directory layout expected under ``root``::

    S1Hand/<chip_id>_S1Hand.tif        2 bands (VV, VH), dB, may contain NaN
    S2Hand/<chip_id>_S2Hand.tif        13 bands (B1..B12 incl. B8A), L1C TOA
    LabelHand/<chip_id>_LabelHand.tif  1 band: 1 water, 0 not water, -1 no data
    splits/flood_{split}_data.csv      no header, first column <chip_id>_S1Hand.tif

Missing-value policy (inputs never reach the model with NaN):
  * Labels: NaN -> IGNORE_INDEX (-1), so those pixels are ignored by loss and metrics.
  * Inputs, in this order:
      1. Mark missing as NaN: any non-finite S1/S2 value, and every S2 pixel where
         ALL 13 bands are 0 (the S2 nodata value). A 0 in only some bands is kept.
         This is decided on all 13 bands, before band selection.
      2. If ``stats`` is given: clip each band to [clip_min, clip_max], then
         normalise with (x - mean) / std. NaN propagates through both steps.
      3. Set every NaN to 0. After normalisation 0 is the train-split band mean,
         i.e. a neutral value carrying no signal.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

S1_BANDS: tuple[str, ...] = ("VV", "VH")
S2_BANDS: tuple[str, ...] = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
)
IGNORE_INDEX: int = -1

SPLITS: tuple[str, ...] = ("train", "valid", "test", "bolivia")
MODALITIES: tuple[str, ...] = ("s1", "s2")
MAX_LISTED_MISSING: int = 10
_STATS_KEYS: tuple[str, ...] = ("mean", "std", "clip_min", "clip_max")

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


def _band_indices(selected: Sequence[str], all_bands: Sequence[str]) -> list[int]:
    unknown = [b for b in selected if b not in all_bands]
    if unknown:
        raise ValueError(f"Unknown band(s) {unknown}; valid names are {list(all_bands)}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate band names in {list(selected)}")
    return [all_bands.index(b) for b in selected]


def _read_float32(path: Path, indexes: Sequence[int] | None = None) -> np.ndarray:
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
    not clipped or normalised. Shared by the Dataset and the stats computation."""
    return mark_missing(_read_float32(chip_path(root, chip_id, modality)), modality)


def load_label(root: str | Path, chip_id: str) -> np.ndarray:
    """Label of one chip as int64 (H, W); NaN -> IGNORE_INDEX."""
    y = _read_float32(chip_path(root, chip_id, "label"))[0]
    y[np.isnan(y)] = IGNORE_INDEX
    return y.astype(np.int64)


class Sen1Floods11(Dataset):
    """Sen1Floods11 hand-labeled chips.

    Args:
        root: dataset root containing S1Hand/, S2Hand/, LabelHand/, splits/.
        split: one of "train", "valid", "test", "bolivia".
        modalities: subset of ("s1", "s2") to return.
        s2_bands: optional S2 band names to keep, in the order given (default: all 13).
        stats: optional per-band clip limits and normalisation statistics, computed
            on the train split:
            ``{"s1": {"mean": [2], "std": [2], "clip_min": [2], "clip_max": [2]},
               "s2": {"mean": [13], "std": [13], "clip_min": [13], "clip_max": [13]}}``
            Lists are always over the *full* band lists (S1_BANDS / S2_BANDS);
            the entries for the selected S2 bands are picked automatically.
            Each band is clipped to [clip_min, clip_max] before normalising.
        transform: optional callable applied to the output sample dict.
        allow_missing: if False (default) raise when any chip lacks a file; if True
            drop such chips and print how many were kept. Only for partial local
            samples; keep False on the full dataset.

    Each item is ``{"id": str, "s1": (2,H,W) float32, "s2": (C,H,W) float32,
    "label": (H,W) int64}``, with only the requested modalities present.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        modalities: Sequence[str] = ("s1", "s2"),
        s2_bands: Sequence[str] | None = None,
        stats: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        transform: Callable[[dict], dict] | None = None,
        allow_missing: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        modalities = tuple(modalities)
        if not modalities or any(m not in MODALITIES for m in modalities):
            raise ValueError(f"modalities must be a non-empty subset of {MODALITIES}, got {modalities}")
        if len(set(modalities)) != len(modalities):
            raise ValueError(f"Duplicate modalities in {modalities}")

        self.root = Path(root)
        self.split = split
        self.modalities = modalities
        self.transform = transform

        self.s2_bands = tuple(S2_BANDS if s2_bands is None else s2_bands)
        self._s2_idx = _band_indices(self.s2_bands, S2_BANDS)
        self._s1_idx = list(range(len(S1_BANDS)))

        self._norm = self._prepare_stats(stats) if stats is not None else None

        csv_path = self.root / "splits" / f"flood_{split}_data.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        all_ids = read_split_ids(csv_path)

        # Every chip must have S1, S2 and label, even if only one modality is
        # requested, so that SAR-only / optical-only / fusion runs see identical chips.
        missing: list[Path] = []
        kept: list[str] = []
        for cid in all_ids:
            chip_missing = [p for k in ("s1", "s2", "label") if not (p := chip_path(self.root, cid, k)).is_file()]
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

        self.ids: list[str] = kept

    def _prepare_stats(self, stats: Mapping) -> dict[str, dict[str, np.ndarray]]:
        """Validate stats and return per-modality arrays of shape (C_selected, 1, 1)."""
        out = {}
        for mod in self.modalities:
            if mod not in stats:
                raise KeyError(f"stats has no entry for modality {mod!r}")
            full = S1_BANDS if mod == "s1" else S2_BANDS
            idx = self._s1_idx if mod == "s1" else self._s2_idx
            arrs = {}
            for key in _STATS_KEYS:
                if key not in stats[mod]:
                    raise KeyError(f"stats[{mod!r}] is missing {key!r}; required keys: {_STATS_KEYS}")
                a = np.asarray(stats[mod][key], dtype=np.float32)
                if a.shape != (len(full),):
                    raise ValueError(
                        f"stats[{mod!r}][{key!r}] must have {len(full)} values "
                        f"(one per band in {full}), got shape {a.shape}"
                    )
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"stats[{mod!r}][{key!r}] must be finite")
                arrs[key] = a
            if not np.all(arrs["std"] > 0):
                raise ValueError(f"stats[{mod!r}]['std'] must be > 0 for every band")
            bad = [full[b] for b in range(len(full)) if not arrs["clip_min"][b] < arrs["clip_max"][b]]
            if bad:
                raise ValueError(f"stats[{mod!r}]: clip_min must be < clip_max, violated for bands {bad}")
            out[mod] = {k: v[idx, None, None] for k, v in arrs.items()}
        return out

    def __len__(self) -> int:
        return len(self.ids)

    def _load_input(self, cid: str, mod: str) -> np.ndarray:
        x = load_raw(self.root, cid, mod)
        x = x[self._s2_idx] if mod == "s2" else x[self._s1_idx]

        if self._norm is not None:
            s = self._norm[mod]
            x = np.clip(x, s["clip_min"], s["clip_max"])  # NaN stays NaN
            x = (x - s["mean"]) / s["std"]
        # Missing -> 0: the train-split band mean after normalisation.
        x[np.isnan(x)] = 0.0
        return x

    def __getitem__(self, i: int) -> dict:
        cid = self.ids[i]
        sample: dict = {"id": cid}
        for mod in self.modalities:
            sample[mod] = torch.from_numpy(self._load_input(cid, mod))
        sample["label"] = torch.from_numpy(load_label(self.root, cid))

        h, w = sample["label"].shape
        for mod in self.modalities:
            if tuple(sample[mod].shape[1:]) != (h, w):
                raise ValueError(
                    f"Chip {cid}: {mod} has spatial shape {tuple(sample[mod].shape[1:])}, label has {(h, w)}"
                )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
