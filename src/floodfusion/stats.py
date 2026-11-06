"""Normalisation statistics (train split only) and descriptive label statistics.

Band stats are computed in two passes over the train chips, using only *valid*
pixels as defined by :func:`floodfusion.data.mark_missing` (the same rule the
Dataset applies, so stats and model inputs agree):

1. ``clip_min`` / ``clip_max`` = lower / upper percentile per band. Exact when the
   train split has at most ``max_pixels_per_band`` pixels; otherwise from a
   fixed-seed random subsample of pixels taken with the *same fraction* from every
   chip, so each valid pixel is equally likely to be picked.
2. ``mean`` / ``std`` per band over *all* valid pixels after clipping to the
   pass-1 limits, i.e. exactly what the model sees before normalisation. Streamed
   chip by chip with float64 accumulators (Chan et al. pairwise update), so memory
   does not grow with the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from tqdm.auto import tqdm

from .chips import (
    IGNORE_INDEX, S1_BANDS, S2_BANDS, SPLITS, chip_path, list_split_chips, load_label, load_raw,
)

STATS_SPLIT = "train"  # never anything else: stats from train only (CLAUDE.md)
PERCENTILES: tuple[float, float] = (1.0, 99.0)
# 10M float32 values x 15 bands ~ 600 MB for pass 1. Full train (252 x 512 x 512
# ~ 66M pixels) is subsampled at ~15%; the 25-chip local sample (~5M) is exact.
DEFAULT_MAX_PIXELS_PER_BAND = 10_000_000

_BANDS = {"s1": S1_BANDS, "s2": S2_BANDS}


def _chip_pixels(root: Path, cid: str) -> int:
    with rasterio.open(chip_path(root, cid, "s1")) as src:  # header only
        return src.height * src.width


def compute_band_stats(
    root: str | Path,
    *,
    allow_missing: bool = False,
    seed: int = 0,
    max_pixels_per_band: int = DEFAULT_MAX_PIXELS_PER_BAND,
    percentiles: tuple[float, float] = PERCENTILES,
    verbose: bool = True,
) -> dict:
    """Per-band clip limits, mean and std from the train split, in the format
    ``Sen1Floods11(stats=...)`` expects (plus extra informational keys)."""
    root = Path(root)
    lo_pct, hi_pct = percentiles
    if not 0 <= lo_pct < hi_pct <= 100:
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}")

    ids = list_split_chips(root, STATS_SPLIT, allow_missing)
    if not ids:
        raise ValueError(f"No usable chips in the {STATS_SPLIT} split under {root}")

    total_pixels = sum(_chip_pixels(root, cid) for cid in ids)
    fraction = min(1.0, max_pixels_per_band / total_pixels)
    method = "exact" if fraction == 1.0 else "subsample"
    rng = np.random.default_rng(seed)
    if verbose:
        how = "all pixels" if method == "exact" else f"random {fraction:.2%} of pixels per chip (seed={seed})"
        print(f"Pass 1 (clip limits = p{lo_pct:g}/p{hi_pct:g} percentiles): {method} on {how}; "
              f"{len(ids)} chips, {total_pixels:,} pixels")

    # ---- pass 1: collect (sampled) valid values per band -> percentiles -----------------
    samples = {m: [[] for _ in bands] for m, bands in _BANDS.items()}
    for cid in tqdm(ids, desc="pass 1/2", disable=not verbose):
        mask = None
        for mod, bands in _BANDS.items():
            x = load_raw(root, cid, mod)
            if mask is None and method == "subsample":
                # One pixel mask per chip, shared by S1 and S2.
                mask = rng.random(x.shape[1:]) < fraction
            v = x.reshape(len(bands), -1) if mask is None else x[:, mask]
            for b in range(len(bands)):
                vb = v[b]
                samples[mod][b].append(vb[~np.isnan(vb)])

    out: dict = {}
    for mod, bands in _BANDS.items():
        clip_min, clip_max, n_sampled = [], [], []
        for b, name in enumerate(bands):
            vals = np.concatenate(samples[mod][b])
            samples[mod][b] = None  # free memory band by band
            if vals.size == 0:
                raise ValueError(f"{mod} band {name}: no valid pixels in the {STATS_SPLIT} split")
            lo, hi = np.percentile(vals.astype(np.float64), [lo_pct, hi_pct])
            if not lo < hi:
                raise ValueError(f"{mod} band {name}: percentiles give clip_min={lo} >= clip_max={hi}")
            clip_min.append(float(lo))
            clip_max.append(float(hi))
            n_sampled.append(int(vals.size))
        out[mod] = {"bands": list(bands), "clip_min": clip_min, "clip_max": clip_max,
                    "n_sampled": n_sampled}

    # ---- pass 2: exact mean/std of clipped valid values (streamed) ----------------------
    acc = {m: np.zeros((3, len(bands)), np.float64) for m, bands in _BANDS.items()}  # n, mean, M2
    for cid in tqdm(ids, desc="pass 2/2", disable=not verbose):
        for mod, bands in _BANDS.items():
            x = load_raw(root, cid, mod)
            lo, hi = out[mod]["clip_min"], out[mod]["clip_max"]
            a = acc[mod]
            for b in range(len(bands)):
                vb = x[b][~np.isnan(x[b])]
                if vb.size == 0:
                    continue
                vb = np.clip(vb, np.float32(lo[b]), np.float32(hi[b])).astype(np.float64)
                n_b, mean_b = vb.size, vb.mean()
                m2_b = np.square(vb - mean_b).sum()
                n_a, mean_a, m2_a = a[:, b]
                n = n_a + n_b
                delta = mean_b - mean_a
                a[:, b] = (n, mean_a + delta * n_b / n, m2_a + m2_b + delta**2 * n_a * n_b / n)

    for mod, bands in _BANDS.items():
        n, mean, m2 = acc[mod]
        std = np.sqrt(m2 / n)
        zero = [bands[b] for b in range(len(bands)) if not std[b] > 0]
        if zero:
            raise ValueError(f"{mod}: std is 0 after clipping for bands {zero}")
        out[mod].update(mean=mean.tolist(), std=std.tolist(), n_valid=n.astype(int).tolist())

    out["meta"] = {
        "split": STATS_SPLIT,
        "n_chips": len(ids),
        "total_pixels": total_pixels,
        "allow_missing": allow_missing,
        "seed": seed,
        "percentiles": [lo_pct, hi_pct],
        "percentile_method": method,
        "sample_fraction": fraction,
        "max_pixels_per_band": max_pixels_per_band,
        "valid_pixels": "finite, and for S2 not all 13 bands == 0 (floodfusion.data.mark_missing)",
        "mean_std": "exact over all valid pixels after clipping to [clip_min, clip_max]",
    }
    return out


def compute_label_stats(root: str | Path, *, allow_missing: bool = False) -> dict:
    """Descriptive label counts per split (chips, labelled, water, water fraction).
    Not used for training or model selection."""
    root = Path(root)
    out: dict = {"meta": {"allow_missing": allow_missing,
                          "labelled": f"label != {IGNORE_INDEX}", "water": "label == 1"}}
    for split in SPLITS:
        ids = list_split_chips(root, split, allow_missing)
        labelled = water = 0
        for cid in ids:
            y = load_label(root, cid)
            bad = np.setdiff1d(np.unique(y), [IGNORE_INDEX, 0, 1])
            if bad.size:
                raise ValueError(f"{split}/{cid}: unexpected label values {bad.tolist()}")
            labelled += int((y != IGNORE_INDEX).sum())
            water += int((y == 1).sum())
        out[split] = {"n_chips": len(ids), "labelled_pixels": labelled, "water_pixels": water,
                      "water_fraction": water / labelled if labelled else None}
    return out


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path


def format_band_table(stats: dict) -> str:
    head = f"{'mod':<4}{'band':<6}{'clip_min':>11}{'clip_max':>11}{'mean':>11}{'std':>11}" \
           f"{'n_valid':>13}{'n_sampled':>13}"
    lines = [head, "-" * len(head)]
    for mod in _BANDS:
        s = stats[mod]
        for b, name in enumerate(s["bands"]):
            lines.append(f"{mod:<4}{name:<6}{s['clip_min'][b]:>11.3f}{s['clip_max'][b]:>11.3f}"
                         f"{s['mean'][b]:>11.3f}{s['std'][b]:>11.3f}"
                         f"{s['n_valid'][b]:>13,}{s['n_sampled'][b]:>13,}")
    m = stats["meta"]
    lines.append(f"split={m['split']}  chips={m['n_chips']}  seed={m['seed']}  "
                 f"percentiles={m['percentiles']}  method={m['percentile_method']} "
                 f"(fraction={m['sample_fraction']:.4f})  allow_missing={m['allow_missing']}")
    return "\n".join(lines)


def format_label_table(label_stats: dict) -> str:
    head = f"{'split':<9}{'chips':>7}{'labelled_px':>14}{'water_px':>12}{'water_frac':>12}"
    lines = [head, "-" * len(head)]
    for split in SPLITS:
        s = label_stats[split]
        frac = "n/a" if s["water_fraction"] is None else f"{s['water_fraction']:.4f}"
        lines.append(f"{split:<9}{s['n_chips']:>7}{s['labelled_pixels']:>14,}"
                     f"{s['water_pixels']:>12,}{frac:>12}")
    return "\n".join(lines)
