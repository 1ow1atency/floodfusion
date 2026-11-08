"""How much ground signal survives under the synthetic evaluation clouds (numpy only).

Question: inside a cloud's covered mask, could a model still see water? Measured per chip
on the robustness-evaluation clouds (thick, ``eval_cloud_seed``), for the clouds as trained
("thick") and for the opaque variant (``clouds_synth.opaque_alpha``), always on the same
covered pixels:

  * opacity (alpha) of the covered pixels;
  * per band, the Pearson correlation of clouded vs original S2 inside the mask
    (corr^2 = share of the clouded band's variance inside the cloud explained by the ground);
  * water / land separability inside the mask, from NDWI (B3, B8) and MNDWI (B3, B11):
    mean index of water minus land, and ROC AUC (0.5 = no information, 1 = perfect),
    per pixel and after a k x k mean over covered pixels (a CNN uses context, which averages
    out the cloud texture but not the ground signal). The same statistics on the original
    S2 at the same pixels are the clear-sky reference.

Chips with real clouds are left out by the caller (their "ground" is already cloud).
Never imports torch (see floodfusion.chips).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .chips import IGNORE_INDEX, S2_BANDS, load_label, load_raw
from .clouds_synth import apply_clouds, cloud_params, eval_cloud_seed, opaque_alpha

INDICES = {"NDWI": ("B3", "B8"), "MNDWI": ("B3", "B11")}
CORR_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")  # shown in the summary (all 13 are computed)
MIN_CLASS_PIXELS = 200  # water and land pixels needed under the cloud for an AUC
SMOOTH = 9  # px, context window of the smoothed AUC
ALPHA_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5)
VARIANTS = ("thick", "opaque")


def normalized_difference(s2: np.ndarray, a: str, b: str) -> np.ndarray:
    xa, xb = s2[S2_BANDS.index(a)].astype(np.float64), s2[S2_BANDS.index(b)].astype(np.float64)
    den = xa + xb
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-6, (xa - xb) / den, np.nan)


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    n1, n0 = len(pos), len(neg)
    if not n1 or not n0:
        return math.nan
    x = np.concatenate([pos, neg]).astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    _, first, counts = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)  # average 1-based rank of ties
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def masked_box_mean(x: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Mean of ``x`` over the valid pixels of each k x k window (NaN where not valid)."""
    v = valid & np.isfinite(x)
    p = k // 2

    def box(a):
        c = np.pad(np.pad(a, p).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    s, n = box(np.where(v, x, 0.0)), box(v.astype(np.float64))
    out = np.full(x.shape, np.nan)
    out[v] = s[v] / n[v]
    return out


def chip_leakage(s2: np.ndarray, label: np.ndarray, clouded: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
                 min_pixels: int = MIN_CLASS_PIXELS, smooth: int = SMOOTH) -> dict:
    """Leakage statistics of one clouded chip (raw S2 DN, (13, H, W); mask = covered pixels)."""
    inside = mask & np.isfinite(s2).all(0) & np.isfinite(clouded).all(0)
    out = {"alpha": alpha[inside], "n_inside": int(inside.sum()), "corr": {}, "index": {}}
    for i, band in enumerate(S2_BANDS):
        x, y = s2[i][inside].astype(np.float64), clouded[i][inside].astype(np.float64)
        if len(x) < 3 or x.std() == 0:
            out["corr"][band] = math.nan
        elif y.std() == 0:  # constant under the cloud: no ground information
            out["corr"][band] = 0.0
        else:
            out["corr"][band] = float(np.corrcoef(x, y)[0, 1])
    water, land = inside & (label == 1), inside & (label == 0)
    enough = water.sum() >= min_pixels and land.sum() >= min_pixels
    for name, (a, b) in INDICES.items():
        res = {"n_water": int(water.sum()), "n_land": int(land.sum())}
        for which, img in (("clear", s2), ("cloud", clouded)):
            nd = normalized_difference(img, a, b)
            sm = masked_box_mean(nd, inside, smooth)
            w, l = water & np.isfinite(nd), land & np.isfinite(nd)
            res[which] = ({"diff": float(nd[w].mean() - nd[l].mean()), "auc": roc_auc(nd[w], nd[l]),
                           "auc_smooth": roc_auc(sm[w], sm[l])} if enough else None)
        out["index"][name] = res
    return out


def measure(root: str | Path, chip_ids: Sequence[str], coverages: Sequence[float], cloud_seeds: Sequence[int],
            seed_fn: Callable[[str, float, int], int] = eval_cloud_seed, log=print) -> dict:
    """{(variant, coverage): [chip_leakage per (chip, cloud seed)]} on the evaluation clouds."""
    out = {(v, c): [] for v in VARIANTS for c in coverages}
    for cid in chip_ids:
        s2, label = load_raw(root, cid, "s2"), load_label(root, cid)
        for cov in coverages:
            for seed in cloud_seeds:
                alpha, mask, shade, spectrum = cloud_params(s2.shape[1], s2.shape[2], cov, seed_fn(cid, cov, seed),
                                                            thin=False)
                for variant, a in (("thick", alpha), ("opaque", opaque_alpha(alpha, mask))):
                    r = chip_leakage(s2, label, apply_clouds(s2, a, spectrum, shade), a, mask)
                    out[(variant, cov)].append({"chip_id": cid, "cloud_seed": seed, **r})
        log(f"  {cid}: done")
    return out


def _med(vals) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.median(vals)) if vals else math.nan


def summary_markdown(results: dict, chip_ids: Sequence[str], cloud_seeds: Sequence[int], max_real_cloud: float) -> str:
    covs = sorted({c for _, c in results})
    L = ["# Ground signal under the synthetic evaluation clouds", "",
         f"Local sample: {len(chip_ids)} chips with < {max_real_cloud:.0%} real cloud (s2cloudless), "
         f"cloud seeds {list(cloud_seeds)} of the robustness evaluation (thick clouds, large or cumulus). "
         "`thick` = the clouds as used in training; `opaque` = the same mask with opacity 1 inside it. "
         "All statistics are over the covered pixels (inside the mask).", "",
         "## Opacity inside the mask (pooled over chips and seeds)", "",
         "| clouds | coverage | " + " | ".join(f"p{q * 100:g}" if q else "min" for q in ALPHA_QUANTILES)
         + " | mean | share < 0.9 | share < 0.8 |", "|---|--:|" + "--:|" * (len(ALPHA_QUANTILES) + 3)]
    for v in VARIANTS:
        for c in covs:
            a = np.concatenate([r["alpha"] for r in results[(v, c)]])
            qs = np.quantile(a, ALPHA_QUANTILES)
            L.append(f"| {v} | {c:.0%} | " + " | ".join(f"{q:.3f}" for q in qs)
                     + f" | {a.mean():.3f} | {(a < 0.9).mean():.1%} | {(a < 0.8).mean():.1%} |")
    L += ["", "## Correlation of clouded vs original S2 inside the mask (median over chips and seeds)", "",
          "corr² is the share of the clouded band's variance (inside the cloud) that comes from the ground.", "",
          "| clouds | coverage | " + " | ".join(CORR_BANDS) + " |", "|---|--:|" + "--:|" * len(CORR_BANDS)]
    for v in VARIANTS:
        for c in covs:
            L.append(f"| {v} | {c:.0%} | " + " | ".join(f"{_med(r['corr'][b] for r in results[(v, c)]):.3f}"
                                                          for b in CORR_BANDS) + " |")
    L += ["", "## Water vs land inside the mask (median over chips and seeds)", "",
          f"AUC: 0.5 = index carries no water information, 1 = perfectly separates water and land. "
          f"'{SMOOTH}×{SMOOTH}' = index averaged over the covered pixels of a {SMOOTH}×{SMOOTH} px window "
          f"(spatial context, as a CNN has). Only (chip, seed) pairs with ≥ {MIN_CLASS_PIXELS} water and land "
          "pixels under the cloud. Δ = mean index of water − land.", ""]
    for name in INDICES:
        L += [f"### {name} ({' − '.join(INDICES[name])}) / (sum)", "",
              "| clouds | coverage | n | Δ clear | Δ clouded | AUC clear | AUC clouded | AUC clear "
              f"{SMOOTH}×{SMOOTH} | AUC clouded {SMOOTH}×{SMOOTH} |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for v in VARIANTS:
            for c in covs:
                rs = [r["index"][name] for r in results[(v, c)] if r["index"][name]["clear"]]
                cells = [_med(r[w][k] for r in rs) for k in ("diff",) for w in ("clear", "cloud")]
                cells += [_med(r[w][k] for r in rs) for k in ("auc", "auc_smooth") for w in ("clear", "cloud")]
                L.append(f"| {v} | {c:.0%} | {len(rs)} | " + " | ".join(f"{x:+.3f}" if i < 2 else f"{x:.3f}"
                                                                     for i, x in enumerate(cells)) + " |")
        L.append("")
    return "\n".join(L)
