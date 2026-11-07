"""Figures for inspecting Sen1Floods11 chips (raw values, not normalised).

Conventions used in every figure:
  * Missing input pixels (NaN, and S2 pixels with all 13 bands == 0) are magenta.
  * SAR panels use the train clip limits from band_stats.json as display range,
    so contrast is identical across chips and matches what the model receives.
  * Label: water blue, not water white, no data (-1) grey.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: works locally, on Colab and in tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .chips import IGNORE_INDEX, S1_BANDS, S2_BANDS, load_label, load_raw  # noqa: E402

TRUE_COLOUR: tuple[str, str, str] = ("B4", "B3", "B2")
FALSE_COLOUR: tuple[str, str, str] = ("B11", "B8", "B4")  # SWIR1/NIR/red: water dark
# L1C DN / scale -> [0, 1]; 3000 ~ TOA reflectance 0.3 (brightened true colour).
TRUE_COLOUR_SCALE = 3000.0
FALSE_COLOUR_SCALE = 5000.0
NODATA_RGB = (1.0, 0.0, 1.0)  # magenta
LABEL_COLOURS = {IGNORE_INDEX: "#9e9e9e", 0: "#ffffff", 1: "#1f6fd1"}
HIST_EDGES = np.arange(-50.0, 15.0 + 1e-9, 0.25)  # dB, 0.25 dB bins


def composite(s2: np.ndarray, bands: Sequence[str], scale: float) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] from all-13-band S2 (NaN = missing -> magenta)."""
    if s2.shape[0] != len(S2_BANDS):
        raise ValueError(f"expected all {len(S2_BANDS)} S2 bands, got {s2.shape[0]}")
    rgb = np.stack([s2[S2_BANDS.index(b)] for b in bands], axis=-1) / scale
    missing = np.isnan(rgb).any(axis=-1)
    rgb = np.clip(np.nan_to_num(rgb), 0.0, 1.0)
    rgb[missing] = NODATA_RGB
    return rgb


def water_fraction(label: np.ndarray) -> float | None:
    """Water pixels / labelled pixels (label != -1); None if nothing is labelled."""
    labelled = int((label != IGNORE_INDEX).sum())
    return float((label == 1).sum()) / labelled if labelled else None


def _clip_limits(stats: dict) -> dict[str, tuple[float, float]]:
    s = stats["s1"]
    names = s.get("bands", list(S1_BANDS))
    return {b: (float(s["clip_min"][names.index(b)]), float(s["clip_max"][names.index(b)]))
            for b in S1_BANDS}


def plot_chip(root: str | Path, chip_id: str, stats: dict, out_path: str | Path) -> Path:
    """Five panels: VV, VH (dB), S2 true colour, S2 false colour, label."""
    s1 = load_raw(root, chip_id, "s1")
    s2 = load_raw(root, chip_id, "s2")
    y = load_label(root, chip_id)
    lim = _clip_limits(stats)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.4), constrained_layout=True)
    sar_cmap = plt.get_cmap("gray").with_extremes(bad=NODATA_RGB)
    for ax, b, name in zip(axes[:2], range(2), S1_BANDS):
        lo, hi = lim[name]
        im = ax.imshow(np.ma.masked_invalid(s1[b]), cmap=sar_cmap, vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="dB")
        nan_pct = np.isnan(s1[b]).mean()
        ax.set_title(f"SAR {name} (dB), shown [{lo:.1f}, {hi:.1f}]\nmissing {nan_pct:.1%}")

    nodata_s2 = np.isnan(s2).all(axis=0).mean()
    axes[2].imshow(composite(s2, TRUE_COLOUR, TRUE_COLOUR_SCALE), interpolation="nearest")
    axes[2].set_title(f"S2 true colour ({','.join(TRUE_COLOUR)}) / {TRUE_COLOUR_SCALE:g}\n"
                      f"missing {nodata_s2:.1%}")
    axes[3].imshow(composite(s2, FALSE_COLOUR, FALSE_COLOUR_SCALE), interpolation="nearest")
    axes[3].set_title(f"S2 false colour ({','.join(FALSE_COLOUR)}) / {FALSE_COLOUR_SCALE:g}\n"
                      "water = dark blue / black")

    keys = sorted(LABEL_COLOURS)
    axes[4].imshow(y, cmap=ListedColormap([LABEL_COLOURS[k] for k in keys]),
                   norm=BoundaryNorm(np.array(keys + [keys[-1] + 1]) - 0.5, len(keys)),
                   interpolation="nearest")
    axes[4].set_title(f"Label (no data {(y == IGNORE_INDEX).mean():.1%})")
    axes[4].legend(handles=[Patch(facecolor=LABEL_COLOURS[1], label="water"),
                            Patch(facecolor=LABEL_COLOURS[0], edgecolor="k", label="not water"),
                            Patch(facecolor=LABEL_COLOURS[IGNORE_INDEX], label="no data (-1)"),
                            Patch(facecolor=NODATA_RGB, label="missing input")],
                   loc="lower left", fontsize=8, framealpha=0.9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    wf = water_fraction(y)
    wf_txt = "n/a (no labelled pixels)" if wf is None else f"{wf:.1%}"
    fig.suptitle(f"{chip_id}  |  water fraction {wf_txt} of labelled pixels  |  magenta = missing input",
                 fontsize=13)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def s1_class_histograms(root: str | Path, chip_ids: Iterable[str], edges: np.ndarray = HIST_EDGES,
                        progress: bool = False) -> dict:
    """Streamed per-class S1 histograms over labelled pixels with valid S1.

    Counts are accumulated into the fixed bins ``edges`` one chip at a time, so memory
    is constant in the number of chips (full train works on Colab). Returns
    ``{band: {"water"|"land": {"counts", "below", "above", "total"}}}`` where
    below/above count values outside ``edges`` (nothing is silently dropped).
    """
    out = {b: {c: {"counts": np.zeros(len(edges) - 1, np.int64), "below": 0, "above": 0, "total": 0}
               for c in ("water", "land")} for b in S1_BANDS}
    n_chips = 0
    for cid in tqdm(chip_ids, desc="S1 histogram", disable=not progress):
        n_chips += 1
        s1 = load_raw(root, cid, "s1")
        y = load_label(root, cid)
        for b, name in enumerate(S1_BANDS):
            valid = ~np.isnan(s1[b])
            for cls, val in (("water", 1), ("land", 0)):
                v = s1[b][valid & (y == val)]
                h = out[name][cls]
                h["counts"] += np.histogram(v, bins=edges)[0]
                h["below"] += int((v < edges[0]).sum())
                h["above"] += int((v > edges[-1]).sum())
                h["total"] += int(v.size)
    out["n_chips"] = n_chips
    return out


def plot_s1_histograms(hist: dict, stats: dict, out_path: str | Path, title: str,
                       edges: np.ndarray = HIST_EDGES) -> Path:
    """VV and VH density for water vs not-water, with clip limits as vertical lines."""
    lim = _clip_limits(stats)
    width = np.diff(edges)
    centres = edges[:-1] + width / 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    colours = {"water": LABEL_COLOURS[1], "land": "#c0612b"}
    for ax, name in zip(axes, S1_BANDS):
        lo, hi = lim[name]
        for cls, label in (("water", "water"), ("land", "not water")):
            h = hist[name][cls]
            if h["total"] == 0:
                continue
            # Fraction of the class outside the clip limits (computed on bin counts).
            # Assumes clip limits fall inside `edges`, which holds for dB data.
            c = h["counts"]
            below = (c[edges[1:] <= lo].sum() + h["below"]) / h["total"]
            above = (c[edges[:-1] >= hi].sum() + h["above"]) / h["total"]
            ax.step(centres, c / (h["total"] * width), where="mid", color=colours[cls], lw=1.6,
                    label=f"{label} (n={h['total']:,}; {below:.1%} < min, {above:.1%} > max)")
        ax.axvline(lo, color="k", ls="--", lw=1.2, label=f"clip_min {lo:.2f}")
        ax.axvline(hi, color="k", ls=":", lw=1.4, label=f"clip_max {hi:.2f}")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.4)  # headroom so the legend never covers a peak
        ax.set_xlabel(f"{name} backscatter (dB)")
        ax.set_ylabel("density (per class)")
        ax.set_title(f"SAR {name}")
        ax.legend(fontsize=8, loc="upper right")  # dB > clip_max is empty, so the curves stay visible
    fig.suptitle(f"{title}: S1 dB by class, labelled pixels only ({hist['n_chips']} chips); "
                 "clip limits from train band_stats.json", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_cloud_examples(root: str | Path, examples: Sequence[tuple[dict, np.ndarray]],
                        out_path: str | Path, title: str, ncols: int = 3) -> Path:
    """Grid of S2 true colour chips with the cloud mask outlined in red.

    ``examples`` is a sequence of (row, mask): row has chip_id, split, cloud_fraction,
    valid_s2_fraction (as in floodfusion.clouds); mask is the boolean (H, W) cloud mask.
    """
    nrows = max(1, -(-len(examples) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.4 * nrows), constrained_layout=True,
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_axis_off()
    for ax, (row, mask) in zip(axes.ravel(), examples):
        ax.set_axis_on()
        s2 = load_raw(root, row["chip_id"], "s2")
        ax.imshow(composite(s2, TRUE_COLOUR, TRUE_COLOUR_SCALE), interpolation="nearest")
        if mask.any():  # faint tint shows which side of the outline is cloud
            ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap(["red"]), alpha=0.22,
                      interpolation="nearest")
        if mask.any() and not mask.all():
            ax.contour(mask.astype(np.uint8), levels=[0.5], colors="red", linewidths=0.9)
        elif mask.all():  # nothing to outline: frame the whole chip instead
            for spine in ax.spines.values():
                spine.set_color("red"); spine.set_linewidth(4)
        ax.set_title(f"{row['chip_id']}\ncloud {row['cloud_fraction']:.1%} of valid S2, "
                     f"valid {row['valid_s2_fraction']:.1%}", fontsize=10)
    fig.suptitle(f"{title}  |  red outline + tint = s2cloudless cloud mask  |  magenta = missing S2", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_true_colour_grid(rows: Sequence[tuple[str, Sequence[tuple[str, np.ndarray]]]], out_path: str | Path,
                          title: str) -> Path:
    """Grid of raw S2 (13, H, W) true colour images: ``rows`` = [(row label, [(panel title,
    s2), ...]), ...]. Same display scale as every other true colour figure."""
    ncols = max(len(panels) for _, panels in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.3 * ncols, 3.6 * len(rows)), constrained_layout=True,
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for r, (label, panels) in enumerate(rows):
        for c, (panel_title, s2) in enumerate(panels):
            ax = axes[r, c]
            ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(composite(s2, TRUE_COLOUR, TRUE_COLOUR_SCALE), interpolation="nearest")
            ax.set_title(panel_title, fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(f"{title}  |  true colour (B4, B3, B2) / {TRUE_COLOUR_SCALE:.0f}  |  magenta = missing S2", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def plot_training_curves(metrics_csv: str | Path, out_path: str | Path, title: str = "") -> Path:
    """Train loss and valid water IoU per epoch from a run's metrics.csv; best epoch marked.
    If present, val_iou_cloud50 (fixed synthetic clouds, information only) is a second line."""
    import csv

    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} has no epochs yet")

    def col(k):
        return np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])

    ep, iou = col("epoch"), col("val_iou")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    a1.plot(ep, col("train_loss"), label="train loss", color="#c0612b")
    a1.plot(ep, col("val_loss"), label="valid loss", color="#555555", alpha=0.7)
    a1.set_xlabel("epoch"); a1.set_ylabel("CE + Dice"); a1.legend(); a1.set_title("Loss")
    a2.plot(ep, iou, color=LABEL_COLOURS[1], label="valid water IoU (selects best.pt)")
    cloud = col("val_iou_cloud50") if "val_iou_cloud50" in rows[0] else None
    if cloud is not None and np.isfinite(cloud).any():
        a2.plot(ep, cloud, color="#888888", linestyle="--",
                label="valid, synthetic clouds 50% (info only)")
    if np.isfinite(iou).any():
        b = int(np.nanargmax(iou))
        a2.scatter([ep[b]], [iou[b]], color="k", zorder=3, label=f"best {iou[b]:.4f} (epoch {int(ep[b])})")
    a2.set_xlabel("epoch"); a2.set_ylabel("water IoU (valid, pooled)"); a2.set_ylim(0, 1)
    a2.legend(loc="lower right"); a2.set_title("Validation")
    fig.suptitle(title or str(Path(metrics_csv).parent.name))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
