"""Audit of the s2cloudless cloud masks behind the real-cloud bins (docs/ROADMAP.md, Stage 5 TODO).

Per chip, the raw s2cloudless probability map is computed once (same input and parameters as
floodfusion.clouds) and turned into masks at several thresholds (0.4 = the current one), with
the same smoothing (average_over) and dilation. For each threshold it records:

  * cloud_fraction: cloudy valid pixels / valid pixels (as in docs/cloud_cover.csv at 0.4);
  * label check: how the hand labels (-1 no data, 0 land, 1 water) split between cloudy and
    clear valid S2 pixels. If cloudy pixels are mostly -1, the annotators skipped what they
    could not see, and a "cloudy" bin is scored mostly on its clear part;
  * mask diagnostics: share of mask pixels whose own probability is at most the threshold
    (added by smoothing and dilation: gaps and edges), and share of mask pixels that are dim
    (mean B2-B4 TOA reflectance < DIM_REFLECTANCE: haze or ground rather than a bright cloud
    top; a heuristic, the figures are the real check).

Never imports torch (see floodfusion.chips): s2cloudless is loaded through floodfusion.clouds.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tqdm.auto import tqdm

from .chips import IGNORE_INDEX, S2_BANDS, SPLITS, list_split_chips, load_label, load_raw
from .clouds import CLOUD_BIN_LABELS, CLOUD_THRESHOLD, CloudDetector, cloud_bin, s2_reflectance

THRESHOLDS = (0.4, 0.5, 0.6)
DIM_REFLECTANCE = 0.15  # mean B2-B4 TOA reflectance; bright cloud tops are well above (~0.3-0.5)
NAMED_CHIPS = ("Paraguay_215904", "Paraguay_36146")  # flagged in the ROADMAP TODO
MID_CLOUD = 0.45  # "mid-cloud" figure chips: train chips closest to this cloud fraction
LABEL_KEYS = ("nodata", "land", "water")  # label -1, 0, 1
EVAL_SPLITS = ("test", "bolivia")
_VIS = [S2_BANDS.index(b) for b in ("B2", "B3", "B4")]


# ----------------------------------------------------------------------------- per chip

def probability_map(detector: CloudDetector, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw s2cloudless probability (H, W) with nodata set to 0 (as floodfusion.clouds), and valid mask."""
    refl, valid = s2_reflectance(s2)
    prob = np.asarray(detector.get_cloud_probability_maps(refl[None]), dtype=np.float32)[0]
    prob[~valid] = 0.0
    return prob, valid


def masks_at(detector: CloudDetector, prob: np.ndarray, valid: np.ndarray,
             thresholds: Sequence[float] = THRESHOLDS) -> dict[float, np.ndarray]:
    """{threshold: cloud mask (H, W) bool, only True on valid pixels} with the detector's smoothing/dilation."""
    return {t: np.asarray(detector.get_mask_from_prob(prob[None], threshold=t))[0].astype(bool) & valid
            for t in thresholds}


def label_counts(label: np.ndarray, cloud: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, int]]:
    """Pixel counts {"cloudy" | "clear": {"nodata", "land", "water"}} over valid S2 pixels
    (label -1 / 0 / 1). Pixels without valid S2 are not counted (no cloud decision there)."""
    if not label.shape == cloud.shape == valid.shape:
        raise ValueError(f"label {label.shape}, cloud {cloud.shape} and valid {valid.shape} must match")
    bad = valid & ~np.isin(label, (IGNORE_INDEX, 0, 1))
    if bad.any():
        raise ValueError(f"labels must be -1, 0 or 1; got {sorted(np.unique(label[bad]).tolist())}")
    out = {}
    for name, sel in (("cloudy", valid & cloud), ("clear", valid & ~cloud)):
        out[name] = {k: int((sel & (label == v)).sum()) for k, v in zip(LABEL_KEYS, (IGNORE_INDEX, 0, 1))}
    return out


def chip_audit(detector: CloudDetector, s2: np.ndarray, label: np.ndarray,
               thresholds: Sequence[float] = THRESHOLDS) -> dict:
    """Audit row of one chip (without id/split): valid_s2_fraction, then per threshold t the keys
    ``t{t}_cloud_fraction``, ``t{t}_{cloudy|clear}_{nodata|land|water}``, ``t{t}_mask_lowprob``,
    ``t{t}_mask_dim`` (shares of the mask; NaN for an empty mask)."""
    prob, valid = probability_map(detector, s2)
    vis = np.nan_to_num(s2[_VIS]).mean(axis=0) / 10000.0  # nodata pixels are never in the mask
    n_valid = int(valid.sum())
    row = {"valid_s2_fraction": n_valid / valid.size}
    for t, mask in masks_at(detector, prob, valid, thresholds).items():
        p = f"t{t:g}"
        n = int(mask.sum())
        row[f"{p}_cloud_fraction"] = n / n_valid if n_valid else math.nan
        for side, counts in label_counts(label, mask, valid).items():
            for k, v in counts.items():
                row[f"{p}_{side}_{k}"] = v
        row[f"{p}_mask_lowprob"] = float((prob[mask] <= t).mean()) if n else math.nan
        row[f"{p}_mask_dim"] = float((vis[mask] < DIM_REFLECTANCE).mean()) if n else math.nan
    return row


def audit_all(root: str | Path, detector: CloudDetector, thresholds: Sequence[float] = THRESHOLDS, *,
              splits: Sequence[str] = SPLITS, allow_missing: bool = False, progress: bool = True) -> list[dict]:
    rows = []
    for split in splits:
        for cid in tqdm(list_split_chips(root, split, allow_missing), desc=f"audit {split}", disable=not progress):
            rows.append({"chip_id": cid, "split": split,
                         **chip_audit(detector, load_raw(root, cid, "s2"), load_label(root, cid), thresholds)})
    return rows


# ----------------------------------------------------------------------------- aggregation

def pooled_labels(rows: Iterable[dict], t: float, group) -> dict[str, dict]:
    """Pooled label counts per group (``group(row)`` -> name or None to skip): {name: {"chips",
    "cloudy": {...}, "clear": {...}}}."""
    p = f"t{t:g}"
    out: dict[str, dict] = {}
    for r in rows:
        g = group(r)
        if g is None:
            continue
        acc = out.setdefault(g, {"chips": 0, "cloudy": dict.fromkeys(LABEL_KEYS, 0), "clear": dict.fromkeys(LABEL_KEYS, 0)})
        acc["chips"] += 1
        for side in ("cloudy", "clear"):
            for k in LABEL_KEYS:
                acc[side][k] += int(r[f"{p}_{side}_{k}"])
    return out


def label_shares(acc: dict) -> dict[str, float]:
    """From pooled counts: share of -1 / land / water among cloudy and among clear pixels, the
    cloudy share of valid pixels, and the share of SCORED pixels (label 0/1) that lie under cloud."""
    def frac(a, b):
        return a / b if b else math.nan

    cl, cr = acc["cloudy"], acc["clear"]
    n_cl, n_cr = sum(cl.values()), sum(cr.values())
    out = {f"cloudy_{k}": frac(cl[k], n_cl) for k in LABEL_KEYS}
    out.update({f"clear_{k}": frac(cr[k], n_cr) for k in LABEL_KEYS})
    out["cloudy_of_valid"] = frac(n_cl, n_cl + n_cr)
    scored_cloudy, scored_clear = cl["land"] + cl["water"], cr["land"] + cr["water"]
    out["scored_under_cloud"] = frac(scored_cloudy, scored_cloudy + scored_clear)
    return out


def nodata_agreement(rows: Iterable[dict], t: float, split: str = "train") -> dict[str, float]:
    """How well the mask at threshold t matches the annotators' no-data (label -1) on valid S2
    pixels of ``split`` (pooled): precision = share of cloudy pixels labelled -1, recall = share
    of -1 pixels that are cloudy, F1. Label -1 marks what annotators could not label, mostly
    because they could not see the ground; a proxy for "optically hidden", not ground truth.
    Train chips only by default, so a threshold choice never looks at test or bolivia labels."""
    acc = pooled_labels(rows, t, lambda r: "all" if r["split"] == split else None).get("all")
    if not acc:
        return {"precision": math.nan, "recall": math.nan, "f1": math.nan, "chips": 0}
    tp, fp, fn = acc["cloudy"]["nodata"], acc["cloudy"]["land"] + acc["cloudy"]["water"], acc["clear"]["nodata"]
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan
    return {"precision": prec, "recall": rec, "f1": f1, "chips": acc["chips"]}


def bin_of(row: dict, t: float) -> str | None:
    return cloud_bin(row[f"t{t:g}_cloud_fraction"])


def bin_counts_at(rows: Sequence[dict], t: float, splits: Sequence[str] = EVAL_SPLITS) -> dict[str, dict[str, int]]:
    out = {s: dict.fromkeys([*CLOUD_BIN_LABELS, "no valid S2"], 0) for s in splits}
    for r in rows:
        if r["split"] in out:
            out[r["split"]][bin_of(r, t) or "no valid S2"] += 1
    return out


def pick_audit_chips(rows: Sequence[dict], n_cloudy: int = 6, n_mid: int = 6, named: Sequence[str] = NAMED_CHIPS,
                     t: float = CLOUD_THRESHOLD, split: str = "train") -> tuple[list[str], list[str]]:
    """(cloudiest, mid-cloud + named) train chip ids at threshold t: the n_cloudy cloudiest, then
    the n_mid chips closest to MID_CLOUD among the rest (ties by id), plus ``named`` chips that
    are present and not picked yet."""
    key = f"t{t:g}_cloud_fraction"
    cand = [r for r in rows if r["split"] == split and not math.isnan(r[key])]
    cloudy = [r["chip_id"] for r in sorted(cand, key=lambda r: (-r[key], r["chip_id"]))[:n_cloudy]]
    rest = [r for r in cand if r["chip_id"] not in cloudy]
    mid = [r["chip_id"] for r in sorted(rest, key=lambda r: (abs(r[key] - MID_CLOUD), r["chip_id"]))[:n_mid]]
    present = {r["chip_id"] for r in rows}
    mid += [c for c in named if c in present and c not in cloudy and c not in mid]
    return cloudy, mid


# ----------------------------------------------------------------------------- csv and summary

def write_rows(rows: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and not math.isnan(v) else "" if isinstance(v, float)
                            else v) for k, v in r.items()})
    return path


def read_rows(path: str | Path) -> list[dict]:
    def conv(k, v):
        if k in ("chip_id", "split"):
            return v
        if v == "":
            return math.nan
        return float(v) if ("fraction" in k or "mask_" in k) else int(v)

    with open(path, newline="") as f:
        return [{k: conv(k, v) for k, v in r.items()} for r in csv.DictReader(f)]


def _pct(x: float) -> str:
    return "–" if math.isnan(x) else f"{x:.1%}"


def _label_table(rows, t, groups: list[tuple[str, object]]) -> list[str]:
    L = ["| group | chips | cloudy of valid S2 | cloudy: -1 / land / water | clear: -1 / land / water | "
         "scored px under cloud |", "|---|--:|--:|--:|--:|--:|"]
    for name, fn in groups:
        for g, acc in sorted(pooled_labels(rows, t, fn).items(), key=lambda kv: _group_order(kv[0])):
            s = label_shares(acc)
            L.append(f"| {name}{g} | {acc['chips']} | {_pct(s['cloudy_of_valid'])} | "
                     f"{_pct(s['cloudy_nodata'])} / {_pct(s['cloudy_land'])} / {_pct(s['cloudy_water'])} | "
                     f"{_pct(s['clear_nodata'])} / {_pct(s['clear_land'])} / {_pct(s['clear_water'])} | "
                     f"{_pct(s['scored_under_cloud'])} |")
    return L


def _group_order(g: str):
    order = [*SPLITS, *(f"{s} · {b}" for s in SPLITS for b in CLOUD_BIN_LABELS)]
    return (order.index(g) if g in order else len(order), g)


def summary_markdown(rows: Sequence[dict], thresholds: Sequence[float] = THRESHOLDS,
                     reference: dict[str, float] | None = None, note: str = "") -> str:
    """Tables of the audit. ``reference`` = {chip_id: cloud fraction from cloud_cover.csv} to check
    that the current threshold reproduces it."""
    splits = [s for s in SPLITS if any(r["split"] == s for r in rows)]
    L = ["# s2cloudless mask audit", ""]
    if note:
        L += [note, ""]
    per_split = ", ".join(f"{s}: {sum(r['split'] == s for r in rows)}" for s in splits)
    L += [f"{len(rows)} chips ({per_split}). "
          f"Thresholds {list(thresholds)}; {CLOUD_THRESHOLD:g} is the one behind docs/cloud_cover.csv. "
          "Same s2cloudless input and smoothing/dilation as floodfusion.clouds.", ""]
    if reference:
        key = f"t{CLOUD_THRESHOLD:g}_cloud_fraction"
        diffs = [abs(r[key] - reference[r["chip_id"]]) for r in rows
                 if r["chip_id"] in reference and not math.isnan(reference[r["chip_id"]]) and not math.isnan(r[key])]
        if diffs:
            L += [f"Check: at {CLOUD_THRESHOLD:g} the cloud fraction matches cloud_cover.csv for {len(diffs)} chips "
                  f"(max abs difference {max(diffs):.2e}).", ""]
    for t in thresholds:
        L += [f"## Label check at threshold {t:g}", "",
              "Pooled over valid S2 pixels. 'scored px under cloud' = share of the labelled (0/1) pixels, the ones "
              "IoU is computed on, that are cloudy.", ""]
        L += _label_table(rows, t, [("", lambda r: r["split"]),
                                    ("", lambda r, t=t: f"{r['split']} · {bin_of(r, t)}" if bin_of(r, t) else None)])
        L.append("")
    L += ["## Agreement of the mask with the annotators' no-data (label -1), train chips only", "",
          "precision = share of cloudy pixels labelled -1; recall = share of -1 pixels that are cloudy "
          "(valid S2 only, pooled). Label -1 is a proxy for 'annotator could not see the ground'.", "",
          "| threshold | precision | recall | F1 |", "|--:|--:|--:|--:|"]
    for t in thresholds:
        a = nodata_agreement(rows, t)
        L.append(f"| {t:g} | {_pct(a['precision'])} | {_pct(a['recall'])} | {_pct(a['f1'])} |")
    L += ["", "## Bin counts per threshold (test and bolivia)", "",
          "| split | threshold | " + " | ".join(CLOUD_BIN_LABELS) + " | no valid S2 |",
          "|---|--:|" + "--:|" * (len(CLOUD_BIN_LABELS) + 1)]
    for s in EVAL_SPLITS:
        for t in thresholds:
            c = bin_counts_at(rows, t)[s]
            L.append(f"| {s} | {t:g} | " + " | ".join(str(c[b]) for b in CLOUD_BIN_LABELS) + f" | {c['no valid S2']} |")
    L += ["", "## Cloud fraction and mask diagnostics per threshold (all chips with valid S2)", "",
          "lowprob = share of mask pixels whose own probability is ≤ the threshold (added by smoothing and "
          f"dilation); dim = share of mask pixels with mean B2–B4 TOA reflectance < {DIM_REFLECTANCE:g} "
          "(haze or ground, not a bright cloud top; heuristic). Medians over chips.", "",
          "| threshold | mean cloud fraction | chips ≥ 10% cloud | chips moving bin vs "
          f"{CLOUD_THRESHOLD:g} | median lowprob | median dim |", "|--:|--:|--:|--:|--:|--:|"]
    base = f"t{CLOUD_THRESHOLD:g}_cloud_fraction"
    for t in thresholds:
        k = f"t{t:g}"
        ok = [r for r in rows if not math.isnan(r[f"{k}_cloud_fraction"])]
        moved = sum(cloud_bin(r[f"{k}_cloud_fraction"]) != cloud_bin(r[base]) for r in ok)
        lp = [r[f"{k}_mask_lowprob"] for r in ok if not math.isnan(r[f"{k}_mask_lowprob"])]
        dim = [r[f"{k}_mask_dim"] for r in ok if not math.isnan(r[f"{k}_mask_dim"])]
        L.append(f"| {t:g} | {np.mean([r[f'{k}_cloud_fraction'] for r in ok]):.1%} | "
                 f"{sum(r[f'{k}_cloud_fraction'] >= 0.1 for r in ok)} | {moved} | "
                 f"{_pct(float(np.median(lp))) if lp else '–'} | {_pct(float(np.median(dim))) if dim else '–'} |")
    L += ["", "## Per chip: cloud fraction per threshold (chips ≥ 10% cloud at any threshold)", "",
          "| chip | split | " + " | ".join(f"{t:g}" for t in thresholds) + " | dim at "
          f"{CLOUD_THRESHOLD:g} |", "|---|---|" + "--:|" * (len(thresholds) + 1)]
    for r in sorted(rows, key=lambda r: (SPLITS.index(r["split"]), -np.nan_to_num(r[base]), r["chip_id"])):
        fr = [r[f"t{t:g}_cloud_fraction"] for t in thresholds]
        if any(not math.isnan(f) and f >= 0.1 for f in fr):
            L.append(f"| {r['chip_id']} | {r['split']} | " + " | ".join(_pct(f) for f in fr)
                     + f" | {_pct(r[f't{CLOUD_THRESHOLD:g}_mask_dim'])} |")
    return "\n".join(L)
