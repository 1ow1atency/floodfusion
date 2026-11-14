"""Core logic of the interactive Gradio demo (app/app.py): chip selection, packed demo
chips, opaque synthetic clouds, checkpoint loading, prediction and per-chip IoU.

The app itself (app/app.py) only wires these functions into a UI; everything testable
lives here, gradio-free.

Demo chips (fixed selection rule, applied to checked-in CSVs only)
------------------------------------------------------------------
``select_demo_chips`` picks the bundled chips from docs/cloud_cover.csv and
docs/results/cloud_audit/chip_audit.csv:

  1. test split only (never train/valid: the models saw those; never Bolivia: it is the
     held-out generalisation split and stays out of any tooling that could feed back);
  2. mostly clear sky: s2cloudless cloud_fraction <= ``max_cloud`` (default 1%), with a
     reliable mask (mask_unreliable = 0) and fully valid S2 (valid_s2_fraction = 1), so
     the only clouds in the demo are the synthetic ones added by the slider;
  3. visible water: water >= ``min_water`` (default 5%) of labelled pixels, from the
     audit's t0.4 clear/cloudy water pixel counts;
  4. a mix of regions: round-robin over regions ordered by their wettest candidate,
     within a region by descending water fraction (ties by chip id), until ``n`` chips.

Packed chips: each demo chip is one compressed .npz (S1 float32 dB with NaN for missing,
S2 raw DN as uint16 when lossless, label int8), ~5-8 MB instead of ~20 MB of GeoTIFFs,
and the app needs no rasterio at runtime. ``load_packed_chip`` re-applies the standard
missing-value rules (floodfusion.chips.mark_missing), so inputs are bit-identical to
loading the original GeoTIFFs with ``load_raw``.

Clouds: the slider adds OPAQUE thick synthetic clouds (clouds_synth thick clouds with
``opaque_alpha``: opacity 1 inside the covered mask, no ground signal survives there;
the stricter variant of the robustness evaluation). The seed depends only on the chip id
(``chip_seed``), not on the coverage, so moving the slider grows the *same* cloud field
instead of redrawing a new one — and every visitor sees identical clouds.

IoU: per-chip water IoU with label -1 ignored (floodfusion.metrics rules). This is a
single-chip number for the demo; the reported results pool counts over whole splits.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .chips import IGNORE_INDEX, S2_BANDS, chip_path, load_label, load_raw, mark_missing, read_float32
from .clouds_synth import EVAL_CLOUD_SEED, apply_clouds, chip_seed, cloud_params, opaque_alpha
from .metrics import metrics_from_counts
from .models import build_model
from .training import check_modalities, forward_model

DEMO_SPLIT = "test"
DEMO_N_CHIPS = 8
MAX_CLOUD = 0.01    # s2cloudless cloud fraction: "mostly clear sky"
MIN_WATER = 0.05    # water / labelled pixels: "visible water"
MANIFEST = "manifest.json"
# The two demo models (seed-0 best.pt of each) and their filenames in the weights repo.
WEIGHT_FILES = {"s2_only_robust": "s2_only_robust_seed0_best.pt",
                "fusion_robust": "fusion_robust_seed0_best.pt"}

SELECTION_RULE = (f"{DEMO_SPLIT} split; s2cloudless cloud fraction <= {MAX_CLOUD:.0%} with a reliable mask "
                  f"and fully valid S2; water >= {MIN_WATER:.0%} of labelled pixels; round-robin over regions "
                  f"(ordered by their wettest candidate), within a region by descending water fraction, "
                  f"until {DEMO_N_CHIPS} chips")


# ----------------------------------------------------------------------------- selection

def select_demo_chips(cloud_csv: str | Path, audit_csv: str | Path, split: str = DEMO_SPLIT,
                      n: int = DEMO_N_CHIPS, max_cloud: float = MAX_CLOUD,
                      min_water: float = MIN_WATER) -> list[dict]:
    """The demo chips by the fixed rule (module docstring). Deterministic: depends only on
    the two checked-in CSVs. Returns [{chip_id, region, cloud_fraction, water_fraction}]."""
    with open(audit_csv, newline="") as f:
        audit = {r["chip_id"]: r for r in csv.DictReader(f)}
    with open(cloud_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"{cloud_csv} has no rows for split {split!r}")

    cands = []
    for r in rows:
        a = audit.get(r["chip_id"])
        if a is None or not r["cloud_fraction"]:  # empty = no valid S2 for a cloud mask
            continue
        cloud = float(r["cloud_fraction"])
        if (cloud > max_cloud or r["mask_unreliable"] == "1"
                or float(r["valid_s2_fraction"]) < 1.0 or not np.isfinite(cloud)):
            continue
        water = int(a["t0.4_cloudy_water"]) + int(a["t0.4_clear_water"])
        labelled = water + int(a["t0.4_cloudy_land"]) + int(a["t0.4_clear_land"])
        wf = water / labelled if labelled else 0.0
        if wf < min_water:
            continue
        cands.append({"chip_id": r["chip_id"], "region": r["chip_id"].split("_")[0],
                      "cloud_fraction": cloud, "water_fraction": wf})

    by_region: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(cands, key=lambda c: (-c["water_fraction"], c["chip_id"])):
        by_region[c["region"]].append(c)
    regions = sorted(by_region, key=lambda reg: (-by_region[reg][0]["water_fraction"], reg))
    picked: list[dict] = []
    rank = 0
    while len(picked) < n:
        row = [by_region[reg][rank] for reg in regions if rank < len(by_region[reg])]
        if not row:
            break
        picked.extend(row[: n - len(picked)])
        rank += 1
    return picked


# ----------------------------------------------------------------------------- packed chips

def pack_chip(root: str | Path, chip_id: str, out_path: str | Path) -> Path:
    """One chip's S1, S2 and label as a compressed .npz (module docstring). The raw S2 DN
    are stored as uint16 when that is lossless (it is for Sen1Floods11), else float32."""
    s1 = load_raw(root, chip_id, "s1")  # float32, NaN = missing (NaN survives npz)
    s2 = read_float32(chip_path(root, chip_id, "s2"))  # raw: the all-bands-0 nodata rule is re-applied on load
    if np.all(np.isfinite(s2)) and np.all(s2 >= 0) and np.all(s2 <= np.iinfo(np.uint16).max) \
            and np.array_equal(s2, np.round(s2)):
        s2 = s2.astype(np.uint16)
    label = load_label(root, chip_id)
    bad = np.setdiff1d(np.unique(label), [IGNORE_INDEX, 0, 1])
    if bad.size:
        raise ValueError(f"{chip_id}: unexpected label values {bad.tolist()}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, chip_id=np.str_(chip_id), s1=s1, s2=s2, label=label.astype(np.int8))
    return out_path


def load_packed_chip(path: str | Path) -> dict:
    """{"id", "s1", "s2", "label"} exactly as ``load_raw``/``load_label`` would give for the
    original GeoTIFFs: float32 inputs with NaN = missing, int64 label."""
    with np.load(path) as z:
        s1 = mark_missing(z["s1"].astype(np.float32), "s1")
        s2 = mark_missing(z["s2"].astype(np.float32), "s2")
        return {"id": str(z["chip_id"]), "s1": s1, "s2": s2, "label": z["label"].astype(np.int64)}


def prepare_demo_chips(root: str | Path, out_dir: str | Path, cloud_csv: str | Path,
                       audit_csv: str | Path, n: int = DEMO_N_CHIPS, log=print) -> list[dict]:
    """Select the demo chips (fixed rule) and pack them into ``out_dir`` with a manifest.
    Needs only the selected chips' GeoTIFFs under ``root`` (not the full dataset)."""
    chips = select_demo_chips(cloud_csv, audit_csv, n=n)
    missing = [c["chip_id"] for c in chips
               if any(not chip_path(root, c["chip_id"], k).is_file() for k in ("s1", "s2", "label"))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected chip(s) not under {root}: {missing}\n"
            "Copy their S1Hand/S2Hand/LabelHand GeoTIFFs from Google Drive first.")
    out_dir = Path(out_dir)
    for c in chips:
        c["file"] = f"{c['chip_id']}.npz"
        p = pack_chip(root, c["chip_id"], out_dir / c["file"])
        log(f"  packed {c['chip_id']:<22s} water {c['water_fraction']:.1%}, real cloud "
            f"{c['cloud_fraction']:.1%} -> {p} ({p.stat().st_size / 1e6:.1f} MB)")
    manifest = {"rule": SELECTION_RULE, "split": DEMO_SPLIT, "chips": chips}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"  wrote {out_dir / MANIFEST}")
    return chips


# ----------------------------------------------------------------------------- clouds

def opaque_cloud_s2(s2_raw: np.ndarray, coverage: float, chip_id: str,
                    base_seed: int = EVAL_CLOUD_SEED) -> np.ndarray:
    """Raw S2 (13, H, W) with an opaque thick synthetic cloud over ``coverage`` of the chip
    (``opaque_alpha``: no ground signal inside the cloud mask; NaN stays NaN). Deterministic:
    the seed is ``chip_seed(chip_id, base_seed)`` and does not depend on the coverage, so a
    growing coverage grows the same cloud field (see module docstring)."""
    if coverage <= 0.0:
        return s2_raw.astype(np.float32, copy=True)
    alpha, mask, shade, spectrum = cloud_params(s2_raw.shape[1], s2_raw.shape[2], coverage,
                                                chip_seed(chip_id, base_seed), thin=False)
    return apply_clouds(s2_raw, opaque_alpha(alpha, mask), spectrum, shade)


# ----------------------------------------------------------------------------- models

@dataclass
class DemoModel:
    """A loaded best.pt ready for CPU inference."""
    name: str
    model: torch.nn.Module
    cfg: dict
    stats: dict
    mods: tuple[str, ...]
    epoch: int
    smoke: bool  # smoke/fake checkpoint: the app must say the predictions are meaningless


def load_demo_model(path: str | Path, device: torch.device | str = "cpu") -> DemoModel:
    """Load a best.pt exactly like floodfusion.evaluation: the model is built from the
    checkpoint's config and normalises with the checkpoint's train-split stats."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # own checkpoint: holds config + stats
    cfg = ckpt["config"]
    model = build_model(cfg, encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    model.to(torch.device(device)).eval()
    return DemoModel(name=cfg["name"], model=model, cfg=cfg, stats=ckpt["stats"],
                     mods=check_modalities(cfg), epoch=int(ckpt.get("epoch", -1)),
                     smoke=bool(ckpt.get("smoke")))


def _normalise(x: np.ndarray, stats_mod: dict, idx: list[int]) -> np.ndarray:
    """The Sen1Floods11 input pipeline on one raw array: select bands, clip, normalise,
    missing -> 0 (= train band mean after normalisation)."""
    sel = {k: np.asarray(stats_mod[k], dtype=np.float32)[idx, None, None]
           for k in ("mean", "std", "clip_min", "clip_max")}
    x = np.clip(x[idx], sel["clip_min"], sel["clip_max"])  # NaN stays NaN
    x = (x - sel["mean"]) / sel["std"]
    x[np.isnan(x)] = 0.0
    return x.astype(np.float32)


@torch.no_grad()
def predict(dm: DemoModel, s1_raw: np.ndarray, s2_raw: np.ndarray) -> np.ndarray:
    """Predicted labels (H, W) uint8 in {0, 1} for one chip from RAW inputs (dB / DN, NaN =
    missing, S2 all 13 bands). Normalisation and the forward pass match training exactly
    (same code path as evaluation: ``forward_model``)."""
    s2_bands = dm.cfg["data"].get("s2_bands") or list(S2_BANDS)
    batch = {}
    if "s1" in dm.mods:
        batch["s1"] = torch.from_numpy(_normalise(s1_raw.copy(), dm.stats["s1"], [0, 1]))[None]
    if "s2" in dm.mods:
        idx = [S2_BANDS.index(b) for b in s2_bands]
        batch["s2"] = torch.from_numpy(_normalise(s2_raw.copy(), dm.stats["s2"], idx))[None]
    device = next(dm.model.parameters()).device
    logits = forward_model(dm.model, batch, dm.cfg, device)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def chip_water_iou(pred: np.ndarray, label: np.ndarray) -> float:
    """Water IoU of one chip, label -1 ignored (NaN if neither label nor prediction has
    water there). Single-chip diagnostic; reported results pool counts over splits."""
    valid = label != IGNORE_INDEX
    t, p = (label == 1) & valid, (pred == 1) & valid
    return metrics_from_counts(tp=int((p & t).sum()), fp=int((p & ~t).sum()),
                               fn=int((~p & t).sum()), tn=int((~p & ~t & valid).sum()))["iou"]


# ----------------------------------------------------------------------------- dev / test assets

DEV_STATS = {
    "s1": {"mean": [-10.0, -17.0], "std": [4.0, 4.5], "clip_min": [-30.0, -40.0], "clip_max": [0.0, -5.0]},
    "s2": {"mean": [1500.0] * 13, "std": [700.0] * 13, "clip_min": [0.0] * 13, "clip_max": [5000.0] * 13},
    "meta": {"split": "train", "note": "synthetic dev stats, not the real band_stats.json"},
}


def _dev_cfg(name: str, modalities: list[str], encoder: str) -> dict:
    cfg = {"name": name, "seed": 0,
           "data": {"modalities": modalities, "s2_bands": None},
           "model": {"encoder": encoder, "encoder_weights": None}}
    if name.startswith("fusion"):
        cfg["model"].update(arch="fusionnet", fusion="cross_attention", attention_heads=8)
    return cfg


def make_dev_assets(out_dir: str | Path, size: int = 128, n_chips: int = 2,
                    encoder: str = "resnet18", seed: int = 0) -> tuple[Path, dict[str, Path]]:
    """Synthetic chips + RANDOM-weight checkpoints so the app and its smoke test run without
    the real dataset or trained weights. Checkpoints are marked ``smoke`` and the app shows a
    warning banner. Returns (chips dir, {model name: checkpoint path})."""
    if size % 32:
        raise ValueError(f"size must be divisible by 32, got {size}")
    out_dir = Path(out_dir)
    chips_dir = out_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chips = []
    for i in range(n_chips):
        cid = f"Dev_{i}"
        y = np.zeros((size, size), np.int64)
        r, c = rng.integers(0, size // 2, size=2)
        y[r: r + size // 3, c: c + size // 3] = 1
        y[:2] = IGNORE_INDEX
        water = y == 1
        s1 = np.stack([np.where(water, -20.0, -8.0), np.where(water, -28.0, -15.0)])
        s1 = (s1 + rng.normal(0, 1, (2, size, size))).astype(np.float32)
        s1[:, -1, :4] = np.nan
        s2 = (rng.integers(500, 3000, (13, size, size)) - 400 * water).astype(np.float32)
        s2[:, -2:, -2:] = 0  # all-band nodata
        np.savez_compressed(chips_dir / f"{cid}.npz", chip_id=np.str_(cid),
                            s1=s1, s2=s2.astype(np.uint16), label=y.astype(np.int8))
        chips.append({"chip_id": cid, "region": "Dev", "cloud_fraction": 0.0,
                      "water_fraction": float(water.mean()), "file": f"{cid}.npz"})
    (chips_dir / MANIFEST).write_text(json.dumps(
        {"rule": "DEV MODE: synthetic chips, random weights", "split": "dev", "chips": chips}, indent=2) + "\n")

    weights = {}
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for name, mods in (("s2_only_robust", ["s2"]), ("fusion_robust", ["s1", "s2"])):
        cfg = _dev_cfg(name, mods, encoder)
        model = build_model(cfg, encoder_weights=None)
        path = weights_dir / WEIGHT_FILES[name]
        torch.save({"model": model.state_dict(), "config": cfg, "stats": DEV_STATS,
                    "epoch": 0, "best_val_iou": 0.0, "val_iou": 0.0, "smoke": True}, path)
        weights[name] = path
    return chips_dir, weights
