"""Robustness evaluation of trained models on test and bolivia.

FINAL REPORTING ONLY: nothing computed here may be used to tune hyperparameters or to select
models or epochs. Every model is its ``best.pt``, chosen earlier on the real valid split.

Settings, per split (test, bolivia):
  * clean: real S2 as acquired. Must reproduce eval_<split>/metrics.json exactly (sanity check).
  * thick_clouds: thick synthetic clouds as used in training (clouds_synth, thin=False, large
    or cumulus drawn per cloud) at coverage 20/40/60/80/100%, for 3 cloud seeds. The cloud of a
    chip depends only on (chip id, coverage, cloud seed) (``eval_cloud_seed``), so every model
    sees identical clouds. Coverage 0 = clean. These clouds are semi-transparent (median opacity
    ~0.8 inside the mask): water stays visible through them (docs/cloud_leakage.md).
  * opaque_clouds: the same clouds (same masks, seeds and coverage) with opacity 1 inside the
    mask (clouds_synth.opaque_alpha), so no ground signal survives under the cloud: the stricter
    test. Also run at coverage 0, where it must reproduce clean exactly (sanity check), and at
    100% s2-only models are expected to collapse (checked, warning only).
  * s2_removed / s1_removed: the modality is missing. FusionNet: zero input + modality mask;
    U-Net: that modality's channels set to 0 after normalisation (= train band mean).
  * real_clouds: the clean predictions pooled per s2cloudless cloud-cover bin (docs/cloud_cover.csv).

Not applicable / unaffected (marked in the outputs, never silently filled):
  * a model without S2 is unaffected by clouds and by removing S2: its clean result is reported;
  * a model without S1 is unaffected by removing S1;
  * removing the only input modality is "n/a" (s2_only without S2 = beyond 100% cloud).

Efficiency: for each (split, setting) the chips are loaded and clouded once and every model is
run on the same batch. ``CloudCache`` keeps each cloud layer on disk (``cache_dir``, or a
temporary folder for one run), so each cloud is generated once: the opaque pass and later runs
reuse it. It stores the cloud (opacity, mask, shade, spectrum: ~2.3 MB per 512 px chip) rather
than the clouded 13-band S2 (~14 MB), and ``apply_clouds`` rebuilds the clouded S2 bit-identically.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .chips import IGNORE_INDEX, load_label, load_raw
from .clouds import CLOUD_BIN_LABELS, cloud_bin, read_cloud_csv
from .clouds_synth import apply_clouds, cloud_params, eval_cloud_seed, opaque_alpha
from .data import Sen1Floods11
from .metrics import WaterConfusion, metrics_from_counts
from .models import build_model
from .training import _autocast, check_modalities, forward_model, load_checkpoint

SPLITS = ("test", "bolivia")
CLOUD_SETTINGS = ("thick_clouds", "opaque_clouds")
COVERAGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CLOUD_SEEDS = (0, 1, 2)
MIN_BIN_CHIPS = 10  # real-cloud bins with fewer chips are flagged unreliable
# Qualitative figure (cloud seed 0): S2 clear / thick 100% / opaque 100%, SAR, label, then each
# robust model's prediction under thick and under opaque cloud, side by side (shows the leak).
QUAL_MODELS = ("s2_only_robust", "early_fusion_robust", "fusion_robust")
QUAL_SETTINGS = (("thick_clouds", 1.0), ("opaque_clouds", 1.0))
QUAL_CHIPS = 4
CSV_COLUMNS = ("model", "split", "setting", "coverage", "cloud_seed", "iou", "precision", "recall",
               "n_chips", "note")
NOTE_NO_S2 = "unaffected: model has no S2 input (clean result)"
NOTE_NO_S1 = "unaffected: model has no S1 input (clean result)"
NOTE_NA_S2 = "n/a: S2 is the only input (no S2-removed result beyond 100% cloud)"
NOTE_NA_S1 = "n/a: S1 is the only input"
NOTE_UNRELIABLE = f"unreliable: fewer than {MIN_BIN_CHIPS} chips"
NOTE_ZERO_THICK = "coverage 0 = clean (real S2 as acquired)"
NOTE_ZERO_OPAQUE = "coverage 0 through the opaque path: checked equal to clean"
OPAQUE_DROP_MAX = 0.5  # s2-only models at 100% opaque cloud are expected below this fraction of their clean IoU
AUDIT_NOTE = ("Real-cloud bins use s2cloudless masks; the mask audit (docs/ROADMAP.md, Stage 5 TODO) is "
              "still pending, so treat these bins as indicative.")


# ----------------------------------------------------------------------------- clouds

class CloudCache:
    """Evaluation clouds per (chip, coverage, cloud seed): thick as generated, or opaque (the
    same cloud with opacity 1 inside its mask). With ``cache_dir`` each cloud layer is generated
    once and stored as .npz (exact float32), then reused by the opaque pass and later runs.
    Picklable (DataLoader workers)."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generated = 0  # clouds generated in this process (not loaded from disk)

    def layer(self, chip_id: str, h: int, w: int, coverage: float, cloud_seed: int, opaque: bool = False):
        """(alpha, shade, spectrum) of the chip's cloud; ``opaque``: alpha 1 inside the cloud's mask."""
        path = (self.cache_dir / f"{chip_id}_c{int(round(coverage * 100)):03d}_s{cloud_seed}_{h}x{w}.npz"
                if self.cache_dir else None)
        z = None
        if path is not None and path.is_file():
            with np.load(path) as f:
                if "mask" in f:  # files written before the opaque setting have no mask: regenerate
                    z = {k: f[k] for k in ("alpha", "mask", "shade", "spectrum")}
        if z is None:
            alpha, mask, shade, spectrum = cloud_params(h, w, coverage, eval_cloud_seed(chip_id, coverage, cloud_seed),
                                                        thin=False)
            z = {"alpha": alpha, "mask": mask, "shade": shade, "spectrum": spectrum}
            self.generated += 1
            if path is not None:
                tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
                np.savez(tmp, **z)
                os.replace(tmp, path)
        alpha = opaque_alpha(z["alpha"], z["mask"]) if opaque else z["alpha"]
        return alpha, z["shade"], z["spectrum"]

    def apply(self, s2: np.ndarray, chip_id: str, coverage: float, cloud_seed: int, opaque: bool = False) -> np.ndarray:
        """Raw S2 (13, H, W) with the chip's evaluation cloud. Thick at coverage 0 returns it
        unchanged; opaque is always blended (at coverage 0 its mask is empty, so the result must
        equal the input: the sanity check of the opaque path)."""
        if coverage <= 0.0 and not opaque:
            return s2
        alpha, shade, spectrum = self.layer(chip_id, s2.shape[1], s2.shape[2], coverage, cloud_seed, opaque)
        return apply_clouds(s2, alpha, spectrum, shade)


class CloudHook:
    """Dataset ``s2_augment`` hook: the evaluation cloud for one (coverage, cloud seed), thick or opaque."""

    def __init__(self, coverage: float, cloud_seed: int, cache: CloudCache, opaque: bool = False):
        self.coverage, self.cloud_seed, self.cache, self.opaque = coverage, cloud_seed, cache, opaque

    def __call__(self, s2: np.ndarray, chip_id: str) -> np.ndarray:
        return self.cache.apply(s2, chip_id, self.coverage, self.cloud_seed, self.opaque)


# ----------------------------------------------------------------------------- models

@dataclass
class EvalModel:
    label: str
    run_dir: Path
    cfg: dict
    stats: dict
    mods: tuple[str, ...]
    family: str
    robust: bool
    epoch: int
    smoke: bool
    model: torch.nn.Module | None = None


def find_runs(runs_root: str | Path) -> list[Path]:
    """Run folders directly under ``runs_root`` that contain a best.pt, sorted by name."""
    return sorted(p.parent for p in Path(runs_root).glob("*/best.pt"))


def load_models(run_dirs: Sequence[str | Path], device: torch.device, *, allow_smoke: bool = False,
                log=print) -> list[EvalModel]:
    out = []
    for rd in map(Path, run_dirs):
        ckpt = load_checkpoint(rd / "best.pt", map_location="cpu")
        cfg = ckpt["config"]
        if ckpt.get("smoke") and not allow_smoke:
            raise ValueError(f"{rd}: best.pt is a smoke-test checkpoint (pass allow_smoke for a pipeline check)")
        model = build_model(cfg, encoder_weights=None)
        model.load_state_dict(ckpt["model"])
        aug = cfg.get("augment") or {}
        robust = cfg["name"].endswith("_robust") or any((aug.get(k) or {}).get("enabled") for k in aug)
        out.append(EvalModel(label=cfg["name"], run_dir=rd, cfg=cfg, stats=ckpt["stats"], mods=check_modalities(cfg),
                             family=cfg["name"].removesuffix("_robust"), robust=robust, epoch=ckpt["epoch"],
                             smoke=bool(ckpt.get("smoke")), model=model.to(device).eval()))
    from .viz import FAMILY_ORDER

    # fixed order: family (as in the figures), non-robust before robust, then name and seed
    out.sort(key=lambda m: (FAMILY_ORDER.index(m.family) if m.family in FAMILY_ORDER else len(FAMILY_ORDER),
                            m.family, m.robust, m.label, m.cfg["seed"]))
    names = [m.label for m in out]
    for m in out:  # several seeds of one config: keep them apart
        if names.count(m.label) > 1:
            m.label = f"{m.label}_seed{m.cfg['seed']}"
    if len({m.label for m in out}) != len(out):
        raise ValueError(f"duplicate runs: {sorted(names)}")
    ref = out[0] if out else None
    for m in out[1:]:  # one shared input pipeline: same normalisation and bands for every model
        if m.stats != ref.stats or m.cfg["data"].get("s2_bands") != ref.cfg["data"].get("s2_bands"):
            raise ValueError(f"{m.run_dir} and {ref.run_dir} use different normalisation stats or S2 bands")
    for m in out:
        log(f"  {m.label:<24s} best.pt epoch {m.epoch:>3}  modalities {list(m.mods)}"
            f"{'  [SMOKE]' if m.smoke else ''}  ({m.run_dir})")
    return out


def status(mods: tuple[str, ...], setting: str) -> str:
    """'eval', or the note explaining why the setting is not run for this model."""
    if setting in ("clean", "real_clouds"):
        return "eval"
    if setting in CLOUD_SETTINGS:
        return "eval" if "s2" in mods else NOTE_NO_S2
    if setting == "s2_removed":
        return NOTE_NO_S2 if "s2" not in mods else NOTE_NA_S2 if mods == ("s2",) else "eval"
    if setting == "s1_removed":
        return NOTE_NO_S1 if "s1" not in mods else NOTE_NA_S1 if mods == ("s1",) else "eval"
    raise ValueError(f"unknown setting {setting!r}")


def remove_modality(batch: dict, modality: str) -> dict:
    """Batch with ``modality`` missing: zeros (= train mean after normalisation) and the mask."""
    out = dict(batch)
    out[modality] = torch.zeros_like(batch[modality])
    mask = torch.ones(batch[modality].shape[0], 2, dtype=torch.bool)
    mask[:, ("s1", "s2").index(modality)] = False
    out["modality_mask"] = mask
    return out


# ----------------------------------------------------------------------------- evaluation

def _settings() -> list[tuple[str, float, int | None]]:
    out = [("clean", 0.0, None)]
    out += [("thick_clouds", c, s) for c in COVERAGES if c > 0 for s in CLOUD_SEEDS]
    out += [("opaque_clouds", 0.0, CLOUD_SEEDS[0])]  # sanity check: must equal clean
    out += [("opaque_clouds", c, s) for c in COVERAGES if c > 0 for s in CLOUD_SEEDS]
    out += [("s2_removed", math.nan, None), ("s1_removed", math.nan, None)]
    return out


def evaluate_split(models: Sequence[EvalModel], split: str, root: str | Path, *, cache: CloudCache,
                   device: torch.device, batch_size: int = 4, num_workers: int = 0, allow_missing: bool = False,
                   qual_ids: Iterable[str] = (), log=print) -> tuple[dict, dict, int]:
    """Run every applicable (setting, model) on one split.

    Returns ({(label, setting, coverage, cloud_seed): WaterConfusion}, {(label, setting, coverage):
    {chip: predicted labels}} for ``qual_ids`` and QUAL_MODELS in QUAL_SETTINGS with cloud seed 0, n_chips)."""
    mods = tuple(m for m in ("s1", "s2") if any(m in em.mods for em in models))
    ref = models[0]
    qual_ids = set(qual_ids)
    confs, preds, n_chips = {}, {}, 0
    for setting, cov, seed in _settings():
        active = [m for m in models if status(m.mods, setting) == "eval"]
        if not active:
            continue
        t0 = time.time()
        hook = CloudHook(cov, seed, cache, opaque=setting == "opaque_clouds") if setting in CLOUD_SETTINGS else None
        ds = Sen1Floods11(root, split, modalities=mods, s2_bands=ref.cfg["data"].get("s2_bands"), stats=ref.stats,
                          s2_augment=hook, allow_missing=allow_missing)
        n_chips = len(ds)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        cur = {m.label: WaterConfusion() for m in active}
        keep = (setting, cov) in QUAL_SETTINGS and seed == CLOUD_SEEDS[0]
        with torch.no_grad():
            for batch in loader:
                if setting == "s2_removed":
                    batch = remove_modality(batch, "s2")
                elif setting == "s1_removed":
                    batch = remove_modality(batch, "s1")
                y = batch["label"].to(device)
                for m in active:
                    amp = bool(m.cfg["train"].get("amp", True)) and device.type == "cuda"
                    with _autocast(device, amp):
                        logits = forward_model(m.model, batch, m.cfg, device)
                    cur[m.label].update(logits.float(), y, list(batch["id"]))
                    if keep and qual_ids and m.label in QUAL_MODELS:
                        lab = logits.argmax(1).cpu().numpy().astype(np.uint8)
                        for i, cid in enumerate(batch["id"]):
                            if cid in qual_ids:
                                preds.setdefault((m.label, setting, cov), {})[cid] = lab[i]
        for label, conf in cur.items():
            confs[(label, setting, cov, seed)] = conf
        what = setting if setting not in CLOUD_SETTINGS else f"{setting.split('_')[0]} clouds {cov:.0%} seed {seed}"
        log(f"  {split}: {what:<22s} {len(active)} model(s), {n_chips} chips, {time.time() - t0:.0f}s")
    return confs, preds, n_chips


def check_clean_matches_eval(models: Sequence[EvalModel], confs: dict, split: str) -> None:
    """Coverage 0 must reproduce eval_<split>/metrics.json (same best.pt) exactly; raises otherwise."""
    problems = []
    for m in models:
        path = m.run_dir / f"eval_{split}" / "metrics.json"
        if not path.is_file():
            problems.append(f"{m.label}: {path} missing (run scripts/evaluate.py --split {split} first)")
            continue
        ref = json.loads(path.read_text())
        if ref.get("checkpoint_epoch") != m.epoch:
            problems.append(f"{m.label}: {path} is from best.pt epoch {ref.get('checkpoint_epoch')}, "
                            f"best.pt is now epoch {m.epoch} (re-run evaluate.py)")
            continue
        got = confs[(m.label, "clean", 0.0, None)].compute()
        want = ref["metrics"]
        diff = {k: (got[k], want[k]) for k in ("tp", "fp", "fn", "tn") if got[k] != want[k]}
        if diff:
            problems.append(f"{m.label}: clean confusion counts differ from {path}: {diff}")
    if problems:
        raise RuntimeError(f"SANITY CHECK FAILED on {split} (coverage 0 must equal the existing evaluation):\n  "
                           + "\n  ".join(problems))


def check_opaque_zero_matches_clean(models: Sequence[EvalModel], confs: dict, split: str) -> None:
    """Opaque clouds at coverage 0 (empty mask, blended through the opaque path) must give
    exactly the clean confusion counts, chip by chip; raises otherwise."""
    problems = []
    for m in models:
        key = (m.label, "opaque_clouds", 0.0, CLOUD_SEEDS[0])
        if key not in confs:
            continue  # model without S2: unaffected
        got, want = confs[key].per_chip(), confs[(m.label, "clean", 0.0, None)].per_chip()
        bad = sorted(c for c in want if {k: got[c][k] for k in ("tp", "fp", "fn", "tn")}
                     != {k: want[c][k] for k in ("tp", "fp", "fn", "tn")})
        if bad or set(got) != set(want):
            problems.append(f"{m.label}: {len(bad)} chip(s) differ, e.g. {bad[:3]}")
    if problems:
        raise RuntimeError(f"SANITY CHECK FAILED on {split} (opaque clouds at coverage 0 must equal clean):\n  "
                           + "\n  ".join(problems))


def opaque_drop(models: Sequence[EvalModel], rows: list[dict], split: str) -> tuple[list[str], list[str]]:
    """(log lines: clean -> thick 100% -> opaque 100% -> S2 removed per model with S2,
    warnings: s2-only models that keep more than OPAQUE_DROP_MAX of their clean IoU at 100%
    opaque cloud, which should be impossible without ground signal)."""
    lines, warnings = [], []
    for m in models:
        if "s2" not in m.mods:
            continue
        clean = dict((c, mu) for c, mu, _ in sweep_stats(rows, m.label, split, setting="thick_clouds"))[0.0]
        thick = dict((c, mu) for c, mu, _ in sweep_stats(rows, m.label, split, setting="thick_clouds")).get(1.0)
        opaque = dict((c, mu) for c, mu, _ in sweep_stats(rows, m.label, split, setting="opaque_clouds")).get(1.0)
        r = next(r for r in rows if r["model"] == m.label and r["split"] == split and r["setting"] == "s2_removed")
        removed = "n/a" if r["note"].startswith("n/a") else f"{_num(r['iou']):.4f}"
        lines.append(f"{m.label}: clean {clean:.4f} -> thick 100% {thick:.4f} -> opaque 100% {opaque:.4f} "
                     f"-> S2 removed {removed}")
        if m.mods == ("s2",) and not opaque < OPAQUE_DROP_MAX * clean:
            warnings.append(f"WARNING {split}: {m.label} keeps {opaque:.4f} water IoU at 100% opaque cloud "
                            f"(clean {clean:.4f}); an S2-only model should collapse without ground signal. "
                            "Check the opaque setting before using these results.")
    return lines, warnings


def real_cloud_bins(per_chip: dict[str, dict], cloud_rows: Iterable[dict], split: str) -> list[dict]:
    """Pooled metrics per s2cloudless cloud bin from per-chip counts ({chip: {tp, fp, fn, tn,
    ...}}). Chips without a cloud fraction (no valid S2) are left out and counted."""
    frac = {r["chip_id"]: r["cloud_fraction"] for r in cloud_rows if r["split"] == split}
    bins: dict[str, list[str]] = {b: [] for b in CLOUD_BIN_LABELS}
    unbinned = []
    for cid in per_chip:
        b = cloud_bin(frac[cid]) if cid in frac else None
        (bins[b] if b else unbinned).append(cid)
    out = []
    for b, ids in bins.items():
        counts = {k: sum(per_chip[c][k] for c in ids) for k in ("tp", "fp", "fn", "tn")}
        m = metrics_from_counts(**counts)
        note = "no chips" if not ids else NOTE_UNRELIABLE if len(ids) < MIN_BIN_CHIPS else ""
        out.append({"bin": b, "n_chips": len(ids), "iou": m["iou"], "precision": m["precision"],
                    "recall": m["recall"], "note": note, "unreliable": len(ids) < MIN_BIN_CHIPS})
    out.append({"bin": "no cloud fraction", "n_chips": len(unbinned), "ids": unbinned})
    return out


def pick_qual_chips(root: str | Path, split: str, cloud_rows: Iterable[dict], ids: Sequence[str],
                    n: int = QUAL_CHIPS) -> list[str]:
    """Fixed rule: chips with 0% real cloud (s2cloudless), highest water fraction first (ties by
    id). If fewer than n chips are cloud-free, the least cloudy chips fill the list."""
    frac = {r["chip_id"]: r["cloud_fraction"] for r in cloud_rows if r["split"] == split}
    cand = [c for c in ids if c in frac and not math.isnan(frac[c])]

    def water(c):
        lab = load_label(root, c)
        labelled = (lab != IGNORE_INDEX).sum()
        return float((lab == 1).sum()) / labelled if labelled else 0.0

    clear = sorted((c for c in cand if frac[c] == 0.0), key=lambda c: (-water(c), c))
    rest = sorted((c for c in cand if frac[c] > 0.0), key=lambda c: (frac[c], -water(c), c))
    return (clear + rest)[:n]


# ----------------------------------------------------------------------------- rows and summary

def _fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}"


def build_rows(models: Sequence[EvalModel], split: str, confs: dict, n_chips: int, cloud_rows: list[dict]) -> list[dict]:
    rows = []

    def add(m, setting, cov, seed, metrics=None, n=n_chips, note=""):
        rows.append({"model": m.label, "split": split, "setting": setting,
                     "coverage": cov if isinstance(cov, str) else ("" if cov is None or math.isnan(cov) else f"{cov:.1f}"),
                     "cloud_seed": "" if seed is None else seed,
                     "iou": _fmt(metrics["iou"]) if metrics else "", "precision": _fmt(metrics["precision"]) if metrics else "",
                     "recall": _fmt(metrics["recall"]) if metrics else "", "n_chips": n, "note": note})

    for m in models:
        clean = confs[(m.label, "clean", 0.0, None)].compute()
        add(m, "thick_clouds", 0.0, None, clean, note=NOTE_ZERO_THICK)
        for setting, cov, seed in _settings()[1:]:
            st = status(m.mods, setting)
            if st == "eval":
                add(m, setting, cov, seed, confs[(m.label, setting, cov, seed)].compute(),
                    note=NOTE_ZERO_OPAQUE if setting == "opaque_clouds" and cov == 0.0 else "")
            elif st.startswith("unaffected"):
                add(m, setting, cov, seed, clean, note=st)
            else:
                add(m, setting, cov, seed, None, n=0, note=st)
        for b in real_cloud_bins(confs[(m.label, "clean", 0.0, None)].per_chip(), cloud_rows, split):
            if b["bin"] in CLOUD_BIN_LABELS:
                add(m, "real_clouds", b["bin"], None, b if b["n_chips"] else None, n=b["n_chips"], note=b["note"])
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> float:
    return float(v) if v not in ("", None) else math.nan


def sweep_stats(rows: list[dict], model: str, split: str, metric: str = "iou",
                setting: str = "thick_clouds") -> list[tuple[float, float, float]]:
    """[(coverage, mean, std over cloud seeds)] of a synthetic cloud sweep (``setting`` thick_clouds
    or opaque_clouds; sample std, ddof=1; 0 at coverage 0 and for models unaffected by clouds)."""
    if setting not in CLOUD_SETTINGS:
        raise ValueError(f"setting must be one of {CLOUD_SETTINGS}, got {setting!r}")
    out = []
    for cov in COVERAGES:
        vals = [_num(r[metric]) for r in rows if r["model"] == model and r["split"] == split
                and r["setting"] == setting and abs(_num(r["coverage"]) - cov) < 1e-9]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            out.append((cov, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0))
    return out


def _sweep_table(rows, labels, split, setting) -> list[str]:
    L = ["| model | " + " | ".join(f"{c:.0%}" for c in COVERAGES) + " |", "|---|" + "--:|" * len(COVERAGES)]
    for lab in labels:
        st = {c: (mu, sd) for c, mu, sd in sweep_stats(rows, lab, split, setting=setting)}
        unaffected = any(r["note"] == NOTE_NO_S2 for r in rows if r["model"] == lab and r["split"] == split
                         and r["setting"] == setting)
        cells = [f"{st[c][0]:.4f} ± {st[c][1]:.4f}" if c in st else "–" for c in COVERAGES]
        L.append(f"| {lab}{' †' if unaffected else ''} | " + " | ".join(cells) + " |")
    return L


def summary_markdown(rows: list[dict], models: Sequence[EvalModel], splits: Sequence[str],
                     unbinned: dict[str, int], cache_note: str = "", warnings: Sequence[str] = ()) -> str:
    labels = [m.label for m in models]
    L = ["# Robustness evaluation", "",
         "**Final reporting only.** Nothing here may be used to tune hyperparameters or to select models.",
         "Each model is its `best.pt`, selected on the real valid split. Metric: pooled water IoU "
         "(label `-1` ignored); precision and recall of the water class.", ""]
    L += ["| model | run | best.pt epoch | modalities | robustness training |", "|---|---|--:|---|---|"]
    for m in models:
        L.append(f"| {m.label} | `{m.run_dir.name}` | {m.epoch} | {', '.join(m.mods)} | {'yes' if m.robust else 'no'} |")
    if any(m.smoke for m in models):
        L += ["", "**WARNING: smoke-test checkpoints. These numbers are a pipeline check, not results.**"]
    if warnings:
        L += [""] + [f"**{w}**" for w in warnings]
    L += ["", "Synthetic clouds, two variants of the same cloud layers (same masks, cloud seeds and coverage):",
          "- **thick**: as used in robustness training (clouds_synth thick clouds, large or cumulus). They are "
          "semi-transparent: median opacity ~0.8 inside the mask, so water stays visible through them "
          "(docs/cloud_leakage.md).",
          "- **opaque**: the same mask with opacity 1 inside it; the ground under the cloud is replaced "
          "entirely by the cloud spectrum. **The stricter test of cloud robustness.** Its coverage 0 is "
          "evaluated through the opaque path and checked equal to clean.",
          f"Each chip's cloud depends only on (chip id, coverage, cloud seed), so every model sees identical "
          f"clouds; {len(CLOUD_SEEDS)} cloud seeds per coverage, mean ± std over cloud seeds (sample std). "
          "Seed of every run: see the run folder.",
          "Missing modality: the input is set to 0 (= train band mean after normalisation); FusionNet also "
          "gets the modality mask. S2 removed is the extreme beyond 100% cloud.", ""]
    for split in splits:
        n = next((r["n_chips"] for r in rows if r["split"] == split and r["setting"] == "thick_clouds"), "?")
        L += [f"## {split} ({n} chips)", "", "### Opaque synthetic clouds (stricter test): water IoU", ""]
        L += _sweep_table(rows, labels, split, "opaque_clouds")
        L += ["", "### Thick synthetic clouds (as trained, semi-transparent): water IoU", ""]
        L += _sweep_table(rows, labels, split, "thick_clouds")
        L += ["", "### At 100% cloud: water IoU (precision / recall)", "",
              "| model | clean | thick 100% | opaque 100% | S2 removed |", "|---|--:|--:|--:|--:|"]
        for lab in labels:
            cells = []
            for setting, cov in (("thick_clouds", 0.0), ("thick_clouds", 1.0), ("opaque_clouds", 1.0)):
                st = {k: {c: (mu, sd) for c, mu, sd in sweep_stats(rows, lab, split, k, setting)}
                      for k in ("iou", "precision", "recall")}
                iou, sd = st["iou"][cov]
                cells.append(f"{iou:.4f}{f' ± {sd:.4f}' if cov else ''} "
                             f"({st['precision'][cov][0]:.3f} / {st['recall'][cov][0]:.3f})")
            r = next(r for r in rows if r["model"] == lab and r["split"] == split and r["setting"] == "s2_removed")
            cells.append("n/a" if r["note"].startswith("n/a") else
                         f"{_num(r['iou']):.4f} ({_num(r['precision']):.3f} / {_num(r['recall']):.3f})")
            unaffected = "s2" not in next(m.mods for m in models if m.label == lab)
            L.append(f"| {lab}{' †' if unaffected else ''} | " + " | ".join(cells) + " |")
        L += ["", "### Missing modality: water IoU (precision / recall)", "",
              "| model | clean | S2 removed | S1 removed |", "|---|--:|--:|--:|"]
        for lab in labels:
            cells = []
            for setting in ("thick_clouds", "s2_removed", "s1_removed"):
                r = next(r for r in rows if r["model"] == lab and r["split"] == split and r["setting"] == setting
                         and (setting != "thick_clouds" or r["coverage"] == "0.0"))
                if r["note"].startswith("n/a"):
                    cells.append("n/a")
                else:
                    mark = " †" if r["note"].startswith("unaffected") else ""
                    cells.append(f"{_num(r['iou']):.4f} ({_num(r['precision']):.3f} / {_num(r['recall']):.3f}){mark}")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L += ["", "### Real clouds (s2cloudless bins): water IoU (chips)", "",
              "| model | " + " | ".join(CLOUD_BIN_LABELS) + " |", "|---|" + "--:|" * len(CLOUD_BIN_LABELS)]
        for lab in labels:
            cells = []
            for b in CLOUD_BIN_LABELS:
                r = next(r for r in rows if r["model"] == lab and r["split"] == split and r["setting"] == "real_clouds"
                         and r["coverage"] == b)
                v = _num(r["iou"])
                cells.append("– (0)" if not int(r["n_chips"]) else
                             f"{v:.4f} ({r['n_chips']}){' ⚠' if r['note'].startswith('unreliable') else ''}")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L += ["", f"⚠ = {NOTE_UNRELIABLE}. Chips without a cloud fraction (no valid S2): {unbinned.get(split, 0)}.",
              f"_{AUDIT_NOTE}_", ""]
    L += ["† = unaffected by this setting (model lacks the changed modality): its clean result is shown.",
          "n/a = the setting removes the model's only input (for s2_only, S2 removed = beyond 100% cloud).", ""]
    if cache_note:
        L += [cache_note, ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- driver

def run_robustness(run_dirs: Sequence[str | Path], root: str | Path, out_dir: str | Path, *,
                   cloud_csv: str | Path = "docs/cloud_cover.csv", splits: Sequence[str] = SPLITS,
                   batch_size: int = 4, num_workers: int = 0, cache_dir: str | Path | None = None,
                   allow_missing: bool = False, allow_smoke: bool = False, qual_split: str = "test",
                   device: torch.device | None = None, figures: bool = True, log=print) -> dict:
    """Full evaluation; writes robustness.csv, summary.md and figures to ``out_dir``. Without
    ``cache_dir`` the cloud layers go to a temporary folder (deleted at the end), so each cloud
    is still generated only once (the opaque pass reuses the thick pass's layers)."""
    for s in splits:
        if s not in SPLITS:
            raise ValueError(f"robustness evaluation is for {SPLITS} only, got {s!r}")
    if not run_dirs:
        raise ValueError("no run folders with a best.pt to evaluate")
    with tempfile.TemporaryDirectory(prefix="floodfusion_clouds_") as tmp:
        return _run_robustness(run_dirs, root, out_dir, cloud_csv=cloud_csv, splits=splits, batch_size=batch_size,
                               num_workers=num_workers, cache=CloudCache(cache_dir or tmp),
                               cache_label=str(cache_dir) if cache_dir else "temporary (this run only)",
                               allow_missing=allow_missing, allow_smoke=allow_smoke, qual_split=qual_split,
                               device=device, figures=figures, log=log)


def _run_robustness(run_dirs, root, out_dir, *, cloud_csv, splits, batch_size, num_workers, cache: CloudCache,
                    cache_label, allow_missing, allow_smoke, qual_split, device, figures, log) -> dict:
    from . import viz

    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("Robustness evaluation: FINAL REPORTING on test/bolivia only (never for tuning or model selection).")
    log(f"device {device} | batch size {batch_size} | cloud cache {cache_label}")
    models = load_models(run_dirs, device, allow_smoke=allow_smoke, log=log)
    cloud_rows = read_cloud_csv(cloud_csv)
    rows, all_preds, qual, unbinned, warnings = [], {}, {}, {}, []
    for split in splits:
        ids = Sen1Floods11(root, split, modalities=("s1",), allow_missing=allow_missing).ids
        if split == qual_split:
            qual[split] = pick_qual_chips(root, split, cloud_rows, ids)
        confs, preds, n_chips = evaluate_split(models, split, root, cache=cache, device=device, batch_size=batch_size,
                                               num_workers=num_workers, allow_missing=allow_missing,
                                               qual_ids=qual.get(split, ()), log=log)
        check_clean_matches_eval(models, confs, split)
        log(f"  {split}: sanity check passed (coverage 0 = eval_{split}/metrics.json for every model)")
        check_opaque_zero_matches_clean(models, confs, split)
        log(f"  {split}: sanity check passed (opaque clouds at coverage 0 = clean, chip by chip)")
        split_rows = build_rows(models, split, confs, n_chips, cloud_rows)
        lines, warn = opaque_drop(models, split_rows, split)
        for line in lines:
            log(f"  {split}: {line}")
        for w in warn:
            log(w)
        if not warn and any(m.mods == ("s2",) for m in models):
            log(f"  {split}: sanity check passed (s2-only models below {OPAQUE_DROP_MAX:.0%} of clean IoU at 100% opaque)")
        rows += split_rows
        warnings += warn
        unbinned[split] = next(b["n_chips"] for b in real_cloud_bins(confs[(models[0].label, "clean", 0.0, None)].per_chip(),
                                                                     cloud_rows, split) if b["bin"] == "no cloud fraction")
        all_preds[split] = preds
    csv_path = write_csv(rows, out_dir / "robustness.csv")
    md = summary_markdown(rows, models, splits, unbinned, warnings=warnings,
                          cache_note=f"Clouds generated in this run (main process): {cache.generated}.")
    (out_dir / "summary.md").write_text(md + "\n")
    written = {"csv": csv_path, "summary": out_dir / "summary.md"}
    if figures:
        styles = viz.model_styles([(m.label, m.family, m.robust) for m in models])
        for split in splits:
            reference = {}
            for m in models:
                r = next(r for r in rows if r["model"] == m.label and r["split"] == split and r["setting"] == "s2_removed")
                if r["iou"] and "s2" in m.mods:
                    reference[m.label] = _num(r["iou"])
            panels = [("thick clouds (as trained, semi-transparent)",
                       {m.label: sweep_stats(rows, m.label, split, setting="thick_clouds") for m in models}),
                      ("opaque clouds (no ground signal under cloud)",
                       {m.label: sweep_stats(rows, m.label, split, setting="opaque_clouds") for m in models})]
            written[f"coverage_{split}"] = viz.plot_iou_vs_coverage(
                panels, styles, out_dir / f"iou_vs_coverage_{split}.png", reference=reference,
                title=f"{split}: water IoU vs synthetic cloud coverage (◆ = S2 removed)")
        written["real_bins"] = viz.plot_real_cloud_bins(rows, [m.label for m in models], styles, splits,
                                                         out_dir / "real_cloud_bins.png", note=AUDIT_NOTE)
        if qual.get(qual_split):
            written["qualitative"] = _qualitative(models, root, qual_split, qual[qual_split], all_preds[qual_split],
                                                  cache, out_dir / "qualitative.png")
        else:
            log(f"  qualitative figure skipped: split {qual_split!r} not evaluated or has no chips")
    for k, p in written.items():
        log(f"wrote {p}")
    return {"rows": rows, "models": models, "written": written, "clouds_generated": cache.generated,
            "warnings": warnings}


def _qualitative(models, root, split, chip_ids, preds, cache: CloudCache, out_path: Path) -> Path:
    from . import viz

    by_label = {m.label: m for m in models}
    shown = [lab for lab in QUAL_MODELS if lab in by_label and all((lab, *k) in preds for k in QUAL_SETTINGS)]

    def name(setting, cov):
        return f"{setting.split('_')[0]} {cov:.0%}"

    rows = []
    for cid in chip_ids:
        s2 = load_raw(root, cid, "s2")
        optical = [("S2 clear", s2)] + [(f"S2 {name(st, c)}", cache.apply(s2, cid, c, CLOUD_SEEDS[0],
                                                                          opaque=st == "opaque_clouds"))
                                        for st, c in QUAL_SETTINGS]
        rows.append({"chip_id": cid, "optical": optical, "s1": load_raw(root, cid, "s1"),
                     "label": load_label(root, cid),
                     "preds": [(lab, name(st, c), preds[(lab, st, c)][cid]) for lab in shown for st, c in QUAL_SETTINGS]})
    stats = models[0].stats
    return viz.plot_qualitative(rows, out_path, s1_range=(stats["s1"]["clip_min"][0], stats["s1"]["clip_max"][0]),
                                title=f"{split}: synthetic clouds (seed {CLOUD_SEEDS[0]}): thick = as trained, "
                                      "opaque = no ground signal; predictions of the robust models")
