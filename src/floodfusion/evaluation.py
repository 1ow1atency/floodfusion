"""Evaluate a checkpoint on full chips of one split: global metrics (JSON) + per-chip CSV.

Uses the config and normalisation stats stored in the checkpoint, so inputs are built
exactly as in training. Metrics are pooled confusion counts (floodfusion.metrics); the
per-chip CSV can be joined with docs/cloud_cover.csv on chip_id.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Sen1Floods11
from .metrics import WaterConfusion
from .models import build_model
from .training import _autocast, check_modalities, forward_model, git_info, load_checkpoint

EVAL_SPLITS = ("valid", "test", "bolivia")
PER_CHIP_COLUMNS = ("chip_id", "tp", "fp", "fn", "tn", "n_valid", "iou")


def evaluate(checkpoint: str | Path, split: str, *, root: str | None = None, out_dir: str | Path | None = None,
             batch_size: int = 4, allow_missing: bool = False, device: torch.device | None = None,
             log=print) -> dict:
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    checkpoint = Path(checkpoint)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    cfg, stats = ckpt["config"], ckpt["stats"]
    mods = check_modalities(cfg)
    if ckpt.get("smoke"):
        log("WARNING: this is a SMOKE-TEST checkpoint; its metrics are not a result.")

    # Pretrained weights are not needed: all weights come from the checkpoint.
    model = build_model(cfg, encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = Sen1Floods11(root or cfg["data"]["root"], split, modalities=mods, s2_bands=cfg["data"].get("s2_bands"),
                      stats=stats, allow_missing=allow_missing)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    conf = WaterConfusion()
    with torch.no_grad():
        for batch in loader:
            with _autocast(device, amp):
                logits = forward_model(model, batch, cfg, device)
            conf.update(logits.float(), batch["label"].to(device), list(batch["id"]))

    out_dir = Path(out_dir) if out_dir else checkpoint.parent / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = conf.compute()
    result = {
        "split": split, "cloud_setting": "real S2 as acquired (no synthetic clouds)",
        "seed": cfg["seed"], "run": cfg["name"], "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt["epoch"], "best_val_iou": ckpt["best_val_iou"],
        "smoke_checkpoint": bool(ckpt.get("smoke")), "n_chips": len(ds),
        "metrics": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()},
        "mean_chip_iou_secondary": _none_if_nan(conf.mean_chip_iou()),
        "train_git_commit": ckpt.get("git_commit"), "eval": git_info(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(out_dir / "per_chip.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PER_CHIP_COLUMNS)
        for cid, r in conf.per_chip().items():
            w.writerow([cid, r["tp"], r["fp"], r["fn"], r["tn"], r["n_valid"],
                        "" if math.isnan(r["iou"]) else f"{r['iou']:.6f}"])
    log(f"split={split} | cloud setting: real | seed={cfg['seed']} | {len(ds)} chips | "
        f"water IoU {m['iou']:.4f} | F1 {m['f1']:.4f} | precision {m['precision']:.4f} | recall {m['recall']:.4f}")
    log(f"wrote {out_dir / 'metrics.json'} and {out_dir / 'per_chip.csv'}")
    return result


def _none_if_nan(v: float):
    return None if math.isnan(v) else v


def format_eval_table(run_dir: str | Path, splits=EVAL_SPLITS) -> str:
    """Table of water IoU, F1, precision, recall from <run_dir>/eval_<split>/metrics.json."""
    def f(v):
        return "   n/a" if v is None else f"{v:6.4f}"

    lines, meta = [], None
    head = f"{'split':<9}{'chips':>6}{'IoU':>9}{'F1':>9}{'prec':>9}{'recall':>9}"
    for split in splits:
        path = Path(run_dir) / f"eval_{split}" / "metrics.json"
        if not path.is_file():
            lines.append(f"{split:<9}{'(not evaluated)':>24}")
            continue
        d = json.loads(path.read_text())
        meta = meta or d
        m = d["metrics"]
        lines.append(f"{split:<9}{d['n_chips']:>6}   {f(m['iou'])}   {f(m['f1'])}   "
                     f"{f(m['precision'])}   {f(m['recall'])}")
    title = (f"{meta['run']} | seed {meta['seed']} | best.pt from epoch {meta['checkpoint_epoch']} | "
             f"cloud setting: real (no synthetic clouds)" if meta else str(run_dir))
    return "\n".join([title, head, "-" * len(head), *lines])
