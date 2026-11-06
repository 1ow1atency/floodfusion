"""Training loop for the U-Net baselines (SAR-only, optical-only, early fusion).

Rules enforced here (CLAUDE.md):
  * Training uses the train split, validation/model selection the valid split only.
    Configs naming "test" or "bolivia" for training or validation are refused.
  * Normalisation stats come from the stats file (train split only) and are stored in
    every checkpoint, so evaluation normalises exactly like training.
  * Seeds (Python, NumPy, PyTorch) come from the config.

Run folder: ``<output_dir>/<name>_seed<seed>`` (``..._smoke`` in smoke mode) with
config.yaml (resolved), run_info.json (git commit, seed, device, resumes), metrics.csv
(one row per epoch), last.pt (every epoch) and best.pt (best valid water IoU).

Resume: last.pt holds model, optimizer, scheduler, AMP scaler, completed epoch, best
score and RNG states (Python, NumPy, torch, CUDA, DataLoader generator). It is written
to a temporary file and renamed, so a disconnect cannot leave a half-written file.
``resume=True`` restores all of it and continues with the next epoch; metrics.csv
rows after the checkpoint's epoch are dropped. At most the interrupted epoch is lost.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .chips import S2_BANDS
from .data import Sen1Floods11
from .losses import CEDiceLoss
from .metrics import WaterConfusion
from .models import build_unet
from .transforms import TrainTransform
from .utils import seed_everything

TRAIN_SPLIT, VAL_SPLIT = "train", "valid"
EVAL_ONLY_SPLITS = ("test", "bolivia")
SMOKE_BATCHES = 2
SMOKE_VAL_CHIPS = 2
METRIC_COLUMNS = ("epoch", "lr", "train_loss", "val_loss", "val_iou", "val_f1", "val_precision",
                  "val_recall", "val_accuracy", "val_tp", "val_fp", "val_fn", "val_tn",
                  "best_val_iou", "epoch_time_s")
# Config keys that must not change when resuming (paths, workers and W&B may).
_RESUME_FIXED = {"data": ("stats", "modalities", "s2_bands", "train_split", "val_split", "crop_size"),
                 "model": ("encoder", "encoder_weights"),
                 "train": None}  # None = whole section
REPO_ROOT = Path(__file__).resolve().parents[2]

SMOKE_BANNER = ("\n" + "!" * 78 + "\n!!  SMOKE TEST: 2 train batches, 'validation' on 2 TRAIN chips of the local sample."
                "\n!!  These numbers are NOT a result. Never report or compare them.\n" + "!" * 78 + "\n")


# ----------------------------------------------------------------------------- config

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section, keys in {"data": ("root", "stats", "modalities", "crop_size", "num_workers"),
                          "model": ("encoder", "encoder_weights"),
                          "train": ("batch_size", "val_batch_size", "epochs", "lr", "weight_decay")}.items():
        missing = [k for k in keys if k not in cfg.get(section, {})]
        if missing:
            raise KeyError(f"{path}: config section '{section}' is missing {missing}")
    for k in ("name", "seed", "output_dir"):
        if k not in cfg:
            raise KeyError(f"{path}: config is missing '{k}'")
    return cfg


def resolve_config(cfg: dict, root: str | None = None, out: str | None = None,
                   no_wandb: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    if root is not None:
        cfg["data"]["root"] = root
    if out is not None:
        cfg["output_dir"] = out
    cfg["data"].setdefault("train_split", TRAIN_SPLIT)
    cfg["data"].setdefault("val_split", VAL_SPLIT)
    cfg["data"].setdefault("s2_bands", None)
    cfg["train"].setdefault("min_lr_ratio", 0.01)
    cfg["train"].setdefault("amp", True)
    cfg["train"].setdefault("ce_weight", 1.0)
    cfg["train"].setdefault("dice_weight", 1.0)
    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    if no_wandb:
        cfg["wandb"]["enabled"] = False
    return cfg


def check_splits(cfg: dict) -> None:
    """Refuse configs that train or validate on evaluation-only splits (CLAUDE.md)."""
    tr, va = cfg["data"]["train_split"], cfg["data"]["val_split"]
    for role, split in (("training", tr), ("validation", va)):
        if split in EVAL_ONLY_SPLITS:
            raise ValueError(f"Refusing to start: '{split}' is evaluation-only and must never be used "
                             f"for {role}, tuning, early stopping or model selection (CLAUDE.md).")
    if tr != TRAIN_SPLIT or va != VAL_SPLIT:
        raise ValueError(f"train_split must be '{TRAIN_SPLIT}' and val_split '{VAL_SPLIT}', got {tr!r}, {va!r}")


def check_modalities(cfg: dict) -> tuple[str, ...]:
    mods = tuple(cfg["data"]["modalities"])
    if mods not in (("s1",), ("s2",), ("s1", "s2")):
        raise ValueError(f"modalities must be [s1], [s2] or [s1, s2] (S1 first), got {list(mods)}")
    return mods


def in_channels(cfg: dict) -> int:
    mods = check_modalities(cfg)
    s2 = cfg["data"].get("s2_bands") or S2_BANDS
    return (2 if "s1" in mods else 0) + (len(s2) if "s2" in mods else 0)


def stack_inputs(batch: dict, modalities: tuple[str, ...]) -> torch.Tensor:
    """Model input: modalities concatenated on the channel axis in config order (S1 first)."""
    return torch.cat([batch[m] for m in modalities], dim=1)


def run_dir_for(cfg: dict, smoke: bool) -> Path:
    return Path(cfg["output_dir"]) / f"{cfg['name']}_seed{cfg['seed']}{'_smoke' if smoke else ''}"


def pick_device(smoke: bool = False) -> torch.device:
    if not smoke and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_info() -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")  # includes new untracked files; ignored files (outputs/) do not count
    return {"git_commit": commit or "unknown", "git_dirty": bool(dirty) if dirty is not None else None}


# ----------------------------------------------------------------------------- data

def load_stats(cfg: dict) -> dict:
    path = Path(cfg["data"]["stats"])
    if not path.is_file():
        raise FileNotFoundError(f"stats file not found: {path} (relative paths are resolved from the "
                                f"current directory; run from the repo root)")
    stats = json.loads(path.read_text())
    meta = stats.get("meta", {})
    if meta.get("split", "train") != "train":
        raise ValueError(f"{path}: stats must come from the train split, got {meta.get('split')!r}")
    return stats


def make_loaders(cfg: dict, stats: dict, smoke: bool, device: torch.device, generator: torch.Generator):
    d, t = cfg["data"], cfg["train"]
    mods = check_modalities(cfg)
    common = dict(root=d["root"], modalities=mods, s2_bands=d.get("s2_bands"), stats=stats, allow_missing=smoke)
    train_ds = Sen1Floods11(split=d["train_split"], transform=TrainTransform(d["crop_size"]), **common)
    if smoke:
        val_ds = Subset(Sen1Floods11(split=d["train_split"], **common), range(min(SMOKE_VAL_CHIPS, len(train_ds))))
        bs, val_bs, workers = min(t["batch_size"], 2), 1, 0
    else:
        val_ds = Sen1Floods11(split=d["val_split"], **common)
        bs, val_bs, workers = t["batch_size"], t["val_batch_size"], d["num_workers"]
    loader_kw = dict(num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, generator=generator, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, **loader_kw)
    if len(train_loader) == 0:
        raise ValueError(f"train split has {len(train_ds)} chips, fewer than batch_size {bs}")
    return train_loader, val_loader


# ----------------------------------------------------------------------------- loops

def _autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, amp, mods,
                    max_batches: int | None = None) -> float:
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = stack_inputs(batch, mods).to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(x)
        loss = loss_fn(logits, y)  # computed in float32
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss {loss.item()} at batch {i}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp, mods, max_batches: int | None = None):
    model.eval()
    conf = WaterConfusion()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = stack_inputs(batch, mods).to(device)
        y = batch["label"].to(device)
        with _autocast(device, amp):
            logits = model(x)
        total += loss_fn(logits, y).item()
        n += 1
        conf.update(logits.float(), y, list(batch["id"]))
    return conf, total / max(n, 1)


# ----------------------------------------------------------------------------- checkpoints

def _rng_state(generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def _set_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    # weights_only=False: our own checkpoints also hold config, stats and RNG states.
    return torch.load(path, map_location=map_location, weights_only=False)


def _fixed_part(cfg: dict) -> dict:
    return {s: (cfg[s] if keys is None else {k: cfg[s].get(k) for k in keys}) for s, keys in _RESUME_FIXED.items()}


# ----------------------------------------------------------------------------- metrics.csv

def _read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_metrics(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _fmt(v):
    return "" if isinstance(v, float) and math.isnan(v) else v


# ----------------------------------------------------------------------------- main entry

def run(cfg: dict, *, resume: bool = False, smoke: bool = False, max_batches: int | None = None,
        device: str | torch.device | None = None, log=print) -> Path:
    """Train according to ``cfg`` (already resolved). Returns the run folder."""
    check_splits(cfg)
    mods = check_modalities(cfg)
    if smoke:
        cfg = copy.deepcopy(cfg)
        cfg["wandb"]["enabled"] = False
        cfg["train"]["epochs"] = 1  # one epoch of SMOKE_BATCHES train + val batches
        max_batches = SMOKE_BATCHES
        log(SMOKE_BANNER)
    run_dir = run_dir_for(cfg, smoke)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"--resume: no checkpoint at {last_path}")
    if not resume and last_path.is_file():
        raise FileExistsError(f"{run_dir} already has a run. Use --resume to continue it, or another --out.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(device) if device is not None else pick_device(smoke)
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    if device.type == "cpu" and not smoke:
        log("WARNING: training on CPU; this will be very slow. Use a Colab GPU runtime.")

    stats = load_stats(cfg)
    train_loader, val_loader = make_loaders(cfg, stats, smoke, device, generator)
    steps_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    epochs = int(cfg["train"]["epochs"])

    model = build_unet(in_channels(cfg), cfg["model"]["encoder"], cfg["model"]["encoder_weights"]).to(device)
    lr = float(cfg["train"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=lr * float(cfg["train"]["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = CEDiceLoss(cfg["train"]["ce_weight"], cfg["train"]["dice_weight"])

    start_epoch, best_iou, best_epoch, wandb_id = 1, -math.inf, None, None
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run_info.json"
    if resume:
        ckpt = load_checkpoint(last_path, map_location=device)
        if _fixed_part(ckpt["config"]) != _fixed_part(cfg):
            raise ValueError("--resume: data/model/train settings differ from the checkpoint's config; "
                             "start a new run instead of changing an existing one")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"], generator)
        start_epoch, best_iou, best_epoch = ckpt["epoch"] + 1, ckpt["best_val_iou"], ckpt["best_epoch"]
        wandb_id = ckpt.get("wandb_id")
        rows = [r for r in _read_metrics(metrics_path) if int(r["epoch"]) <= ckpt["epoch"]]
        _write_metrics(metrics_path, rows)
        info = json.loads(info_path.read_text())
        info.setdefault("resumes", []).append({"from_epoch": ckpt["epoch"], "at": _now(), **git_info()})
        log(f"Resumed {run_dir} after epoch {ckpt['epoch']} (best valid IoU {best_iou:.4f} at epoch {best_epoch})")
    else:
        rows = []
        _write_metrics(metrics_path, rows)
        info = {"name": cfg["name"], "seed": seed, "smoke": smoke, "started": _now(), **git_info(),
                "device": str(device), "amp": amp, "torch": torch.__version__,
                "command": " ".join(sys.argv), "in_channels": in_channels(cfg), "modalities": list(mods),
                "n_train_chips": len(train_loader.dataset), "n_val_chips": len(val_loader.dataset),
                "steps_per_epoch": steps_per_epoch}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    wandb_run = _init_wandb(cfg, run_dir, wandb_id) if cfg["wandb"]["enabled"] else None
    log(f"run {run_dir} | device {device} | amp {amp} | in_channels {in_channels(cfg)} | "
        f"{len(train_loader.dataset)} train chips, {len(val_loader.dataset)} val chips | "
        f"{steps_per_epoch} steps/epoch | epochs {start_epoch}..{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device, amp,
                                     mods, max_batches)
        conf, val_loss = validate(model, val_loader, loss_fn, device, amp, mods, max_batches)
        m = conf.compute()
        improved = not math.isnan(m["iou"]) and m["iou"] > best_iou
        if improved:
            best_iou, best_epoch = m["iou"], epoch
        row = {"epoch": epoch, "lr": lr_now, "train_loss": train_loss, "val_loss": val_loss,
               "val_iou": m["iou"], "val_f1": m["f1"], "val_precision": m["precision"],
               "val_recall": m["recall"], "val_accuracy": m["accuracy"], "val_tp": m["tp"],
               "val_fp": m["fp"], "val_fn": m["fn"], "val_tn": m["tn"],
               "best_val_iou": best_iou if best_epoch else math.nan, "epoch_time_s": round(time.time() - t0, 2)}
        rows.append({k: _fmt(v) for k, v in row.items()})
        _write_metrics(metrics_path, rows)  # before the checkpoint: resume drops rows past last.pt

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "best_val_iou": best_iou, "best_epoch": best_epoch, "val_iou": m["iou"], "config": cfg,
                "stats": stats, "smoke": smoke, "rng": _rng_state(generator),
                "wandb_id": wandb_run.id if wandb_run else wandb_id, **git_info()}
        if improved:
            save_atomic(ckpt, best_path)
        save_atomic(ckpt, last_path)
        if wandb_run:
            wandb_run.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            wandb_run.summary.update({"best_val_iou": best_iou, "best_epoch": best_epoch})
        log(f"epoch {epoch:3d}/{epochs} | lr {lr_now:.2e} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val IoU {m['iou']:.4f} | best {best_iou:.4f} (epoch {best_epoch}){' *' if improved else ''} | "
            f"{row['epoch_time_s']:.0f}s")

    if wandb_run:
        wandb_run.finish()
    log(f"done: best valid water IoU {best_iou:.4f} at epoch {best_epoch} -> {best_path}")
    if smoke:
        log(SMOKE_BANNER)
    return run_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init_wandb(cfg: dict, run_dir: Path, wandb_id: str | None):
    import wandb  # only when enabled

    w = cfg["wandb"]
    run = wandb.init(project=w.get("project", "floodfusion"), entity=w.get("entity"), name=run_dir.name,
                     id=wandb_id, resume="allow", config=cfg, dir=str(run_dir))
    (run_dir / "wandb_id.txt").write_text(run.id + "\n")
    return run
