"""Training loop for the U-Net baselines (SAR-only, optical-only, early fusion) and FusionNet.

Rules enforced here (CLAUDE.md):
  * Training uses the train split, validation/model selection the valid split only.
    Configs naming "test" or "bolivia" for training or validation are refused.
  * Normalisation stats come from the stats file (train split only) and are stored in
    every checkpoint, so evaluation normalises exactly like training.
  * Seeds (Python, NumPy, PyTorch) come from the config.

Run folder: ``<output_dir>/<name>_seed<seed>`` (``..._smoke`` in smoke mode) with
config.yaml (resolved), run_info.json (git commit, seed, device, resumes), metrics.csv
(one row per epoch), last.pt (full training state, every ``train.save_last_every``
epochs and always on the final epoch) and best.pt (written whenever valid water IoU
improves; model weights, config and stats only).

save_last_every (default 5): last.pt is ~294 MB for ResNet34 + AdamW, and on a
Colab-mounted Drive every replaced version can go to Drive's Bin and count against the
quota, so it is not rewritten every epoch.

Models: ``model.arch`` "unet" (default; one input tensor, modalities stacked on the channel
axis) or "fusionnet" (separate s1 / s2 inputs plus an (N, 2) modality mask; see
floodfusion.models). ``forward_model`` is the only place that knows the difference. The
mask is ``batch["modality_mask"]`` if the batch has one, else all modalities present.

Robustness augmentation (config section ``augment``, training split only, off by default):
synthetic clouds on raw S2 and modality dropout (floodfusion.transforms). Validation
always uses the real, unmodified valid split.

Resume: last.pt holds model, optimizer, scheduler, AMP scaler, completed epoch, best
score and RNG states (Python, NumPy, torch, CUDA, DataLoader generator). It is written
to a temporary file and renamed, so a disconnect cannot leave a half-written file.
``resume=True`` restores all of it and continues with the next epoch; metrics.csv
rows after the checkpoint's epoch are dropped and those epochs are trained again, so
at most ``save_last_every`` epochs are lost. If best.pt is newer than last.pt (it was
written during the lost epochs), it was still selected on valid only and the earlier
best has been overwritten, so it is kept: its score becomes the best to beat, and the
resume entry in run_info.json records it.
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
from .models import ARCHS, FUSIONS, build_model, count_parameters
from .transforms import Compose, ModalityDropout, SyntheticClouds, TrainTransform
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
                 "model": ("arch", "encoder", "encoder_weights", "fusion", "attention_heads"),
                 "train": None,  # None = whole section
                 "augment": None}
# ...except these, which do not affect training and may change on resume.
_RESUME_FREE = {"train": ("save_last_every",)}
DEFAULT_SAVE_LAST_EVERY = 5
# Robustness augmentation defaults (all off). See configs/*_robust.yaml for the values used.
AUGMENT_DEFAULTS = {
    "synthetic_clouds": {"enabled": False, "p": 0.5, "coverage": [0.1, 1.0], "thin_p": 0.3},
    "modality_dropout": {"enabled": False, "p_drop_s1": 0.1, "p_drop_s2": 0.2},
}
REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_STATE_KEYS = ("optimizer", "scheduler", "scaler", "rng")

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
                   no_wandb: bool = False, seed: int | None = None,
                   save_last_every: int | None = None) -> dict:
    """Apply CLI overrides and defaults. The resolved config is saved in the run folder,
    so a run stays reproducible from its own files."""
    cfg = copy.deepcopy(cfg)
    if seed is not None:
        cfg["seed"] = int(seed)
    if root is not None:
        cfg["data"]["root"] = root
    if out is not None:
        cfg["output_dir"] = out
    cfg["data"].setdefault("train_split", TRAIN_SPLIT)
    cfg["data"].setdefault("val_split", VAL_SPLIT)
    cfg["data"].setdefault("s2_bands", None)
    cfg["train"].setdefault("min_lr_ratio", 0.01)
    if save_last_every is not None:
        cfg["train"]["save_last_every"] = save_last_every
    cfg["train"].setdefault("save_last_every", DEFAULT_SAVE_LAST_EVERY)
    k = cfg["train"]["save_last_every"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"train.save_last_every must be an integer >= 1, got {k!r}")
    _apply_model_defaults(cfg)
    _apply_augment_defaults(cfg)
    cfg["train"].setdefault("amp", True)
    cfg["train"].setdefault("ce_weight", 1.0)
    cfg["train"].setdefault("dice_weight", 1.0)
    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    if no_wandb:
        cfg["wandb"]["enabled"] = False
    return cfg


def _apply_model_defaults(cfg: dict) -> None:
    """model.arch defaults to "unet" (configs and checkpoints from before FusionNet have no
    arch); fusionnet gets fusion / attention_heads defaults. Validates the values."""
    m = cfg["model"]
    m.setdefault("arch", "unet")
    if m["arch"] not in ARCHS:
        raise ValueError(f"model.arch must be one of {ARCHS}, got {m['arch']!r}")
    if m["arch"] == "fusionnet":
        m.setdefault("fusion", "cross_attention")
        m.setdefault("attention_heads", 8)
        if m["fusion"] not in FUSIONS:
            raise ValueError(f"model.fusion must be one of {FUSIONS}, got {m['fusion']!r}")
        h = m["attention_heads"]
        if isinstance(h, bool) or not isinstance(h, int) or h < 1:
            raise ValueError(f"model.attention_heads must be an integer >= 1, got {h!r}")
        if tuple(cfg["data"]["modalities"]) != ("s1", "s2"):
            raise ValueError(f"fusionnet needs data.modalities [s1, s2], got {list(cfg['data']['modalities'])}")


def _apply_augment_defaults(cfg: dict) -> None:
    """Fill the ``augment`` section (configs from before it existed have none: all off)
    and validate it."""
    aug = cfg.setdefault("augment", {}) or {}
    cfg["augment"] = aug
    unknown = set(aug) - set(AUGMENT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown augment section(s) {sorted(unknown)}; known: {list(AUGMENT_DEFAULTS)}")
    for name, defaults in AUGMENT_DEFAULTS.items():
        sec = aug.setdefault(name, {})
        bad = set(sec) - set(defaults)
        if bad:
            raise ValueError(f"augment.{name}: unknown key(s) {sorted(bad)}; known: {list(defaults)}")
        for k, v in defaults.items():
            sec.setdefault(k, copy.deepcopy(v))
        if not isinstance(sec["enabled"], bool):
            raise ValueError(f"augment.{name}.enabled must be true or false, got {sec['enabled']!r}")
    mods = tuple(cfg["data"]["modalities"])
    sc, md = aug["synthetic_clouds"], aug["modality_dropout"]
    SyntheticClouds(sc["p"], sc["coverage"], sc["thin_p"])  # validates the values
    ModalityDropout(md["p_drop_s1"], md["p_drop_s2"])
    if sc["enabled"] and "s2" not in mods:
        raise ValueError("augment.synthetic_clouds needs s2 in data.modalities")
    if md["enabled"] and mods != ("s1", "s2"):
        raise ValueError(f"augment.modality_dropout needs data.modalities [s1, s2], got {list(mods)}")


def augment_description(cfg: dict) -> str:
    aug = cfg.get("augment") or {}
    sc, md = aug.get("synthetic_clouds", {}), aug.get("modality_dropout", {})
    parts = []
    if sc.get("enabled"):
        parts.append(f"synthetic clouds p={sc['p']} coverage={sc['coverage']} thin_p={sc['thin_p']}")
    if md.get("enabled"):
        parts.append(f"modality dropout p_drop_s1={md['p_drop_s1']} p_drop_s2={md['p_drop_s2']}")
    return "; ".join(parts) or "none"


def model_description(cfg: dict) -> str:
    m = cfg["model"]
    if m.get("arch", "unet") == "fusionnet":
        return f"fusionnet ({m['fusion']}, {m['encoder']} x2)"
    return f"unet ({m['encoder']}, in_channels {in_channels(cfg)})"


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


def forward_model(model, batch: dict, cfg: dict, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
    """Logits (N, 2, H, W) for a batch. U-Net: one stacked tensor (exactly as for the
    baselines). FusionNet: s1, s2 and the modality mask as separate inputs."""
    if cfg["model"].get("arch", "unet") == "unet":
        return model(stack_inputs(batch, check_modalities(cfg)).to(device, non_blocking=non_blocking))
    s1 = batch["s1"].to(device, non_blocking=non_blocking)
    s2 = batch["s2"].to(device, non_blocking=non_blocking)
    mask = batch.get("modality_mask")
    mask = torch.ones(s1.shape[0], 2, dtype=torch.bool) if mask is None else mask
    return model(s1, s2, mask.to(device))


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
    aug = copy.deepcopy(cfg)
    _apply_augment_defaults(aug)
    sc, md = aug["augment"]["synthetic_clouds"], aug["augment"]["modality_dropout"]
    clouds = SyntheticClouds(sc["p"], sc["coverage"], sc["thin_p"]) if sc["enabled"] else None
    transform = TrainTransform(d["crop_size"])
    if md["enabled"]:
        transform = Compose([transform, ModalityDropout(md["p_drop_s1"], md["p_drop_s2"])])
    train_ds = Sen1Floods11(split=d["train_split"], transform=transform, s2_augment=clouds, **common)
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


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, amp, cfg,
                    max_batches: int | None = None) -> float:
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = forward_model(model, batch, cfg, device, non_blocking=True)
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
def validate(model, loader, loss_fn, device, amp, cfg, max_batches: int | None = None):
    model.eval()
    conf = WaterConfusion()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        y = batch["label"].to(device)
        with _autocast(device, amp):
            logits = forward_model(model, batch, cfg, device)
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
    # torch/cuda/generator states must be CPU ByteTensors; .cpu() also accepts states that a
    # non-CPU map_location moved to the GPU (no-op for CPU tensors).
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    # weights_only=False: our own checkpoints also hold config, stats and RNG states.
    return torch.load(path, map_location=map_location, weights_only=False)


def _fixed_part(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    _apply_model_defaults(cfg)  # checkpoints from before FusionNet have no model.arch
    _apply_augment_defaults(cfg)  # ... and none from before augmentation have an augment section
    out = {s: (dict(cfg[s]) if keys is None else {k: cfg[s].get(k) for k in keys}) for s, keys in _RESUME_FIXED.items()}
    for s, keys in _RESUME_FREE.items():
        for k in keys:
            out[s].pop(k, None)
    return out


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

    model = build_model(cfg).to(device)
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
        # Load on CPU: RNG states must stay CPU ByteTensors. model.load_state_dict copies into
        # the parameters (already on `device`) and optimizer.load_state_dict moves its state to
        # each parameter's device, so nothing else needs moving.
        ckpt = load_checkpoint(last_path, map_location="cpu")
        if _fixed_part(ckpt["config"]) != _fixed_part(cfg):
            raise ValueError("--resume: data/model/train settings differ from the checkpoint's config; "
                             "start a new run instead of changing an existing one")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"], generator)
        start_epoch, best_iou, best_epoch = ckpt["epoch"] + 1, ckpt["best_val_iou"], ckpt["best_epoch"]
        resume_note = {"from_epoch": ckpt["epoch"], "at": _now(), **git_info()}
        if best_path.is_file():
            b = load_checkpoint(best_path)
            if b["epoch"] > ckpt["epoch"] and b["val_iou"] > best_iou:
                best_iou, best_epoch = b["val_iou"], b["epoch"]
                resume_note["kept_newer_best"] = {"epoch": b["epoch"], "val_iou": b["val_iou"]}
                log(f"best.pt is from epoch {b['epoch']} (after last.pt): keeping it as the best to beat "
                    f"(valid IoU {b['val_iou']:.4f}); epochs {start_epoch}.. are trained again")
        wandb_id = ckpt.get("wandb_id")
        all_rows = _read_metrics(metrics_path)
        rows = [r for r in all_rows if int(r["epoch"]) <= ckpt["epoch"]]
        resume_note["epochs_retrained"] = [int(r["epoch"]) for r in all_rows if int(r["epoch"]) > ckpt["epoch"]]
        _write_metrics(metrics_path, rows)
        info = json.loads(info_path.read_text())
        info.setdefault("resumes", []).append(resume_note)
        log(f"Resumed {run_dir} after epoch {ckpt['epoch']} (best valid IoU {best_iou:.4f} at epoch {best_epoch})")
    else:
        rows = []
        _write_metrics(metrics_path, rows)
        info = {"name": cfg["name"], "seed": seed, "smoke": smoke, "started": _now(), **git_info(),
                "device": str(device), "amp": amp, "torch": torch.__version__,
                "command": " ".join(sys.argv), "model": model_description(cfg), "arch": cfg["model"]["arch"],
                "fusion": cfg["model"].get("fusion"), "n_params": count_parameters(model),
                "augment": augment_description(cfg),
                "in_channels": in_channels(cfg), "modalities": list(mods),
                "n_train_chips": len(train_loader.dataset), "n_val_chips": len(val_loader.dataset),
                "steps_per_epoch": steps_per_epoch}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    wandb_run = _init_wandb(cfg, run_dir, wandb_id) if cfg["wandb"]["enabled"] else None
    log(f"run {run_dir} | device {device} | amp {amp} | {model_description(cfg)} | augment: {augment_description(cfg)} | "
        f"{len(train_loader.dataset)} train chips, {len(val_loader.dataset)} val chips | "
        f"{steps_per_epoch} steps/epoch | epochs {start_epoch}..{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device, amp,
                                     cfg, max_batches)
        conf, val_loss = validate(model, val_loader, loss_fn, device, amp, cfg, max_batches)
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
        if improved:  # best.pt is for evaluation only: no optimizer/scheduler/scaler/RNG (~1/3 the size)
            save_atomic({k: v for k, v in ckpt.items() if k not in _TRAIN_STATE_KEYS}, best_path)
        if epoch % int(cfg["train"]["save_last_every"]) == 0 or epoch == epochs:
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
