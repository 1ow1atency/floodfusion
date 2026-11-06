"""Tests for floodfusion.training and floodfusion.evaluation on tiny synthetic data (CPU)."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from floodfusion.evaluation import evaluate
from floodfusion.training import (
    check_splits, in_channels, load_checkpoint, load_config, resolve_config, run, stack_inputs,
)

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def quiet(*_a, **_k):
    pass


def _rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- configs and guards


@pytest.mark.parametrize("name", ["s1_only", "s2_only", "early_fusion"])
def test_repo_configs_load_and_differ_only_in_modalities(name):
    cfg = resolve_config(load_config(REPO / "configs" / f"{name}.yaml"))
    check_splits(cfg)
    assert in_channels(cfg) == {"s1_only": 2, "s2_only": 13, "early_fusion": 15}[name]
    ref = resolve_config(load_config(REPO / "configs" / "early_fusion.yaml"))
    a, b = copy.deepcopy(cfg), copy.deepcopy(ref)
    for c in (a, b):
        c.pop("name"); c["data"].pop("modalities")
    assert a == b


@pytest.mark.parametrize("key, split", [("train_split", "bolivia"), ("train_split", "test"),
                                        ("val_split", "bolivia"), ("val_split", "test")])
def test_refuses_eval_only_splits(synth_cfg, key, split):
    synth_cfg["data"][key] = split
    with pytest.raises(ValueError, match="evaluation-only"):
        check_splits(synth_cfg)
    with pytest.raises(ValueError, match="evaluation-only"):
        run(synth_cfg, device=CPU, log=quiet)


def test_refuses_other_split_names_and_bad_modalities(synth_cfg):
    cfg = copy.deepcopy(synth_cfg)
    cfg["data"]["val_split"] = "train"
    with pytest.raises(ValueError, match="val_split"):
        check_splits(cfg)
    cfg = copy.deepcopy(synth_cfg)
    cfg["data"]["modalities"] = ["s2", "s1"]
    with pytest.raises(ValueError, match="S1 first"):
        in_channels(cfg)


def test_stack_inputs_puts_s1_first():
    batch = {"s1": torch.zeros(1, 2, 4, 4), "s2": torch.ones(1, 13, 4, 4)}
    x = stack_inputs(batch, ("s1", "s2"))
    assert x.shape == (1, 15, 4, 4) and x[:, :2].sum() == 0 and torch.all(x[:, 2:] == 1)


def test_stats_must_come_from_train(synth_cfg, tmp_path):
    stats = json.loads(Path(synth_cfg["data"]["stats"]).read_text())
    stats["meta"]["split"] = "bolivia"
    bad = tmp_path / "bad_stats.json"
    bad.write_text(json.dumps(stats))
    synth_cfg["data"]["stats"] = str(bad)
    with pytest.raises(ValueError, match="train split"):
        run(synth_cfg, device=CPU, log=quiet)


# ----------------------------------------------------------------------------- run folder, resume


def test_run_writes_folder_contents(synth_cfg):
    run_dir = run(synth_cfg, device=CPU, log=quiet)
    assert run_dir.name == "tiny_seed0"
    for f in ("config.yaml", "run_info.json", "metrics.csv", "last.pt", "best.pt"):
        assert (run_dir / f).is_file(), f
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["seed"] == 0 and len(info["git_commit"]) >= 7 and info["smoke"] is False
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == synth_cfg
    rows = _rows(run_dir)
    assert [r["epoch"] for r in rows] == ["1", "2"]
    ckpt = load_checkpoint(run_dir / "last.pt")
    assert ckpt["epoch"] == 2 and ckpt["stats"]["meta"]["split"] == "train"
    assert set(ckpt) >= {"model", "optimizer", "scheduler", "scaler", "best_val_iou", "rng", "config"}
    best = load_checkpoint(run_dir / "best.pt")
    assert best["val_iou"] == pytest.approx(max(float(r["val_iou"]) for r in rows))
    with pytest.raises(FileExistsError, match="--resume"):
        run(synth_cfg, device=CPU, log=quiet)


class Disconnect(Exception):
    pass


def _stop_after_epoch(n):
    def log(msg, *_a, **_k):
        if msg.startswith(f"epoch {n:3d}/"):  # logged right after last.pt is saved
            raise Disconnect
    return log


def test_resume_reproduces_an_uninterrupted_run(synth_cfg, tmp_path):
    straight = copy.deepcopy(synth_cfg)
    straight["output_dir"] = str(tmp_path / "straight")
    a = run(straight, device=CPU, log=quiet)

    interrupted = copy.deepcopy(synth_cfg)
    interrupted["output_dir"] = str(tmp_path / "interrupted")
    with pytest.raises(Disconnect):
        run(interrupted, device=CPU, log=_stop_after_epoch(1))
    # A metrics row written after the checkpoint (e.g. by a later, unsaved epoch) is dropped.
    b_dir = Path(interrupted["output_dir"]) / "tiny_seed0"
    with open(b_dir / "metrics.csv", "a") as f:
        f.write("99" + "," * 14 + "\n")
    b = run(interrupted, resume=True, device=CPU, log=quiet)

    wa, wb = load_checkpoint(a / "last.pt")["model"], load_checkpoint(b / "last.pt")["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "resumed run diverged from the uninterrupted one"
    ra, rb = _rows(a), _rows(b)
    assert [r["epoch"] for r in rb] == ["1", "2"]
    assert [r["val_iou"] for r in ra] == [r["val_iou"] for r in rb]
    assert len(json.loads((b / "run_info.json").read_text())["resumes"]) == 1


def test_resume_errors(synth_cfg):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        run(synth_cfg, resume=True, device=CPU, log=quiet)
    run(synth_cfg, device=CPU, log=quiet)
    changed = copy.deepcopy(synth_cfg)
    changed["train"]["lr"] = 0.5
    with pytest.raises(ValueError, match="differ"):
        run(changed, resume=True, device=CPU, log=quiet)


def test_smoke_mode(synth_cfg, synth_root):
    # Smoke validates on 2 TRAIN chips even if the valid split is missing locally.
    (synth_root / "splits" / "flood_valid_data.csv").write_text("Missing_1_S1Hand.tif,x\r\n")
    msgs = []
    run_dir = run(synth_cfg, smoke=True, device=CPU, log=msgs.append)
    assert run_dir.name == "tiny_seed0_smoke"
    assert sum("SMOKE TEST" in m for m in msgs) == 2
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["smoke"] is True and info["n_val_chips"] == 2 and info["steps_per_epoch"] == 2
    assert [r["epoch"] for r in _rows(run_dir)] == ["1"]  # a single epoch
    assert load_checkpoint(run_dir / "last.pt")["smoke"] is True


# ----------------------------------------------------------------------------- evaluation


def test_evaluate_writes_json_and_per_chip_csv(synth_cfg, tmp_path):
    run_dir = run(synth_cfg, device=CPU, log=quiet)
    res = evaluate(run_dir / "best.pt", "test", out_dir=tmp_path / "ev", device=CPU, log=quiet)
    data = json.loads((tmp_path / "ev" / "metrics.json").read_text())
    assert data["split"] == "test" and data["n_chips"] == 2 and data["seed"] == 0
    assert "no synthetic clouds" in data["cloud_setting"] and data == json.loads(json.dumps(res))
    with open(tmp_path / "ev" / "per_chip.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chip_id"] for r in rows] == ["Te_0", "Te_1"]
    assert list(rows[0]) == ["chip_id", "tp", "fp", "fn", "tn", "n_valid", "iou"]
    for k in ("tp", "fp", "fn", "tn"):
        assert sum(int(r[k]) for r in rows) == data["metrics"][k]   # per-chip counts add up
    with pytest.raises(ValueError, match="split must be"):
        evaluate(run_dir / "best.pt", "train", device=CPU, log=quiet)


def test_evaluate_matches_validation_metrics(synth_cfg):
    run_dir = run(synth_cfg, device=CPU, log=quiet)
    best = load_checkpoint(run_dir / "best.pt")
    res = evaluate(run_dir / "best.pt", "valid", device=CPU, log=quiet)
    assert res["metrics"]["iou"] == pytest.approx(best["val_iou"])  # same chips, same model
    assert (run_dir / "eval_valid" / "per_chip.csv").is_file()


def test_scripts_run_end_to_end(synth_cfg, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(synth_cfg))
    subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(cfg_path),
                    "--out", str(tmp_path / "cli"), "--no-wandb"], check=True, capture_output=True)
    ckpt = tmp_path / "cli" / "tiny_seed0" / "best.pt"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt),
                        "--split", "bolivia"], check=True, capture_output=True, text=True)
    assert "split=bolivia" in r.stdout and (ckpt.parent / "eval_bolivia" / "metrics.json").is_file()
