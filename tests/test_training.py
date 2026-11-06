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


def fusion_cfg(cfg: dict, fusion: str = "cross_attention") -> dict:
    """synth_cfg with FusionNet (resnet18 encoders, random init)."""
    c = copy.deepcopy(cfg)
    c["model"].update(arch="fusionnet", fusion=fusion, attention_heads=8)
    return resolve_config(c)


def robust_cfg(cfg: dict, clouds: bool = True, dropout: bool = True) -> dict:
    """cfg with synthetic clouds (p=1 so every chip gets some) and modality dropout on."""
    c = copy.deepcopy(cfg)
    c["augment"]["synthetic_clouds"].update(enabled=clouds, p=1.0)
    c["augment"]["modality_dropout"].update(enabled=dropout, p_drop_s1=0.3, p_drop_s2=0.3)
    # FusionNet runs an encoder only on samples that have its modality: with one such sample
    # and 32 px crops its stride-32 BatchNorm would see a single value (1x1). Real 256 crops: 8x8.
    c["data"]["crop_size"] = 64
    return resolve_config(c)


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


def test_fusion_config_matches_early_fusion_training_settings():
    """Fair comparison: fusion.yaml differs from early_fusion.yaml only in the model
    (and save_last_every, which does not affect training)."""
    fus = resolve_config(load_config(REPO / "configs" / "fusion.yaml"))
    ref = resolve_config(load_config(REPO / "configs" / "early_fusion.yaml"))
    assert fus["model"] == {"arch": "fusionnet", "encoder": "resnet34", "encoder_weights": "imagenet",
                            "fusion": "cross_attention", "attention_heads": 8}
    assert fus["seed"] == ref["seed"] and fus["data"] == ref["data"] and fus["wandb"] == ref["wandb"]
    a, b = dict(fus["train"]), dict(ref["train"])
    a.pop("save_last_every"), b.pop("save_last_every")
    assert a == b


@pytest.mark.parametrize("robust, base", [("fusion_robust", "fusion"), ("early_fusion_robust", "early_fusion"),
                                          ("s2_only_robust", "s2_only")])
def test_robust_configs_differ_only_in_augment(robust, base):
    r = resolve_config(load_config(REPO / "configs" / f"{robust}.yaml"))
    b = resolve_config(load_config(REPO / "configs" / f"{base}.yaml"))
    assert r["name"] == robust
    sc, md = r["augment"]["synthetic_clouds"], r["augment"]["modality_dropout"]
    assert sc == {"enabled": True, "p": 0.5, "coverage": [0.1, 1.0], "thin_p": 0.3}
    assert md["enabled"] is (base != "s2_only")
    if md["enabled"]:
        assert (md["p_drop_s1"], md["p_drop_s2"]) == (0.1, 0.2)
    assert not any(s["enabled"] for s in b["augment"].values())  # base configs: augmentation off
    for c in (r, b):
        c.pop("name"), c.pop("augment")
    assert r == b  # identical seed, data, model and train settings


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
    assert not set(best) & {"optimizer", "scheduler", "scaler", "rng"} and "model" in best
    with pytest.raises(FileExistsError, match="--resume"):
        run(synth_cfg, device=CPU, log=quiet)


class Disconnect(Exception):
    pass


def _stop_after_epoch(n):
    def log(msg, *_a, **_k):
        if msg.startswith(f"epoch {n:3d}/"):  # logged right after last.pt is saved
            raise Disconnect
    return log


def _with(cfg, out, **train):
    c = copy.deepcopy(cfg)
    c["output_dir"] = str(out)
    c["train"].update(train)
    return c


@pytest.mark.parametrize("every, epochs, stop_after, ckpt_epoch, arch", [
    (1, 2, 1, 1, "unet"),        # last.pt every epoch: nothing is lost
    (2, 4, 3, 2, "unet"),        # last.pt at 2 and 4: disconnect after 3 loses epoch 3, which is trained again
    (2, 3, 2, 2, "fusionnet"),   # FusionNet (cross-attention): same guarantee
    (1, 2, 1, 1, "unet+aug"),    # synthetic clouds + modality dropout: augmentation RNG is restored too
    (2, 3, 2, 2, "fusionnet+aug"),
])
def test_resume_reproduces_an_uninterrupted_run(synth_cfg, tmp_path, every, epochs, stop_after, ckpt_epoch, arch):
    arch, _, aug = arch.partition("+")
    if arch == "fusionnet":
        synth_cfg = fusion_cfg(synth_cfg)
    if aug:
        synth_cfg = robust_cfg(synth_cfg)
    a = run(_with(synth_cfg, tmp_path / "straight", save_last_every=every, epochs=epochs), device=CPU, log=quiet)

    interrupted = _with(synth_cfg, tmp_path / "interrupted", save_last_every=every, epochs=epochs)
    with pytest.raises(Disconnect):
        run(interrupted, device=CPU, log=_stop_after_epoch(stop_after))
    b_dir = Path(interrupted["output_dir"]) / "tiny_seed0"
    assert json.loads((b_dir / "run_info.json").read_text())["arch"] == arch
    assert load_checkpoint(b_dir / "last.pt")["epoch"] == ckpt_epoch
    assert len(_rows(b_dir)) == stop_after          # metrics.csv was ahead of last.pt
    b = run(interrupted, resume=True, device=CPU, log=quiet)

    wa, wb = load_checkpoint(a / "last.pt")["model"], load_checkpoint(b / "last.pt")["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "resumed run diverged from the uninterrupted one"
    ra, rb = _rows(a), _rows(b)
    assert [r["epoch"] for r in rb] == [str(e) for e in range(1, epochs + 1)]  # no duplicated epochs
    assert [r["val_iou"] for r in ra] == [r["val_iou"] for r in rb]
    (resume,) = json.loads((b / "run_info.json").read_text())["resumes"]
    assert resume["from_epoch"] == ckpt_epoch
    assert resume["epochs_retrained"] == list(range(ckpt_epoch + 1, stop_after + 1))
    assert len(resume["epochs_retrained"]) <= every   # at most save_last_every epochs lost


# The resume tests above all train on CPU, where map_location=device is a no-op, so the
# Colab crash (RNG ByteTensors loaded onto the GPU) only shows up on a real accelerator.
ACCEL = (torch.device("cuda") if torch.cuda.is_available()
         else torch.device("mps") if torch.backends.mps.is_available() else None)
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a CUDA or MPS device")


@needs_accel
def test_rng_state_loaded_onto_accelerator_can_be_restored(tmp_path):
    from floodfusion.training import _rng_state, _set_rng_state, save_atomic

    gen = torch.Generator().manual_seed(1)
    save_atomic({"rng": _rng_state(gen)}, tmp_path / "c.pt")
    expected = (torch.rand(3), torch.rand(3, generator=gen))
    state = load_checkpoint(tmp_path / "c.pt", map_location=ACCEL)["rng"]
    assert state["torch"].device.type == ACCEL.type   # what the old resume code passed in
    torch.manual_seed(123), gen.manual_seed(123)
    _set_rng_state(state, gen)
    got = (torch.rand(3), torch.rand(3, generator=gen))
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


@needs_accel
def test_resume_on_accelerator(synth_cfg, tmp_path):
    cfg = _with(synth_cfg, tmp_path, save_last_every=1, epochs=2)
    with pytest.raises(Disconnect):
        run(cfg, device=ACCEL, log=_stop_after_epoch(1))
    run_dir = run(cfg, resume=True, device=ACCEL, log=quiet)
    assert [r["epoch"] for r in _rows(run_dir)] == ["1", "2"]
    # Saved after the resumed epoch: optimizer state was on the training device, not left on CPU.
    ckpt = torch.load(run_dir / "last.pt", map_location=None, weights_only=False)
    assert ckpt["epoch"] == 2
    moments = [v for s in ckpt["optimizer"]["state"].values() for k, v in s.items() if k != "step"]
    assert moments and all(t.device.type == ACCEL.type for t in moments)


def test_last_pt_schedule(synth_cfg, tmp_path, monkeypatch):
    import floodfusion.training as training

    saved, real = [], training.save_atomic
    monkeypatch.setattr(training, "save_atomic", lambda obj, path: (saved.append((path.name, obj["epoch"])),
                                                                     real(obj, path)))
    run_dir = run(_with(synth_cfg, tmp_path / "a", save_last_every=2, epochs=5), device=CPU, log=quiet)
    assert [e for name, e in saved if name == "last.pt"] == [2, 4, 5]   # every 2 + always the final epoch
    improved, running = [], float("-inf")
    for r in _rows(run_dir):
        if r["val_iou"] and float(r["val_iou"]) > running:
            improved.append(int(r["epoch"]))
            running = float(r["val_iou"])
    assert [e for name, e in saved if name == "best.pt"] == improved     # best.pt on every improvement

    saved.clear()
    run(_with(synth_cfg, tmp_path / "b", epochs=2), device=CPU, log=quiet)  # default 5 > epochs
    assert [e for name, e in saved if name == "last.pt"] == [2]


def test_resume_keeps_best_pt_written_after_last_pt(synth_cfg, tmp_path):
    cfg = _with(synth_cfg, tmp_path, save_last_every=2, epochs=4)
    with pytest.raises(Disconnect):
        run(cfg, device=CPU, log=_stop_after_epoch(3))
    run_dir = Path(cfg["output_dir"]) / "tiny_seed0"
    # Make best.pt come from the lost epoch 3 with an unbeatable score.
    best = load_checkpoint(run_dir / "best.pt")
    best.update(epoch=3, val_iou=1.0)
    torch.save(best, run_dir / "best.pt")
    msgs = []
    run(cfg, resume=True, device=CPU, log=msgs.append)
    assert any("keeping it as the best to beat" in m for m in msgs)
    final_best, last = load_checkpoint(run_dir / "best.pt"), load_checkpoint(run_dir / "last.pt")
    assert (final_best["epoch"], final_best["val_iou"]) == (3, 1.0)        # not replaced by a worse model
    assert (last["best_epoch"], last["best_val_iou"]) == (3, 1.0)
    (resume,) = json.loads((run_dir / "run_info.json").read_text())["resumes"]
    assert resume["kept_newer_best"] == {"epoch": 3, "val_iou": 1.0}


def test_save_last_every_validation_and_change_on_resume(synth_cfg, tmp_path):
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="save_last_every"):
            resolve_config(synth_cfg, save_last_every=bad)
    assert resolve_config({**synth_cfg, "train": {k: v for k, v in synth_cfg["train"].items()
                                                  if k != "save_last_every"}})["train"]["save_last_every"] == 5
    cfg = _with(synth_cfg, tmp_path, save_last_every=2, epochs=4)
    with pytest.raises(Disconnect):
        run(cfg, device=CPU, log=_stop_after_epoch(2))
    run(resolve_config(cfg, save_last_every=1), resume=True, device=CPU, log=quiet)  # allowed: not training-relevant


def test_checkpoints_from_before_fusionnet_still_resume_and_evaluate(synth_cfg, tmp_path):
    """Baseline checkpoints written before model.arch existed (no arch key in their config)."""
    cfg = _with(synth_cfg, tmp_path, save_last_every=1, epochs=2)
    with pytest.raises(Disconnect):
        run(cfg, device=CPU, log=_stop_after_epoch(1))
    run_dir = Path(cfg["output_dir"]) / "tiny_seed0"
    for name in ("last.pt", "best.pt"):  # no model.arch and no augment section, as before
        ck = load_checkpoint(run_dir / name)
        del ck["config"]["model"]["arch"]
        del ck["config"]["augment"]
        torch.save(ck, run_dir / name)
    old = copy.deepcopy(cfg)
    del old["model"]["arch"]
    del old["augment"]
    run(resolve_config(old), resume=True, device=CPU, log=quiet)
    res = evaluate(run_dir / "best.pt", "valid", out_dir=tmp_path / "ev", device=CPU, log=quiet)
    assert res["n_chips"] == 2
    with pytest.raises(ValueError, match="differ"):  # switching to fusionnet on resume is refused
        run(fusion_cfg(cfg), resume=True, device=CPU, log=quiet)


def test_augment_section_absent_or_disabled_trains_identically(synth_cfg, tmp_path):
    old = copy.deepcopy(synth_cfg)
    del old["augment"]  # a config from before augmentation existed
    a = run(_with(old, tmp_path / "a"), device=CPU, log=quiet)
    b = run(_with(synth_cfg, tmp_path / "b"), device=CPU, log=quiet)
    wa, wb = load_checkpoint(a / "last.pt")["model"], load_checkpoint(b / "last.pt")["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert json.loads((b / "run_info.json").read_text())["augment"] == "none"


def test_s2_only_with_synthetic_clouds_trains(synth_cfg):
    c = copy.deepcopy(synth_cfg)
    c["data"]["modalities"] = ["s2"]
    run_dir = run(robust_cfg(resolve_config(c), dropout=False), device=CPU, log=quiet)
    assert "synthetic clouds" in json.loads((run_dir / "run_info.json").read_text())["augment"]


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


@pytest.mark.parametrize("fusion", ["cross_attention", "concat"])
def test_fusionnet_trains_and_evaluates(synth_cfg, tmp_path, fusion):
    run_dir = run(fusion_cfg(synth_cfg, fusion), device=CPU, log=quiet)
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["arch"] == "fusionnet" and info["fusion"] == fusion and info["n_params"] > 0
    best = load_checkpoint(run_dir / "best.pt")
    res = evaluate(run_dir / "best.pt", "valid", out_dir=tmp_path / "ev", device=CPU, log=quiet)
    assert res["metrics"]["iou"] == pytest.approx(best["val_iou"])  # same inputs as in validation


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
                    "--out", str(tmp_path / "cli"), "--no-wandb", "--seed", "3", "--save-last-every", "1"], check=True, capture_output=True)
    ckpt = tmp_path / "cli" / "tiny_seed3" / "best.pt"
    assert yaml.safe_load((ckpt.parent / "config.yaml").read_text())["seed"] == 3
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt),
                        "--split", "bolivia"], check=True, capture_output=True, text=True)
    assert "split=bolivia" in r.stdout and (ckpt.parent / "eval_bolivia" / "metrics.json").is_file()


def test_eval_table_and_training_curves(synth_cfg):
    from floodfusion.evaluation import format_eval_table
    from floodfusion.viz import plot_training_curves

    run_dir = run(synth_cfg, device=CPU, log=quiet)
    assert "(not evaluated)" in format_eval_table(run_dir)
    results = {s: evaluate(run_dir / "best.pt", s, device=CPU, log=quiet) for s in ("valid", "test", "bolivia")}
    table = format_eval_table(run_dir)
    assert "seed 0" in table and "cloud setting: real" in table and "(not evaluated)" not in table
    for s, r in results.items():
        line = next(l for l in table.splitlines() if l.startswith(s))
        assert f"{r['metrics']['iou']:6.4f}" in line
    png = plot_training_curves(run_dir / "metrics.csv", run_dir / "training_curves.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
