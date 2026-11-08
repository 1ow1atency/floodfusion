"""Robustness evaluation (floodfusion.robustness) on tiny synthetic chips with dummy models
(randomly initialised, saved as best.pt and evaluated like real runs)."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import floodfusion.robustness as rb
from floodfusion.clouds_synth import cloudify
from floodfusion.data import Sen1Floods11
from floodfusion.evaluation import evaluate
from floodfusion.models import build_model
from floodfusion.training import resolve_config, stack_inputs

CPU = torch.device("cpu")
REPO = Path(__file__).resolve().parents[1]


def quiet(*_a, **_k):
    pass


def _dummy_run(base: dict, out: Path, name: str, modalities, arch="unet", robust=False) -> Path:
    cfg = copy.deepcopy(base)
    cfg["name"], cfg["data"]["modalities"] = name, list(modalities)
    cfg["model"]["arch"] = arch
    if robust:
        cfg["augment"]["synthetic_clouds"]["enabled"] = True
    cfg = resolve_config(cfg)
    torch.manual_seed(len(name))
    model = build_model(cfg, encoder_weights=None)
    run_dir = out / f"{name}_seed0"
    run_dir.mkdir(parents=True)
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    torch.save({"model": model.state_dict(), "epoch": 3, "best_val_iou": 0.5, "val_iou": 0.5, "config": cfg,
                "stats": stats, "smoke": False}, run_dir / "best.pt")
    for split in rb.SPLITS:
        evaluate(run_dir / "best.pt", split, device=CPU, log=quiet)
    return run_dir


@pytest.fixture
def runs(synth_cfg, synth_root, tmp_path):
    out = tmp_path / "runs_eval"
    dirs = {"s1_only": _dummy_run(synth_cfg, out, "s1_only", ["s1"]),
            "s2_only": _dummy_run(synth_cfg, out, "s2_only", ["s2"]),
            "early_fusion": _dummy_run(synth_cfg, out, "early_fusion", ["s1", "s2"]),
            "fusion_robust": _dummy_run(synth_cfg, out, "fusion_robust", ["s1", "s2"], arch="fusionnet", robust=True)}
    rows = []
    for split in ("train", "valid", "test", "bolivia"):
        for i, cid in enumerate(Sen1Floods11(synth_root, split, modalities=("s1",)).ids):
            frac = "" if (split, i) == ("test", 1) else f"{0.05 + 0.3 * i:.6f}"  # test chip 1: no valid S2
            rows.append({"chip_id": cid, "split": split, "cloud_fraction": frac, "valid_s2_fraction": "1.0"})
    cloud_csv = tmp_path / "cloud_cover.csv"
    with open(cloud_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return {"dirs": dirs, "root": synth_root, "cloud_csv": cloud_csv, "tmp": tmp_path}


def _run(runs, out, **kw):
    kw.setdefault("cache_dir", None)
    return rb.run_robustness(list(runs["dirs"].values()), runs["root"], out, cloud_csv=runs["cloud_csv"],
                             device=CPU, log=quiet, **kw)


# ----------------------------------------------------------------------------- end to end

def test_full_evaluation_outputs_and_markings(runs):
    res = _run(runs, runs["tmp"] / "res")
    out = runs["tmp"] / "res"
    for f in ("robustness.csv", "summary.md", "iou_vs_coverage_test.png", "iou_vs_coverage_bolivia.png",
              "real_cloud_bins.png", "qualitative.png"):
        assert (out / f).is_file(), f
    rows = rb.read_csv(out / "robustness.csv")
    assert list(rows[0]) == list(rb.CSV_COLUMNS)
    assert [m.label for m in res["models"]] == ["s1_only", "s2_only", "early_fusion", "fusion_robust"]  # fixed order

    def get(model, split, setting, cov="", seed=""):
        return [r for r in rows if r["model"] == model and r["split"] == split and r["setting"] == setting
                and r["coverage"] == cov and r["cloud_seed"] == str(seed)]

    for split in rb.SPLITS:
        clean = get("s1_only", split, "synthetic_clouds", "0.0")[0]
        for cov in ("0.2", "1.0"):
            for seed in rb.CLOUD_SEEDS:  # s1_only is unaffected by clouds: its clean result, marked
                (r,) = get("s1_only", split, "synthetic_clouds", cov, seed)
                assert r["iou"] == clean["iou"] and r["note"] == rb.NOTE_NO_S2
        assert get("s2_only", split, "s2_removed")[0]["note"] == rb.NOTE_NA_S2 and get("s2_only", split, "s2_removed")[0]["iou"] == ""
        assert get("s1_only", split, "s1_removed")[0]["note"] == rb.NOTE_NA_S1
        assert get("s1_only", split, "s2_removed")[0]["note"] == rb.NOTE_NO_S2
        assert get("s2_only", split, "s1_removed")[0]["note"] == rb.NOTE_NO_S1
        for m in ("early_fusion", "fusion_robust"):
            assert get(m, split, "s2_removed")[0]["note"] == "" and get(m, split, "s1_removed")[0]["note"] == ""
        n_synth = len([r for r in rows if r["split"] == split and r["setting"] == "synthetic_clouds"])
        assert n_synth == 4 * (1 + 5 * len(rb.CLOUD_SEEDS))
    md = (out / "summary.md").read_text()
    assert "Final reporting only" in md and "audit" in md and "n/a" in md and "†" in md and "⚠" in md


def test_cloud_cache_gives_identical_results(runs):
    cache = runs["tmp"] / "cache"
    first = _run(runs, runs["tmp"] / "a", cache_dir=cache, figures=False)
    second = _run(runs, runs["tmp"] / "b", cache_dir=cache, figures=False)
    nocache = _run(runs, runs["tmp"] / "c", figures=False)
    n_clouds = 5 * len(rb.CLOUD_SEEDS) * (2 + 1)  # coverages x seeds x (2 test + 1 bolivia chips)
    assert first["clouds_generated"] == n_clouds and len(list(cache.glob("*.npz"))) == n_clouds
    assert second["clouds_generated"] == 0  # every cloud came from the cache
    assert nocache["clouds_generated"] == n_clouds  # each cloud generated once, even without a cache
    assert first["rows"] == second["rows"] == nocache["rows"]


def test_cache_layer_rebuilds_cloudify_exactly(synth_root, tmp_path):
    from floodfusion.chips import load_raw

    cid = Sen1Floods11(synth_root, "test", modalities=("s1",)).ids[0]
    s2 = load_raw(synth_root, cid, "s2")
    for cache in (rb.CloudCache(), rb.CloudCache(tmp_path / "c"), rb.CloudCache(tmp_path / "c")):  # fresh, write, read
        got = cache.apply(s2, cid, 0.6, 1)
        want = cloudify(s2, 0.6, rb.eval_cloud_seed(cid, 0.6, 1), thin=False)
        assert np.array_equal(got, want, equal_nan=True)
    assert cache.apply(s2, cid, 0.0, 1) is s2  # coverage 0: untouched
    seeds = {rb.eval_cloud_seed(c, cov, s) for c in ("A", "B") for cov in (0.2, 0.4) for s in (0, 1)}
    assert len(seeds) == 8  # distinct per chip, coverage and cloud seed


# ----------------------------------------------------------------------------- identical inputs, missing modality

def _spy(monkeypatch):
    calls = []
    real = rb.forward_model

    def spy(model, batch, cfg, device, non_blocking=False):
        calls.append({"batch": id(batch), "model": cfg["name"], "ids": tuple(batch["id"]),
                      "s1": batch.get("s1").clone() if "s1" in batch else None, "s2": batch["s2"].clone(),
                      "mask": batch.get("modality_mask"), "x": stack_inputs(batch, tuple(cfg["data"]["modalities"]))})
        return real(model, batch, cfg, device, non_blocking)

    monkeypatch.setattr(rb, "forward_model", spy)
    return calls


def test_identical_clouds_for_every_model(runs, monkeypatch):
    calls = _spy(monkeypatch)
    cache = rb.CloudCache()
    a = rb.load_models([runs["dirs"]["s2_only"], runs["dirs"]["fusion_robust"]], CPU, log=quiet)
    rb.evaluate_split(a, "test", runs["root"], cache=cache, device=CPU, log=quiet)
    first, calls[:] = list(calls), []
    b = rb.load_models([runs["dirs"]["early_fusion"]], CPU, log=quiet)
    rb.evaluate_split(b, "test", runs["root"], cache=cache, device=CPU, log=quiet)
    second = list(calls)
    by_batch = {}
    for c in first:  # within one evaluation: every model gets the very same batch
        by_batch.setdefault(c["batch"], []).append(c)
    for group in by_batch.values():
        assert all(torch.equal(g["s2"], group[0]["s2"]) for g in group)
    # across evaluations with other models: the same S2 inputs, batch by batch (clean, 15 cloud passes)
    s2_a = [c["s2"] for c in first if c["model"] == "fusion_robust"][:16]
    s2_b = [c["s2"] for c in second if c["model"] == "early_fusion"][:16]
    assert len(s2_a) == 16 and all(torch.equal(x, y) for x, y in zip(s2_a, s2_b))


def test_missing_modality_inputs_per_model_type(runs, monkeypatch):
    assert rb.status(("s1",), "synthetic_clouds") == rb.NOTE_NO_S2
    assert rb.status(("s2",), "s2_removed") == rb.NOTE_NA_S2 and rb.status(("s2",), "s1_removed") == rb.NOTE_NO_S1
    assert rb.status(("s1",), "s1_removed") == rb.NOTE_NA_S1 and rb.status(("s1",), "s2_removed") == rb.NOTE_NO_S2
    assert rb.status(("s1", "s2"), "s2_removed") == rb.status(("s1", "s2"), "s1_removed") == "eval"
    calls = _spy(monkeypatch)
    models = rb.load_models(list(runs["dirs"].values()), CPU, log=quiet)
    confs, _, _ = rb.evaluate_split(models, "test", runs["root"], cache=rb.CloudCache(), device=CPU, log=quiet)
    assert ("s2_only", "s2_removed", math.nan, None) not in confs
    removed = [c for c in calls if c["mask"] is not None]
    assert {c["model"] for c in removed} == {"early_fusion", "fusion_robust"}  # only these get S2/S1 removed
    for c in removed:
        gone = 1 if not c["mask"][0, 1] else 0  # 1 = S2 removed, 0 = S1 removed
        assert c["mask"][:, gone].eq(False).all() and c["mask"][:, 1 - gone].all()
        x = c["x"]  # early fusion input: S1 channels 0-1, S2 channels 2-14, after normalisation
        assert not (x[:, 2:] if gone else x[:, :2]).any() and (x[:, :2] if gone else x[:, 2:]).any()


# ----------------------------------------------------------------------------- sanity check, bins, selection

def test_sanity_check_fails_loudly(runs):
    path = runs["dirs"]["early_fusion"] / "eval_test" / "metrics.json"
    good = path.read_text()
    d = json.loads(good)
    d["metrics"]["tp"] += 1
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match=r"(?s)SANITY CHECK FAILED.*early_fusion.*tp"):
        _run(runs, runs["tmp"] / "x", figures=False)
    d = json.loads(good)
    d["checkpoint_epoch"] = 99
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="epoch 99"):
        _run(runs, runs["tmp"] / "y", figures=False)
    path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        _run(runs, runs["tmp"] / "z", figures=False)


def test_real_cloud_binning():
    def counts(tp, fp, fn, tn):
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    per_chip = {"a": counts(8, 2, 0, 90), "b": counts(2, 0, 2, 96), "c": counts(5, 5, 5, 85),
                "d": counts(1, 0, 0, 99), "e": counts(9, 9, 9, 9)}
    cloud_rows = [{"chip_id": "a", "split": "test", "cloud_fraction": 0.0}, {"chip_id": "b", "split": "test", "cloud_fraction": 0.05},
                  {"chip_id": "c", "split": "test", "cloud_fraction": 0.45}, {"chip_id": "d", "split": "test", "cloud_fraction": math.nan},
                  {"chip_id": "e", "split": "bolivia", "cloud_fraction": 0.9}]  # other split: ignored
    out = {b["bin"]: b for b in rb.real_cloud_bins(per_chip, cloud_rows, "test")}
    assert out["0-10%"]["n_chips"] == 2 and out["0-10%"]["iou"] == pytest.approx(10 / 14)  # pooled, not averaged
    assert out["30-60%"]["n_chips"] == 1 and out["30-60%"]["iou"] == pytest.approx(5 / 15)
    assert out["10-30%"]["n_chips"] == 0 and out["10-30%"]["note"] == "no chips"
    assert all(out[b]["unreliable"] for b in ("0-10%", "30-60%")) and out["0-10%"]["note"] == rb.NOTE_UNRELIABLE
    assert out["no cloud fraction"]["ids"] == ["d", "e"]  # NaN fraction, and no row for this split
    many = {f"x{i}": counts(1, 0, 0, 1) for i in range(rb.MIN_BIN_CHIPS)}
    rows = [{"chip_id": c, "split": "test", "cloud_fraction": 0.7} for c in many]
    big = {b["bin"]: b for b in rb.real_cloud_bins(many, rows, "test")}
    assert big["60-100%"]["n_chips"] == rb.MIN_BIN_CHIPS and not big["60-100%"]["unreliable"]


def test_qualitative_chip_rule(runs):
    ids = Sen1Floods11(runs["root"], "test", modalities=("s1",)).ids
    from floodfusion.clouds import read_cloud_csv

    rows = read_cloud_csv(runs["cloud_csv"])
    picked = rb.pick_qual_chips(runs["root"], "test", rows, ids, n=4)
    assert picked == [ids[0]]  # ids[1] has no cloud fraction; the rule is deterministic
    assert rb.pick_qual_chips(runs["root"], "test", rows, ids, n=4) == picked


def test_refuses_smoke_checkpoints_and_other_splits(runs):
    d = runs["dirs"]["s1_only"]
    ck = torch.load(d / "best.pt", weights_only=False)
    ck["smoke"] = True
    torch.save(ck, d / "best.pt")
    with pytest.raises(ValueError, match="smoke"):
        rb.load_models([d], CPU, log=quiet)
    assert rb.load_models([d], CPU, allow_smoke=True, log=quiet)[0].smoke
    with pytest.raises(ValueError, match="test.*bolivia"):
        rb.run_robustness([d], runs["root"], runs["tmp"] / "v", splits=["valid"], device=CPU, log=quiet)


def test_find_runs(tmp_path):
    for name in ("b_seed0", "a_seed0", "no_ckpt"):
        (tmp_path / name).mkdir()
    for name in ("b_seed0", "a_seed0"):
        (tmp_path / name / "best.pt").write_bytes(b"")
    assert [p.name for p in rb.find_runs(tmp_path)] == ["a_seed0", "b_seed0"]


def test_model_styles():
    from floodfusion.viz import CATEGORICAL, model_styles

    st = model_styles([("s1_only", "s1_only", False), ("fusion", "fusion", False), ("fusion_robust", "fusion", True),
                       ("s2_only", "s2_only", False)])
    assert st["fusion"]["colour"] == st["fusion_robust"]["colour"] == CATEGORICAL[3]  # colour follows the family
    assert st["fusion"]["dashed"] and not st["fusion_robust"]["dashed"]
    assert not st["s1_only"]["dashed"] and not st["s2_only"]["dashed"]  # no robust sibling: solid
