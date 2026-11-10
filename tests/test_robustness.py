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


def _dummy_run(base: dict, out: Path, name: str, modalities, arch="unet", robust=False, seed=0) -> Path:
    cfg = copy.deepcopy(base)
    cfg["name"], cfg["data"]["modalities"], cfg["seed"] = name, list(modalities), seed
    cfg["model"]["arch"] = arch
    if robust:
        cfg["augment"]["synthetic_clouds"]["enabled"] = True
    cfg = resolve_config(cfg)
    torch.manual_seed(len(name) + 100 * seed)
    model = build_model(cfg, encoder_weights=None)
    run_dir = out / f"{name}_seed{seed}"
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
        clean = get("s1_only", split, "thick_clouds", "0.0")[0]
        for setting in rb.CLOUD_SETTINGS:
            for cov in ("0.2", "1.0"):
                for seed in rb.CLOUD_SEEDS:  # s1_only is unaffected by clouds: its clean result, marked
                    (r,) = get("s1_only", split, setting, cov, seed)
                    assert r["iou"] == clean["iou"] and r["note"] == rb.NOTE_NO_S2
        for m in ("s2_only", "early_fusion", "fusion_robust"):  # opaque at 0%: evaluated, equal to clean
            (z,) = get(m, split, "opaque_clouds", "0.0", rb.CLOUD_SEEDS[0])
            (c,) = get(m, split, "thick_clouds", "0.0")
            assert z["note"] == rb.NOTE_ZERO_OPAQUE and (z["iou"], z["precision"], z["recall"]) == (
                c["iou"], c["precision"], c["recall"])
        assert get("s2_only", split, "s2_removed")[0]["note"] == rb.NOTE_NA_S2 and get("s2_only", split, "s2_removed")[0]["iou"] == ""
        assert get("s1_only", split, "s1_removed")[0]["note"] == rb.NOTE_NA_S1
        assert get("s1_only", split, "s2_removed")[0]["note"] == rb.NOTE_NO_S2
        assert get("s2_only", split, "s1_removed")[0]["note"] == rb.NOTE_NO_S1
        for m in ("early_fusion", "fusion_robust"):
            assert get(m, split, "s2_removed")[0]["note"] == "" and get(m, split, "s1_removed")[0]["note"] == ""
        for setting in rb.CLOUD_SETTINGS:
            n_synth = len([r for r in rows if r["split"] == split and r["setting"] == setting])
            assert n_synth == 4 * (1 + 5 * len(rb.CLOUD_SEEDS))
    md = (out / "summary.md").read_text()
    assert "Final reporting only" in md and "audit" in md and "n/a" in md and "†" in md and "⚠" in md
    assert "Opaque synthetic clouds (stricter test)" in md and "semi-transparent" in md and "opaque 100%" in md


def test_cloud_cache_gives_identical_results(runs):
    cache = runs["tmp"] / "cache"
    first = _run(runs, runs["tmp"] / "a", cache_dir=cache, figures=False)
    second = _run(runs, runs["tmp"] / "b", cache_dir=cache, figures=False)
    nocache = _run(runs, runs["tmp"] / "c", figures=False)
    # (coverages x seeds + the empty opaque cloud at 0%) x (2 test + 1 bolivia chips); thick and
    # opaque share each layer
    n_clouds = (5 * len(rb.CLOUD_SEEDS) + 1) * (2 + 1)
    assert first["clouds_generated"] == n_clouds and len(list(cache.glob("*.npz"))) == n_clouds
    assert second["clouds_generated"] == 0  # every cloud came from the cache
    assert nocache["clouds_generated"] == n_clouds  # each cloud generated once, even without a cache dir
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
    z = cache.apply(s2, cid, 0.0, 1, opaque=True)  # opaque path at 0%: blended, but must equal the input
    assert z is not s2 and np.array_equal(z, s2, equal_nan=True)
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
    assert rb.status(("s1",), "thick_clouds") == rb.status(("s1",), "opaque_clouds") == rb.NOTE_NO_S2
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


# ----------------------------------------------------------------------------- opaque clouds

def _ground(seed, h=64, w=64):
    """Raw S2 DN (13, h, w) with a water strip (dark NIR/SWIR) and land, plus one nodata pixel."""
    rng = np.random.default_rng(seed)
    s2 = rng.uniform(500, 3000, (13, h, w)).astype(np.float32)
    label = np.zeros((h, w), np.int64)
    label[:, : w // 3] = 1
    from floodfusion.chips import S2_BANDS
    for b, (water, land) in {"B3": (900, 800), "B8": (300, 2800), "B11": (150, 2200)}.items():
        s2[S2_BANDS.index(b)] = np.where(label == 1, water, land) * rng.uniform(0.9, 1.1, (h, w))
    s2[:, 5, 5] = np.nan
    return s2, label


@pytest.mark.parametrize("coverage", [0.3, 0.6, 1.0])
def test_opaque_cloud_same_mask_and_no_ground_signal(coverage):
    from floodfusion.clouds_synth import apply_clouds, cloud_params, opaque_alpha

    s2_a, _ = _ground(1)
    s2_b, _ = _ground(2)  # a different ground under the same cloud
    for seed in range(4):  # large and cumulus clouds
        alpha, mask, shade, spectrum = cloud_params(64, 64, coverage, seed, thin=False)
        op = opaque_alpha(alpha, mask)
        assert mask.mean() == pytest.approx(coverage, abs=1 / mask.size)
        assert (op[mask] == 1.0).all() and np.array_equal(op[~mask], alpha[~mask])  # same mask, same fringe
        assert (alpha[mask] < 1.0).any()  # the thick cloud it came from is not opaque
        a, b = apply_clouds(s2_a, op, spectrum, shade), apply_clouds(s2_b, op, spectrum, shade)
        inside = mask & np.isfinite(s2_a).all(0)
        assert np.array_equal(a[:, inside], b[:, inside])  # nothing of the ground survives inside the mask
        assert np.array_equal(a[:, inside], (spectrum[:, None] * shade[inside][None]).astype(np.float32))
        assert np.isnan(a[:, 5, 5]).all()  # nodata stays nodata
        thick = apply_clouds(s2_a, alpha, spectrum, shade)
        assert np.array_equal(a[:, ~mask], thick[:, ~mask], equal_nan=True)  # outside the mask: as thick
    alpha0, mask0, _, _ = cloud_params(64, 64, 0.0, 0, thin=False)
    assert not mask0.any() and not opaque_alpha(alpha0, mask0).any()  # coverage 0: nothing to replace


def test_cache_opaque_shares_the_thick_layer(synth_root, tmp_path):
    from floodfusion.chips import load_raw
    from floodfusion.clouds_synth import cloud_params

    cid = Sen1Floods11(synth_root, "test", modalities=("s1",)).ids[0]
    s2 = load_raw(synth_root, cid, "s2")
    h, w = s2.shape[1:]
    cache = rb.CloudCache(tmp_path / "c")
    thick, opaque = cache.apply(s2, cid, 0.6, 2), cache.apply(s2, cid, 0.6, 2, opaque=True)
    assert cache.generated == 1 and len(list((tmp_path / "c").glob("*.npz"))) == 1  # one layer for both
    _, mask, shade, spectrum = cloud_params(h, w, 0.6, rb.eval_cloud_seed(cid, 0.6, 2), thin=False)
    inside = mask & np.isfinite(s2).all(0)
    assert np.array_equal(opaque[:, inside], (spectrum[:, None] * shade[inside][None]).astype(np.float32))
    assert np.array_equal(opaque[:, ~mask], thick[:, ~mask], equal_nan=True)
    # a cache file from before the opaque setting (no mask) is regenerated, not misread
    (f,) = (tmp_path / "c").glob("*.npz")
    with np.load(f) as z:
        np.savez(f, alpha=z["alpha"], shade=z["shade"], spectrum=z["spectrum"])
    again = rb.CloudCache(tmp_path / "c")
    assert np.array_equal(again.apply(s2, cid, 0.6, 2, opaque=True), opaque, equal_nan=True) and again.generated == 1


def test_opaque_zero_sanity_check_fails_loudly(runs, monkeypatch):
    # a broken opaque path that clouds the whole chip even where the mask is empty
    monkeypatch.setattr(rb, "opaque_alpha", lambda alpha, mask: np.ones_like(alpha))
    with pytest.raises(RuntimeError, match=r"(?s)SANITY CHECK FAILED.*opaque clouds at coverage 0.*s2_only"):
        _run(runs, runs["tmp"] / "o", figures=False)


def test_opaque_drop_warning():
    def row(model, setting, cov, iou, seed="", note=""):
        return {"model": model, "split": "test", "setting": setting, "coverage": cov, "cloud_seed": seed,
                "iou": iou, "precision": "0.5", "recall": "0.5", "n_chips": 2, "note": note}

    def em(label, mods):
        return rb.EvalModel(label=label, run_dir=Path(label), cfg={}, stats={}, mods=mods, family=label,
                            robust=False, epoch=1, smoke=False)

    def rows(opaque_iou):
        out = []
        for m in ("s2_only", "early_fusion"):
            out.append(row(m, "thick_clouds", "0.0", "0.8"))
            out += [row(m, s, "1.0", v, seed=k) for s, v in (("thick_clouds", "0.7"), ("opaque_clouds", opaque_iou))
                    for k in rb.CLOUD_SEEDS]
            out.append(row(m, "s2_removed", "", "" if m == "s2_only" else "0.6",
                           note=rb.NOTE_NA_S2 if m == "s2_only" else ""))
        return out

    models = [em("s2_only", ("s2",)), em("early_fusion", ("s1", "s2"))]
    lines, warnings = rb.opaque_drop(models, rows("0.05"), "test")
    assert not warnings and len(lines) == 2 and "S2 removed n/a" in lines[0] and "S2 removed 0.6000" in lines[1]
    _, warnings = rb.opaque_drop(models, rows("0.7"), "test")  # s2_only still sees water: flagged
    assert len(warnings) == 1 and "s2_only" in warnings[0] and "WARNING" in warnings[0]


def test_leakage_measure_separates_thick_from_opaque():
    from floodfusion.cloud_leakage import chip_leakage, masked_box_mean, roc_auc
    from floodfusion.clouds_synth import apply_clouds, cloud_params, opaque_alpha

    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(2, 7, 30).astype(float)  # with ties
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert roc_auc(pos, neg) == pytest.approx(brute) and roc_auc(pos, pos) == pytest.approx(0.5)
    x = rng.normal(size=(7, 9))
    valid = rng.random((7, 9)) > 0.3
    got = masked_box_mean(x, valid, 3)
    i, j = 3, 4
    win, vw = x[i - 1:i + 2, j - 1:j + 2], valid[i - 1:i + 2, j - 1:j + 2]
    assert np.isnan(got[~valid]).all() and (not valid[i, j] or got[i, j] == pytest.approx(win[vw].mean()))

    s2, label = _ground(3)
    alpha, mask, shade, spectrum = cloud_params(64, 64, 1.0, 0, thin=False)
    thick = chip_leakage(s2, label, apply_clouds(s2, alpha, spectrum, shade), alpha, mask, min_pixels=50)
    op_alpha = opaque_alpha(alpha, mask)
    opaque = chip_leakage(s2, label, apply_clouds(s2, op_alpha, spectrum, shade), op_alpha, mask, min_pixels=50)
    assert thick["alpha"].max() < 1.0 and (opaque["alpha"] == 1.0).all()
    assert thick["index"]["MNDWI"]["cloud"]["auc"] > 0.9  # water still visible through the thick cloud
    assert abs(opaque["index"]["MNDWI"]["cloud"]["diff"]) < 1e-9  # no water/land difference left
    assert opaque["index"]["MNDWI"]["clear"]["auc"] == thick["index"]["MNDWI"]["clear"]["auc"] > 0.9  # reference
    assert thick["corr"]["B8"] > 0.3 and abs(opaque["corr"]["B8"]) < 0.3


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


def test_qualitative_figure_shows_thick_vs_opaque(runs, synth_cfg, monkeypatch):
    import floodfusion.viz as viz
    from floodfusion.chips import load_raw
    from floodfusion.training import forward_model

    out = runs["tmp"] / "runs_eval"
    dirs = [*runs["dirs"].values(), _dummy_run(synth_cfg, out, "s2_only_robust", ["s2"], robust=True),
            _dummy_run(synth_cfg, out, "early_fusion_robust", ["s1", "s2"], robust=True)]
    seen, real = {}, viz.plot_qualitative

    def spy(rows, out_path, **kw):
        seen["rows"] = rows
        return real(rows, out_path, **kw)

    monkeypatch.setattr(viz, "plot_qualitative", spy)
    res = rb.run_robustness(dirs, runs["root"], runs["tmp"] / "q", cloud_csv=runs["cloud_csv"], splits=["test"],
                            device=CPU, log=quiet)
    assert res["written"]["qualitative"].is_file()
    (row,) = seen["rows"]  # one test chip has a cloud fraction
    cid = row["chip_id"]
    assert [t for t, _ in row["optical"]] == ["S2 clear", "S2 thick 100%", "S2 opaque 100%"]
    assert [(m, t) for m, t, _ in row["preds"]] == [(m, t) for m in rb.QUAL_MODELS
                                                    for t in ("thick 100%", "opaque 100%")]  # robust models only
    s2 = load_raw(runs["root"], cid, "s2")
    cache = rb.CloudCache()
    assert np.array_equal(row["optical"][0][1], s2, equal_nan=True)
    assert np.array_equal(row["optical"][1][1], cache.apply(s2, cid, 1.0, 0), equal_nan=True)
    assert np.array_equal(row["optical"][2][1], cache.apply(s2, cid, 1.0, 0, opaque=True), equal_nan=True)
    # each prediction is the model's output on that very input
    by_label = {m.label: m for m in res["models"]}
    for (model, title, pred) in row["preds"]:
        m = by_label[model]
        hook = rb.CloudHook(1.0, 0, rb.CloudCache(), opaque=title.startswith("opaque"))
        ds = Sen1Floods11(runs["root"], "test", modalities=("s1", "s2"), s2_bands=m.cfg["data"].get("s2_bands"),
                          stats=m.stats, s2_augment=hook)
        batch = next(b for b in torch.utils.data.DataLoader(ds, batch_size=1) if b["id"][0] == cid)
        with torch.no_grad():
            want = forward_model(m.model, batch, m.cfg, CPU).argmax(1)[0].numpy()
        assert np.array_equal(pred, want), (model, title)


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


# ----------------------------------------------------------------------------- training seeds

def _row(model, setting, cov, iou, seed="", note="", split="test", n=2):
    return {"model": model, "split": split, "setting": setting, "coverage": cov, "cloud_seed": seed,
            "iou": iou, "precision": "0.5", "recall": "0.25", "n_chips": n, "note": note}


def _ap(v):  # CSV values have 6 decimals
    return pytest.approx(v, abs=1e-6)


def _em(label, config, seed, mods=("s1", "s2")):
    return rb.EvalModel(label=label, run_dir=Path(label), cfg={"name": config, "seed": seed}, stats={}, mods=mods,
                        family=config, robust=False, epoch=1, smoke=False)


def test_aggregate_pools_cloud_seeds_then_training_seeds():
    # run A: cloud seeds 0.5/0.6/0.7 (mean 0.6); run B: 0.7/0.8/0.9 (mean 0.8)
    rows = []
    for label, base, removed in (("m_seed0", 0.5, "0.4"), ("m_seed1", 0.7, "0.6")):
        rows.append(_row(label, "thick_clouds", "0.0", "0.9"))
        rows += [_row(label, "opaque_clouds", "1.0", f"{base + 0.1 * k:.6f}", seed=k) for k in rb.CLOUD_SEEDS]
        rows.append(_row(label, "s2_removed", "", removed))
    rows.append(_row("one", "thick_clouds", "0.0", "0.7"))
    rows += [_row("one", "opaque_clouds", "1.0", v, seed=k) for k, v in zip(rb.CLOUD_SEEDS, ("0.1", "0.2", "0.6"))]
    rows.append(_row("one", "s2_removed", "", "", note=rb.NOTE_NA_S2, n=0))
    models = [_em("m_seed1", "m", 1), _em("m_seed0", "m", 0), _em("one", "one", 0)]
    groups = rb.model_groups(models)
    assert [(c, [m.label for m in runs]) for c, runs in groups] == [("m", ["m_seed0", "m_seed1"]), ("one", ["one"])]
    agg = {(r["model"], r["setting"], r["coverage"]): r for r in rb.aggregate_rows(rows, groups)}
    assert list(rb.aggregate_rows(rows, groups)[0]) == list(rb.AGG_COLUMNS)

    r = agg[("m", "opaque_clouds", "1.0")]  # mean of run means; std over the 2 run means, not 6 values
    assert (r["n_seeds"], r["train_seeds"], r["std_over"]) == (2, "0 1", rb.STD_TRAIN)
    assert float(r["iou"]) == _ap(0.7) and float(r["iou_std"]) == _ap(np.std([0.6, 0.8], ddof=1))
    assert float(r["precision"]) == _ap(0.5) and float(r["recall"]) == _ap(0.25)
    r = agg[("m", "s2_removed", "")]  # one value per run: still a std over training seeds
    assert float(r["iou"]) == _ap(0.5) and float(r["iou_std"]) == _ap(np.std([0.4, 0.6], ddof=1))

    # one training seed: exactly the per-run sweep (std over cloud seeds); single values have std 0
    got = rb.agg_sweep(list(agg.values()), "one", "test", "opaque_clouds")
    want = rb.sweep_stats(rows, "one", "test", setting="opaque_clouds")
    assert len(got) == len(want) == 1 and all(g == _ap(w) for g, w in zip(got[0], want[0]))
    r = agg[("one", "opaque_clouds", "1.0")]
    assert (r["n_seeds"], r["std_over"]) == (1, rb.STD_CLOUD)
    assert float(r["iou_std"]) == _ap(np.std([0.1, 0.2, 0.6], ddof=1))
    assert (agg[("one", "thick_clouds", "0.0")]["iou_std"], agg[("one", "thick_clouds", "0.0")]["std_over"]) == (0.0, "")
    na = agg[("one", "s2_removed", "")]
    assert math.isnan(na["iou"]) and math.isnan(na["iou_std"]) and (na["note"], na["n_chips"]) == (rb.NOTE_NA_S2, 0)
    assert rb.agg_csv_row(na)["iou"] == "" and rb.agg_csv_row(r)["iou_std"] == f"{float(r['iou_std']):.6f}"

    with pytest.raises(ValueError, match="m_seed1"):  # runs of one config must cover the same settings
        rb.aggregate_rows([r for r in rows if not (r["model"] == "m_seed1" and r["setting"] == "s2_removed")], groups)


def test_training_seeds_grouped_end_to_end(runs, synth_cfg, monkeypatch):
    import floodfusion.viz as viz

    out = runs["tmp"] / "runs_eval"
    extra = _dummy_run(synth_cfg, out, "fusion_robust", ["s1", "s2"], arch="fusionnet", robust=True, seed=1)
    seen = {}
    real_cov, real_qual = viz.plot_iou_vs_coverage, viz.plot_qualitative

    def spy_cov(panels, styles, out_path, **kw):
        seen.setdefault("cov", []).append((panels, kw))
        return real_cov(panels, styles, out_path, **kw)

    def spy_qual(rows, out_path, **kw):
        seen["qual"] = rows
        return real_qual(rows, out_path, **kw)

    monkeypatch.setattr(viz, "plot_iou_vs_coverage", spy_cov)
    monkeypatch.setattr(viz, "plot_qualitative", spy_qual)
    dirs = [*runs["dirs"].values(), extra]
    res = rb.run_robustness(dirs, runs["root"], runs["tmp"] / "ms", cloud_csv=runs["cloud_csv"], device=CPU, log=quiet)
    labels = [m.label for m in res["models"]]
    assert labels == ["s1_only", "s2_only", "early_fusion", "fusion_robust_seed0", "fusion_robust_seed1"]

    rows = rb.read_csv(runs["tmp"] / "ms" / "robustness.csv")  # per run, as before
    agg = rb.read_csv(runs["tmp"] / "ms" / "robustness_by_model.csv")
    assert agg == [{k: str(v) for k, v in rb.agg_csv_row(r).items()} for r in res["by_model"]]
    assert sorted({r["model"] for r in agg}) == ["early_fusion", "fusion_robust", "s1_only", "s2_only"]
    for split in rb.SPLITS:
        per_run = [dict((c, mu) for c, mu, _ in rb.sweep_stats(rows, lab, split, setting="opaque_clouds"))[1.0]
                   for lab in ("fusion_robust_seed0", "fusion_robust_seed1")]
        assert per_run[0] != per_run[1]  # two different models
        r = next(r for r in agg if r["model"] == "fusion_robust" and r["split"] == split
                 and r["setting"] == "opaque_clouds" and r["coverage"] == "1.0")
        assert (r["n_seeds"], r["train_seeds"], r["std_over"]) == ("2", "0 1", rb.STD_TRAIN)
        assert float(r["iou"]) == pytest.approx(np.mean(per_run), abs=1e-6)
        assert float(r["iou_std"]) == pytest.approx(np.std(per_run, ddof=1), abs=1e-6)
        r = next(r for r in agg if r["model"] == "s2_only" and r["split"] == split
                 and r["setting"] == "opaque_clouds" and r["coverage"] == "1.0")
        assert (r["n_seeds"], r["std_over"]) == ("1", rb.STD_CLOUD)  # single seed: as before

    md = (runs["tmp"] / "ms" / "summary.md").read_text()
    assert "| fusion_robust | 2 |" in md and "| s2_only | 1 |" in md and "over training seeds" in md
    assert "| fusion_robust_seed0 |" in md and "| fusion_robust_seed1 |" in md  # per-run tables kept
    assert "Per run: opaque synthetic clouds" in md

    # figures: one line per config; band = std over training seeds for the multi-seed model
    panels, kw = seen["cov"][0]
    assert set(panels[1][1]) == {"s1_only", "s2_only", "early_fusion", "fusion_robust"}
    assert kw["names"]["fusion_robust"] == "fusion_robust (2 seeds)" and "training seeds" in kw["band_label"]
    by_model = [r for r in res["by_model"] if r["split"] == "test"]
    assert panels[1][1]["fusion_robust"] == rb.agg_sweep(by_model, "fusion_robust", "test", "opaque_clouds")
    # qualitative: the lowest training seed of fusion_robust, named with its seed
    assert {m for m, _, _ in seen["qual"][0]["preds"]} == {"fusion_robust (seed 0)"}


def test_coverage_zero_check_stays_per_run(runs, synth_cfg):
    out = runs["tmp"] / "runs_eval"
    extra = _dummy_run(synth_cfg, out, "fusion_robust", ["s1", "s2"], arch="fusionnet", robust=True, seed=1)
    path = extra / "eval_test" / "metrics.json"
    ref = json.loads(path.read_text())
    ref["metrics"]["tp"] += 1  # seed 1's stored evaluation no longer matches its best.pt
    path.write_text(json.dumps(ref))
    with pytest.raises(RuntimeError, match=r"(?s)SANITY CHECK FAILED on test.*fusion_robust_seed1") as e:
        rb.run_robustness([*runs["dirs"].values(), extra], runs["root"], runs["tmp"] / "z",
                          cloud_csv=runs["cloud_csv"], splits=["test"], device=CPU, figures=False, log=quiet)
    assert "fusion_robust_seed0" not in str(e.value)


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
