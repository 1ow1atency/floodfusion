"""Synthetic clouds (clouds_synth), the augmentation transforms and their dataset wiring."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

import floodfusion.transforms as transforms
from floodfusion.clouds_synth import (
    BRIGHTNESS, SHADE, THICK_CLOUD_DN, THIN_OPACITY, cloud_layer, cloudify, fractal_noise,
)
from floodfusion.chips import S2_BANDS, load_raw
from floodfusion.data import Sen1Floods11
from floodfusion.training import forward_model, make_loaders, resolve_config, stack_inputs
from floodfusion.transforms import ModalityDropout, SyntheticClouds

VIS = [S2_BANDS.index(b) for b in ("B2", "B3", "B4")]


def ground(h=64, w=96, seed=0):
    """Fake raw S2 (13, H, W) in DN, clear-sky-like (~1000), with one nodata pixel (NaN)."""
    g = np.random.default_rng(seed)
    s2 = (900 + 300 * g.random((13, h, w))).astype(np.float32)
    s2[:, 0, 0] = np.nan
    return s2


# ----------------------------------------------------------------------------- clouds_synth

@pytest.mark.parametrize("thin", [False, True], ids=["thick", "thin"])
@pytest.mark.parametrize("h, w", [(64, 96), (128, 128)])
def test_coverage_matches_target(thin, h, w):
    for coverage in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        alpha, mask, _ = cloud_layer(h, w, coverage, np.random.default_rng(7), thin=thin)
        assert abs(mask.mean() - coverage) <= 1 / (h * w) + 1e-9, coverage
        assert alpha.dtype == np.float32 and alpha.min() >= 0 and alpha.max() <= 1
        assert (alpha[mask] > 0).all()  # every covered pixel has some cloud
        if coverage == 0:
            assert not alpha.any()


def test_thick_and_thin_opacity():
    thick, mt, _ = cloud_layer(128, 128, 0.6, np.random.default_rng(1), thin=False)
    thin, mn, _ = cloud_layer(128, 128, 0.6, np.random.default_rng(1), thin=True)
    assert thick[mt].mean() > 0.7  # mostly opaque
    assert thin.max() <= THIN_OPACITY[1] and thin[mn].mean() < 0.4  # ground stays visible


def test_cloudify_is_deterministic():
    s2 = ground()
    a = cloudify(s2, 0.4, seed=3)
    assert np.array_equal(a, cloudify(s2, 0.4, seed=3), equal_nan=True)
    assert not np.array_equal(a, cloudify(s2, 0.4, seed=4), equal_nan=True)
    state = np.random.get_state()[1].copy()
    cloudify(s2, 0.4, seed=3)
    assert np.array_equal(np.random.get_state()[1], state)  # global NumPy RNG untouched


def test_cloudify_values():
    s2 = ground()
    assert np.array_equal(cloudify(s2, 0.0, seed=1), s2, equal_nan=True)  # coverage 0: unchanged
    out, alpha, mask = cloudify(s2, 1.0, seed=1, thin=False, return_alpha=True)
    assert out.shape == s2.shape and out.dtype == np.float32
    assert np.isnan(out[:, 0, 0]).all() and np.isfinite(out[:, 1:, :]).all()  # nodata stays NaN
    mask[0, 0] = False  # the nodata pixel
    vis = out[VIS][:, mask]
    assert (vis > s2[VIS][:, mask]).all()  # clouds brighten the visible bands
    lo = THICK_CLOUD_DN[VIS].min() * BRIGHTNESS[0] * (1 - SHADE[False]) * 0.9 * 0.85
    hi = THICK_CLOUD_DN[VIS].max() * BRIGHTNESS[1] * (1 + SHADE[False])
    assert lo < np.nanmedian(vis) < hi
    b10 = S2_BANDS.index("B10")
    assert np.nanmedian(out[b10]) < 500  # thick (low/mid) clouds stay dark in the cirrus band
    thin = cloudify(s2, 1.0, seed=1, thin=True)
    assert np.nanmedian(thin[b10]) > np.nanmedian(out[b10])  # cirrus is bright in B10


def test_cloudify_input_checks():
    with pytest.raises(ValueError, match="all 13 bands"):
        cloudify(ground()[:4], 0.5, seed=0)
    with pytest.raises(ValueError, match="coverage"):
        cloudify(ground(), 1.5, seed=0)


def test_fractal_noise_is_standardised():
    n = fractal_noise(64, 64, np.random.default_rng(0), stretch=3, angle=0.7)
    assert n.shape == (64, 64) and abs(n.mean()) < 1e-5 and abs(n.std() - 1) < 1e-4


# ----------------------------------------------------------------------------- transforms

def test_synthetic_clouds_transform(monkeypatch):
    s2 = ground()
    assert SyntheticClouds(p=0.0, coverage=(0.1, 1.0))(s2) is s2
    seen = []
    monkeypatch.setattr(transforms, "cloudify", lambda x, cov, seed, thin_p: seen.append((cov, seed)) or x + 1)
    torch.manual_seed(0)
    t = SyntheticClouds(p=0.5, coverage=(0.2, 0.6))
    n_cloudy = sum(t(s2) is not s2 for _ in range(400))
    assert 160 < n_cloudy < 240 and len(seen) == n_cloudy
    assert all(0.2 <= c <= 0.6 for c, _ in seen)
    torch.manual_seed(0)
    again = []
    monkeypatch.setattr(transforms, "cloudify", lambda x, cov, seed, thin_p: again.append((cov, seed)) or x)
    for _ in range(400):
        t(s2)
    assert again == seen  # reproducible from torch's seed
    for bad in [dict(p=1.5, coverage=(0, 1)), dict(p=0.5, coverage=(0.8, 0.2)), dict(p=0.5, coverage=(0, 1.2))]:
        with pytest.raises(ValueError):
            SyntheticClouds(**bad)


def _sample():
    return {"id": "x", "s1": torch.randn(2, 8, 8), "s2": torch.randn(13, 8, 8), "label": torch.zeros(8, 8)}


def test_modality_dropout_never_drops_both_and_sets_mask():
    torch.manual_seed(0)
    t = ModalityDropout(p_drop_s1=0.1, p_drop_s2=0.2)
    counts = {"none": 0, "s1": 0, "s2": 0}
    for _ in range(3000):
        x = _sample()
        y = t(x)
        mask = y["modality_mask"]
        assert mask.dtype == torch.bool and mask.shape == (2,) and mask.any()  # never both dropped
        for i, m in enumerate(("s1", "s2")):
            if mask[i]:
                assert torch.equal(y[m], x[m])
            else:
                assert not y[m].any()  # dropped -> zeros (train mean after normalisation)
        counts["none" if mask.all() else "s1" if not mask[0] else "s2"] += 1
    assert abs(counts["s1"] / 3000 - 0.1) < 0.02 and abs(counts["s2"] / 3000 - 0.2) < 0.025
    assert ModalityDropout(0, 0)(_sample())["modality_mask"].all()
    with pytest.raises(ValueError):
        ModalityDropout(0.6, 0.6)
    with pytest.raises(KeyError):
        ModalityDropout(0.1, 0.1)({"s1": torch.zeros(2, 4, 4)})


# ----------------------------------------------------------------------------- dataset

def test_s2_augment_runs_on_raw_dn_before_clipping(synth_root, synth_cfg):
    stats = json.loads(open(synth_cfg["data"]["stats"]).read())
    seen = {}

    def aug(x, cid):
        seen[cid] = x.copy()
        return x + 1000.0

    bands = ["B4", "B8"]
    plain = Sen1Floods11(synth_root, "train", stats=stats, s2_bands=bands)
    cloudy = Sen1Floods11(synth_root, "train", stats=stats, s2_bands=bands, s2_augment=aug)
    cid = cloudy.ids[0]
    out = cloudy[0]
    raw = seen[cid]
    assert raw.shape[0] == 13
    assert np.array_equal(raw, load_raw(synth_root, cid, "s2"), equal_nan=True)  # raw DN, NaN = missing
    idx = [S2_BANDS.index(b) for b in bands]
    s = stats["s2"]
    exp = (raw[idx] + 1000.0)
    lo, hi = np.array(s["clip_min"])[idx, None, None], np.array(s["clip_max"])[idx, None, None]
    exp = (np.clip(exp, lo, hi) - np.array(s["mean"])[idx, None, None]) / np.array(s["std"])[idx, None, None]
    exp[np.isnan(exp)] = 0
    assert torch.allclose(out["s2"], torch.from_numpy(exp.astype(np.float32)), atol=1e-5)
    ref = plain[0]
    assert torch.equal(out["s1"], ref["s1"]) and torch.equal(out["label"], ref["label"])


def test_synthetic_clouds_leave_s1_and_labels_untouched(synth_root, synth_cfg):
    stats = json.loads(open(synth_cfg["data"]["stats"]).read())
    plain = Sen1Floods11(synth_root, "train", stats=stats)
    torch.manual_seed(0)
    cloudy = Sen1Floods11(synth_root, "train", stats=stats, s2_augment=SyntheticClouds(1.0, (1.0, 1.0)))
    for i in range(len(plain)):
        a, b = plain[i], cloudy[i]
        assert torch.equal(a["s1"], b["s1"]) and torch.equal(a["label"], b["label"])
        assert not torch.equal(a["s2"], b["s2"]) and torch.isfinite(b["s2"]).all()


def test_augment_off_changes_nothing(synth_cfg):
    """Loaders from a config without an augment section (as before), with the section
    disabled but non-default values, and with defaults: identical batches."""
    stats = json.loads(open(synth_cfg["data"]["stats"]).read())
    old = copy.deepcopy(synth_cfg)
    del old["augment"]
    off = copy.deepcopy(synth_cfg)
    off["augment"]["synthetic_clouds"].update(enabled=False, p=1.0)
    off["augment"]["modality_dropout"].update(enabled=False, p_drop_s2=0.9)
    batches = []
    for cfg in (old, resolve_config(off), synth_cfg):
        torch.manual_seed(0)
        tl, _ = make_loaders(cfg, stats, smoke=False, device=torch.device("cpu"),
                             generator=torch.Generator().manual_seed(0))
        batches.append(list(tl))
        assert tl.dataset.s2_augment is None and isinstance(tl.dataset.transform, transforms.TrainTransform)
    for other in batches[1:]:
        for a, b in zip(batches[0], other):
            assert a.keys() == b.keys() and "modality_mask" not in a
            assert all(torch.equal(a[k], b[k]) for k in ("s1", "s2", "label"))


def _robust(cfg, modalities=("s1", "s2"), clouds=True, dropout=True, **drop):
    c = copy.deepcopy(cfg)
    c["data"]["modalities"] = list(modalities)
    c["augment"]["synthetic_clouds"].update(enabled=clouds)
    c["augment"]["modality_dropout"].update(enabled=dropout, **drop)
    return resolve_config(c)


def test_dropout_in_loader_zeroes_inputs_and_sets_mask(synth_cfg):
    stats = json.loads(open(synth_cfg["data"]["stats"]).read())
    cfg = _robust(synth_cfg, clouds=False, p_drop_s1=0.0, p_drop_s2=1.0)
    tl, vl = make_loaders(cfg, stats, smoke=False, device=torch.device("cpu"), generator=torch.Generator())
    batch = next(iter(tl))
    assert batch["modality_mask"].tolist() == [[True, False]] * batch["s1"].shape[0]
    assert not batch["s2"].any() and batch["s1"].any()
    x = stack_inputs(batch, ("s1", "s2"))  # early-fusion U-Net input: S2 channels are zero
    assert not x[:, 2:].any() and x[:, :2].any()
    assert "modality_mask" not in next(iter(vl))  # validation is never augmented

    class Spy(torch.nn.Module):
        def forward(self, s1, s2, mask):
            self.mask = mask
            return torch.zeros(s1.shape[0], 2, *s1.shape[-2:])

    spy = Spy()
    fcfg = copy.deepcopy(cfg)
    fcfg["model"].update(arch="fusionnet")
    forward_model(spy, batch, resolve_config(fcfg), torch.device("cpu"))
    assert torch.equal(spy.mask, batch["modality_mask"])  # FusionNet gets the mask


def test_augment_config_validation(synth_cfg):
    for bad, msg in [(dict(modalities=("s2",), clouds=False), "modality_dropout needs"),
                     (dict(modalities=("s1",), dropout=False), "synthetic_clouds needs s2"),
                     (dict(p_drop_s1=0.7, p_drop_s2=0.7), "sum <= 1")]:
        with pytest.raises(ValueError, match=msg):
            _robust(synth_cfg, **bad)
    c = copy.deepcopy(synth_cfg)
    c["augment"]["synthetic_clouds"]["coverage"] = [0.5, 0.2]
    with pytest.raises(ValueError, match="coverage"):
        resolve_config(c)
    c = copy.deepcopy(synth_cfg)
    c["augment"]["cutmix"] = {}
    with pytest.raises(ValueError, match="unknown augment"):
        resolve_config(c)
    c = copy.deepcopy(synth_cfg)
    c["augment"]["modality_dropout"]["p"] = 0.3
    with pytest.raises(ValueError, match="unknown key"):
        resolve_config(c)


def test_true_colour_grid_figure(tmp_path):
    from floodfusion.viz import plot_true_colour_grid

    s2 = ground(32, 32)
    rows = [("synthetic", [(f"{c:.0%}", cloudify(s2, c, seed=i)) for i, c in enumerate((0.0, 0.5, 1.0))]),
            ("real", [("clear", s2)])]
    out = plot_true_colour_grid(rows, tmp_path / "f" / "grid.png", "test")
    assert out.is_file() and out.stat().st_size > 1000
