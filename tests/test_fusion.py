"""FusionNet: shapes, positional encoding, missing-modality masking, gradients, builder."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from floodfusion.losses import CEDiceLoss
from floodfusion.models import FusionNet, build_model, build_unet, count_parameters, sinusoidal_2d
from floodfusion.training import resolve_config

FUSIONS = ("cross_attention", "concat")


def make(fusion="cross_attention", train=False, trained_biases=False):
    """resnet18 FusionNet. trained_biases: random biases / LayerNorm shifts, as after training
    (at init they are all 0, which would hide an ungated update from an all-zero modality)."""
    torch.manual_seed(0)
    m = FusionNet(encoder="resnet18", encoder_weights=None, fusion=fusion)
    if trained_biases:
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.endswith("bias"):
                    p.add_(torch.randn_like(p) * 0.1)
    return m.train(train)


def inputs(n=3, size=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, size, size, generator=g), torch.randn(n, 13, size, size, generator=g)


@pytest.mark.parametrize("fusion", FUSIONS)
@pytest.mark.parametrize("size", [256, 512])  # training crops and full validation chips
def test_output_shapes(fusion, size):
    s1, s2 = inputs(1, size)
    with torch.no_grad():
        out = make(fusion)(s1, s2, torch.tensor([[True, True]]))
    assert out.shape == (1, 2, size, size) and torch.isfinite(out).all()


def test_positional_encoding_is_size_independent():
    small, big = sinusoidal_2d(16, 16, 256), sinusoidal_2d(32, 32, 256)
    assert small.shape == (256, 256) and big.shape == (1024, 256)
    # The same token position gets the same code on a 16x16 (256 crop) and 32x32 (512 chip) grid.
    assert torch.allclose(big.view(32, 32, 256)[:16, :16], small.view(16, 16, 256))
    assert torch.unique(big, dim=0).shape[0] == 1024  # every position distinct
    with pytest.raises(ValueError, match="divisible by 4"):
        sinusoidal_2d(4, 4, 6)


def test_positional_encoding_is_used(monkeypatch):
    import floodfusion.models as models

    model, (s1, s2) = make(), inputs(2)
    with torch.no_grad():
        ref = model(s1, s2)
        monkeypatch.setattr(models, "sinusoidal_2d", lambda h, w, dim, device=None: torch.zeros(h * w, dim))
        assert not torch.allclose(model(s1, s2), ref)


def test_attention_only_where_configured():
    attn = lambda m: [n for n, mod in m.named_modules() if isinstance(mod, nn.MultiheadAttention)]
    assert sorted({n.split(".")[1] for n in attn(make("cross_attention"))}) == ["3", "4"]  # strides 16, 32
    assert attn(make("concat")) == []


@pytest.mark.parametrize("fusion", FUSIONS)
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("missing", [0, 1], ids=["s1_missing", "s2_missing"])
def test_missing_modality_input_does_not_leak(fusion, train, missing):
    model = make(fusion, train, trained_biases=True)
    s1, s2 = inputs()
    mask = torch.ones(3, 2, dtype=torch.bool)
    mask[0, missing] = False
    x = [s1, s2]
    x[missing][0] = 0  # a missing modality's input is zeros
    with torch.no_grad():
        ref = model(*x, mask)
        for junk in (torch.randn_like(x[missing][0]) * 100, torch.full_like(x[missing][0], float("nan"))):
            y = [t.clone() for t in x]
            y[missing][0] = junk
            out = model(*y, mask)
            # every sample's output is unchanged (train mode: not even via BatchNorm batch stats)
            assert torch.equal(out, ref)
        # sanity: the present modality of that sample still matters
        y = [t.clone() for t in x]
        y[1 - missing][0] += 1
        assert not torch.equal(model(*y, mask)[0], ref[0])


@pytest.mark.parametrize("fusion", FUSIONS)
@pytest.mark.parametrize("missing", [0, 1], ids=["s1_missing", "s2_missing"])
def test_missing_branch_is_zero_at_every_merge(fusion, missing):
    """The missing modality's half of each merge input is exactly zero (in cross-attention its
    branch would otherwise carry the attention update computed from the present modality)."""
    model = make(fusion, trained_biases=True)
    seen = []
    for f in model.fuse:
        f.merge.register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
    mask = torch.ones(3, 2, dtype=torch.bool)
    mask[1, missing] = False
    s1, s2 = inputs()
    with torch.no_grad():
        model(s1, s2, mask)
        present_enc, present_x = (model.encoder_s2, s2) if missing == 0 else (model.encoder_s1, s1)
        own = present_enc(present_x[1:2])  # the present modality's own encoder features
    assert len(seen) == 5
    for level, x in enumerate(seen, start=1):
        c = x.shape[1] // 2
        gone, kept = (x[:, :c], x[:, c:]) if missing == 0 else (x[:, c:], x[:, :c])
        assert torch.all(gone[1] == 0) and gone[0].abs().sum() > 0 and gone[2].abs().sum() > 0
        # No update from the missing modality: the present branch is its plain encoder features.
        assert torch.allclose(kept[1], own[level][0], atol=1e-5), level


def test_full_mask_equals_no_mask_and_bad_masks_raise():
    model = make()
    s1, s2 = inputs(2)
    with torch.no_grad():
        assert torch.equal(model(s1, s2, torch.ones(2, 2, dtype=torch.bool)), model(s1, s2))
    for bad, msg in [(torch.tensor([[True, True], [False, False]]), "at least one modality"),
                     (torch.ones(2, 2), "bool tensor"), (torch.ones(2, 3, dtype=torch.bool), "shape")]:
        with pytest.raises(ValueError, match=msg):
            model(s1, s2, bad)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(s1[..., :48, :48], s2[..., :48, :48])


def _backward(model, mask):
    s1, s2 = inputs(2)
    y = torch.randint(-1, 2, (2, 64, 64))
    CEDiceLoss()(model(s1, s2, mask), y).backward()


def _grad_norm(params) -> float:
    return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_gradients_reach_both_encoders_and_fusion(fusion):
    model = make(fusion, train=True)
    _backward(model, None)
    for name in ("encoder_s1.conv1.weight", "encoder_s2.conv1.weight", "encoder_s1.layer4.0.conv1.weight",
                 "encoder_s2.layer4.0.conv1.weight", "fuse.0.merge.0.weight", "fuse.4.merge.0.weight"):
        g = model.get_parameter(name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    if fusion == "cross_attention":
        for lvl in (3, 4):
            for d in ("attn_ab", "attn_ba"):
                assert _grad_norm(model.fuse[lvl].get_submodule(d).parameters()) > 0, (lvl, d)


def test_no_gradient_from_a_missing_modality():
    model = make(train=True)
    _backward(model, torch.tensor([[True, False], [True, False]]))  # S2 missing for the whole batch
    assert _grad_norm(model.encoder_s2.parameters()) == 0
    assert _grad_norm(model.fuse[3].attn_ab.parameters()) == 0  # S1 <- S2 update is gated off
    assert _grad_norm(model.encoder_s1.parameters()) > 0


def test_concat_variant_trains():
    model = make("concat", train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s1, s2 = inputs(2)
    y = (s1[:, 0] > 0).long()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = CEDiceLoss()(model(s1, s2), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def _cfg(arch="unet", modalities=("s1", "s2"), **model):
    return {"name": "x", "seed": 0, "output_dir": "o",
            "data": {"modalities": list(modalities), "s2_bands": None},
            "model": {"arch": arch, "encoder": "resnet18", "encoder_weights": None, **model},
            "train": {}}


def test_build_model_and_config_validation():
    unet = build_model(_cfg())
    assert count_parameters(unet) == count_parameters(build_unet(15, "resnet18", None))
    assert count_parameters(build_model(_cfg(modalities=("s1",)))) == count_parameters(build_unet(2, "resnet18", None))
    assert isinstance(build_model(_cfg("fusionnet", fusion="concat")), FusionNet)
    with pytest.raises(ValueError, match="needs modalities"):
        build_model(_cfg("fusionnet", modalities=("s2",)))
    resolved = resolve_config(_cfg("fusionnet"))["model"]
    assert resolved["fusion"] == "cross_attention" and resolved["attention_heads"] == 8
    cfg = _cfg()
    del cfg["model"]["arch"]  # configs from before FusionNet
    assert resolve_config(cfg)["model"]["arch"] == "unet"
    for bad, msg in [(_cfg("vit"), "model.arch"), (_cfg("fusionnet", fusion="sum"), "model.fusion"),
                     (_cfg("fusionnet", attention_heads=0), "attention_heads"),
                     (_cfg("fusionnet", modalities=("s1",)), "needs data.modalities")]:
        with pytest.raises(ValueError, match=msg):
            resolve_config(copy.deepcopy(bad))
