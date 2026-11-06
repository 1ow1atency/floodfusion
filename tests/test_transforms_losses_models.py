"""Tests for floodfusion.transforms, floodfusion.losses and floodfusion.models."""

from __future__ import annotations

import pytest
import torch

from floodfusion.losses import CEDiceLoss, masked_cross_entropy, soft_dice_loss
from floodfusion.models import build_unet
from floodfusion.transforms import TrainTransform

# ----------------------------------------------------------------------------- transforms


def _coded_sample(h=16, w=16):
    """Every tensor encodes the pixel position, so any misalignment is detectable."""
    code = torch.arange(h * w).reshape(h, w)
    return {"id": "x", "s1": torch.stack([code.float(), -code.float()]),
            "s2": torch.stack([code.float()] * 13), "label": code.clone()}


def test_same_crop_and_geometry_for_every_tensor():
    torch.manual_seed(0)
    seen = set()
    for _ in range(200):
        s = TrainTransform(crop_size=8)(_coded_sample())
        assert s["id"] == "x"
        assert s["s1"].shape == (2, 8, 8) and s["s2"].shape == (13, 8, 8) and s["label"].shape == (8, 8)
        assert torch.equal(s["s1"][0].long(), s["label"]) and torch.equal(s["s1"][1].long(), -s["label"])
        assert all(torch.equal(s["s2"][c].long(), s["label"]) for c in range(13))
        lab = s["label"]
        # The crop is a rotated/flipped contiguous 8x8 window: neighbours differ by 1 or 16.
        steps = {abs(int(lab[0, 1] - lab[0, 0])), abs(int(lab[1, 0] - lab[0, 0]))}
        assert steps == {1, 16}
        seen.add((int(lab[0, 1] - lab[0, 0]), int(lab[1, 0] - lab[0, 0])))
    assert len(seen) == 8  # all 8 flip/rotation combinations occur


def test_transform_is_seeded_and_does_not_modify_input():
    sample = _coded_sample()
    before = {k: v.clone() for k, v in sample.items() if k != "id"}
    torch.manual_seed(3)
    a = TrainTransform(8)(sample)
    torch.manual_seed(3)
    b = TrainTransform(8)(sample)
    assert all(torch.equal(a[k], b[k]) for k in ("s1", "s2", "label"))
    assert all(torch.equal(sample[k], before[k]) for k in before)


def test_transform_options_and_errors():
    s = TrainTransform(crop_size=None, flips=False, rot90=False)(_coded_sample())
    assert torch.equal(s["label"], _coded_sample()["label"])
    with pytest.raises(ValueError, match="larger than chip"):
        TrainTransform(crop_size=32)(_coded_sample())
    bad = _coded_sample()
    bad["s1"] = bad["s1"][:, :8]
    with pytest.raises(ValueError, match="spatial shape"):
        TrainTransform(8)(bad)


# ----------------------------------------------------------------------------- losses


def _loss_case():
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 4, 5)
    target = torch.randint(0, 2, (2, 4, 5))
    target[:, :, -1] = -1          # last column ignored
    target[1, 0, 0] = -1
    return logits, target


@pytest.mark.parametrize("fn", [masked_cross_entropy, soft_dice_loss, CEDiceLoss()])
def test_ignored_pixels_do_not_change_loss_or_get_gradient(fn):
    logits, target = _loss_case()
    ignored = (target == -1).unsqueeze(1).expand_as(logits)
    other = logits.clone()
    other[ignored] = torch.randn(int(ignored.sum())) * 100  # wild values where ignored
    assert torch.allclose(fn(logits, target), fn(other, target))
    x = logits.clone().requires_grad_(True)
    fn(x, target).backward()
    assert torch.all(x.grad[ignored] == 0) and x.grad[~ignored].abs().sum() > 0


@pytest.mark.parametrize("fn", [masked_cross_entropy, soft_dice_loss])
def test_loss_equals_loss_without_the_ignored_pixels(fn):
    logits, target = _loss_case()
    target[1, 0, 0] = 0                             # only the last column is ignored now
    assert torch.allclose(fn(logits, target), fn(logits[..., :-1], target[..., :-1]))


def test_all_ignored_batch_gives_zero_loss_and_no_nan():
    logits = torch.randn(1, 2, 3, 3, requires_grad=True)
    target = torch.full((1, 3, 3), -1)
    loss = CEDiceLoss()(logits, target)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_dice_hand_computed():
    # Water probability 0.5 everywhere (equal logits); 4 valid pixels, 2 of them water.
    logits = torch.zeros(1, 2, 1, 5)
    target = torch.tensor([[[1, 1, 0, 0, -1]]])
    # inter = 0.5 * 2 = 1; sum(p) = 2; sum(t) = 2 -> dice = (2 + 1) / (4 + 1) = 0.6
    assert soft_dice_loss(logits, target).item() == pytest.approx(1 - 0.6)
    assert masked_cross_entropy(logits, target).item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_loss_shape_errors():
    with pytest.raises(ValueError, match=r"\(N, 2, H, W\)"):
        CEDiceLoss()(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="does not match"):
        CEDiceLoss()(torch.zeros(1, 2, 2, 2), torch.zeros(1, 3, 3, dtype=torch.long))


# ----------------------------------------------------------------------------- models


@pytest.mark.parametrize("c", [2, 13, 15])
def test_unet_shapes(c):
    model = build_unet(c, encoder="resnet18", encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, c, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert model.encoder.conv1.in_channels == c
