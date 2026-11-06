"""Segmentation models: the U-Net baselines and FusionNet.

ImageNet weights with != 3 input channels (smp ``patch_first_conv``, smp 0.6): only the
first convolution changes. Input channel i gets a copy of the pretrained RGB filter
``i % 3`` (R, G, B, R, G, ...), and the whole kernel is scaled by 3 / C so the first
layer's output keeps roughly the magnitude it had on RGB. 2 channels: VV <- R, VH <- G,
x1.5. 13 or 15 channels: the RGB filters repeat across bands, x3/13 or x3/15. Everything
after the first convolution keeps its ImageNet weights unchanged. FusionNet's two encoders
are built with the same smp call (``get_encoder``), so they are adapted the same way.

FusionNet (two-branch encoder with cross-attention fusion)
-----------------------------------------------------------
SAR (VV, VH) and optical (13 S2 bands) images go through two separate ResNet-34 encoders,
each ImageNet-initialised and adapted to its own number of bands, so neither modality has
to share low-level filters with the other. Each encoder produces features at strides 2, 4,
8, 16 and 32. The two feature maps of every level are fused into one map, which becomes
that level's skip connection in a single U-Net decoder (the same decoder as the baselines,
256-128-64-32-16 channels, 2 output classes). The encoders never see each other's
features; interaction happens only in the fusion modules.

At the two deepest levels (strides 16 and 32: 256 and 512 channels, a 16x16 / 8x8 token
grid for a 256x256 crop) the features are fused with bidirectional cross-attention. Every
SAR position queries all optical positions and every optical position queries all SAR
positions (8-head attention), each with a pre-norm residual, ``x + MHA(LN(x), LN(y))``.
Then the two attended maps are concatenated and merged by a 1x1 convolution (+ BatchNorm,
ReLU). At this depth each token describes a large area, and attention lets each modality
pick the context from the other one that it needs; for example, SAR can supply evidence
where the optical image is cloudy. At the shallower, high-resolution levels (strides 2, 4,
8) attention over all positions would be expensive and the features are mostly local
edges and textures, so the two maps are simply concatenated and merged with a 1x1
convolution. Queries and keys carry a fixed 2D sinusoidal positional encoding, half the
channels for the row and half for the column, computed from the absolute token index for
whatever grid size arrives. The token spacing is therefore the same for 256x256 training
crops and 512x512 evaluation chips, and the larger grid simply extends the smaller one.
With ``fusion="concat"`` all five levels use concatenation + 1x1 convolution: this is
the ablation that isolates the effect of cross-attention with the same two encoders.

Missing modality: ``forward(s1, s2, modality_mask)``, where ``modality_mask`` is (N, 2) bool
(S1 present, S2 present); at least one modality must be present for every sample. An
encoder runs only on the samples whose modality is present; the missing samples' feature
rows are zeros, so the missing input (zeros, noise, even NaN) never reaches the network,
and it never enters BatchNorm statistics. In cross-attention, a present modality's update
from a missing one is gated off (its residual input passes unchanged), which is equivalent
to masking every key and value of the missing modality but avoids a softmax over an empty
key set. The missing modality's own branch is set to zero before the merge convolution. So
for a sample with S2 missing, the output does not depend on its S2 input at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .chips import S2_BANDS

NUM_CLASSES = 2  # 0 = not water, 1 = water; outputs logits (N, 2, H, W)
ARCHS = ("unet", "fusionnet")
FUSIONS = ("cross_attention", "concat")
DECODER_CHANNELS = (256, 128, 64, 32, 16)  # smp.Unet default, same as the baselines
ATTENTION_LEVELS = (4, 5)  # encoder levels (strides 16 and 32) fused by cross-attention
OUTPUT_STRIDE = 32
_FROM_CONFIG = object()


def build_unet(in_channels: int, encoder: str = "resnet34", encoder_weights: str | None = "imagenet") -> nn.Module:
    """U-Net with ``in_channels`` inputs and 2 output logits. ``encoder_weights=None``
    gives random initialisation (used in tests, no download)."""
    import segmentation_models_pytorch as smp  # heavy import, only when a model is built

    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=NUM_CLASSES)


def build_model(cfg: dict, encoder_weights=_FROM_CONFIG) -> nn.Module:
    """Model for a (resolved) config: ``model.arch`` "unet" (default, the baselines) or
    "fusionnet". ``encoder_weights`` overrides the config (evaluation passes None)."""
    m, d = cfg["model"], cfg["data"]
    weights = m["encoder_weights"] if encoder_weights is _FROM_CONFIG else encoder_weights
    arch = m.get("arch", "unet")
    mods = tuple(d["modalities"])
    s2_channels = len(d.get("s2_bands") or S2_BANDS)
    if arch == "unet":
        return build_unet((2 if "s1" in mods else 0) + (s2_channels if "s2" in mods else 0), m["encoder"], weights)
    if arch == "fusionnet":
        if mods != ("s1", "s2"):
            raise ValueError(f"fusionnet needs modalities [s1, s2], got {list(mods)}")
        return FusionNet(s1_channels=2, s2_channels=s2_channels, encoder=m["encoder"], encoder_weights=weights,
                         fusion=m.get("fusion", "cross_attention"), attention_heads=int(m.get("attention_heads", 8)))
    raise ValueError(f"model.arch must be one of {ARCHS}, got {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- FusionNet

def sinusoidal_2d(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """Fixed 2D sinusoidal positional encoding, (h*w, dim), row-major over the grid.

    The first dim/2 channels encode the row index, the last dim/2 the column index, each as
    sin/cos pairs at geometric frequencies 1/10000^(2k/(dim/2)). Uses absolute indices, so
    position (i, j) gets the same code for any grid size (a 32x32 grid extends a 16x16 one).
    """
    if dim % 4:
        raise ValueError(f"positional encoding dim must be divisible by 4, got {dim}")
    half = dim // 2
    freqs = torch.exp(torch.arange(0, half, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / half))

    def axis(n):
        ang = torch.arange(n, device=device, dtype=torch.float32)[:, None] * freqs[None]
        return torch.stack([ang.sin(), ang.cos()], dim=-1).flatten(1)  # (n, half): sin, cos interleaved

    py, px = axis(h), axis(w)
    return torch.cat([py[:, None].expand(h, w, half), px[None].expand(h, w, half)], dim=-1).reshape(h * w, dim)


def _merge(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class ConcatFuse(nn.Module):
    """Concatenate the two maps (missing modality zeroed), then 1x1 conv + BN + ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.merge = _merge(2 * channels, channels)

    def forward(self, a, b, m1, m2):  # a, b: (N, C, H, W); m1, m2: (N, 1, 1, 1) float
        # Missing rows are already zero (_encode); the mask product keeps that true by construction.
        return self.merge(torch.cat([a * m1, b * m2], dim=1))


class CrossAttentionFuse(nn.Module):
    """Bidirectional cross-attention (S1 <- S2 and S2 <- S1), pre-norm residual, then a
    1x1 merge. Positional encoding on queries and keys only."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels} channels are not divisible by {heads} attention heads")
        self.norm_a, self.norm_b = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.attn_ab = nn.MultiheadAttention(channels, heads, batch_first=True)  # S1 queries, S2 keys/values
        self.attn_ba = nn.MultiheadAttention(channels, heads, batch_first=True)  # S2 queries, S1 keys/values
        self.merge = _merge(2 * channels, channels)

    def forward(self, a, b, m1, m2):
        n, c, h, w = a.shape
        ta, tb = a.flatten(2).transpose(1, 2), b.flatten(2).transpose(1, 2)  # (N, HW, C)
        pos = sinusoidal_2d(h, w, c, device=a.device)[None]
        na, nb = self.norm_a(ta), self.norm_b(tb)
        upd_a = self.attn_ab(na + pos, nb + pos, nb, need_weights=False)[0]
        upd_b = self.attn_ba(nb + pos, na + pos, na, need_weights=False)[0]
        g1, g2 = m1.view(n, 1, 1), m2.view(n, 1, 1)
        ta = ta + g2 * upd_a  # no update from a missing S2
        tb = tb + g1 * upd_b  # no update from a missing S1
        a2, b2 = (t.transpose(1, 2).reshape(n, c, h, w) for t in (ta, tb))
        return self.merge(torch.cat([a2 * m1, b2 * m2], dim=1))


def _encode(encoder: nn.Module, x: torch.Tensor, present: torch.Tensor) -> list[torch.Tensor]:
    """Encoder features for the present samples only; rows of missing samples are zeros."""
    if bool(present.all()):
        return encoder(x)
    idx = present.nonzero().squeeze(1)
    n, (h, w) = x.shape[0], x.shape[-2:]
    if idx.numel() == 0:
        return [x.new_zeros(n, c, h // 2 ** i, w // 2 ** i) for i, c in enumerate(encoder.out_channels)]
    return [f.new_zeros((n, *f.shape[1:])).index_copy(0, idx, f) for f in encoder(x[idx])]


class FusionNet(nn.Module):
    """Two ResNet encoders (S1, S2), per-level fusion, one U-Net decoder. See module docstring."""

    def __init__(self, s1_channels: int = 2, s2_channels: int = len(S2_BANDS), encoder: str = "resnet34",
                 encoder_weights: str | None = "imagenet", fusion: str = "cross_attention", attention_heads: int = 8):
        super().__init__()
        from segmentation_models_pytorch.base import SegmentationHead
        from segmentation_models_pytorch.base import initialization as init
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        from segmentation_models_pytorch.encoders import get_encoder

        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        self.fusion = fusion
        self.encoder_s1 = get_encoder(encoder, in_channels=s1_channels, depth=5, weights=encoder_weights)
        self.encoder_s2 = get_encoder(encoder, in_channels=s2_channels, depth=5, weights=encoder_weights)
        chans = self.encoder_s1.out_channels  # (in, 64, 64, 128, 256, 512) for resnet18/34
        if tuple(chans[1:]) != tuple(self.encoder_s2.out_channels[1:]):
            raise ValueError("both encoders must produce the same feature channels")
        self.fuse = nn.ModuleList([
            CrossAttentionFuse(c, attention_heads) if fusion == "cross_attention" and level in ATTENTION_LEVELS
            else ConcatFuse(c)
            for level, c in enumerate(chans) if level > 0])
        # Level 0 (the raw input) is dropped by UnetDecoder; only its spatial size is used.
        self.decoder = UnetDecoder(encoder_channels=(0, *chans[1:]), decoder_channels=DECODER_CHANNELS,
                                   n_blocks=5, use_norm="batchnorm", add_center_block=False,
                                   attention_type=None, interpolation_mode="nearest")
        self.segmentation_head = SegmentationHead(DECODER_CHANNELS[-1], NUM_CLASSES, activation=None, kernel_size=3)
        init.initialize_decoder(self.decoder)  # same initialisation as smp.Unet
        init.initialize_head(self.segmentation_head)

    @staticmethod
    def check_mask(mask: torch.Tensor | None, n: int, device) -> torch.Tensor:
        if mask is None:
            return torch.ones(n, 2, dtype=torch.bool, device=device)
        if mask.dtype != torch.bool or mask.shape != (n, 2):
            raise ValueError(f"modality_mask must be a bool tensor of shape ({n}, 2), got "
                             f"{mask.dtype} {tuple(mask.shape)}")
        if bool((~mask).all(dim=1).any()):
            raise ValueError("modality_mask: every sample needs at least one modality (S1 or S2)")
        return mask.to(device)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        n, (h, w) = s1.shape[0], s1.shape[-2:]
        if s2.shape[0] != n or s2.shape[-2:] != (h, w):
            raise ValueError(f"s1 {tuple(s1.shape)} and s2 {tuple(s2.shape)} must match in batch and size")
        if h % OUTPUT_STRIDE or w % OUTPUT_STRIDE:
            raise ValueError(f"input size {h}x{w} must be divisible by {OUTPUT_STRIDE}")
        mask = self.check_mask(modality_mask, n, s1.device)
        fa = _encode(self.encoder_s1, s1, mask[:, 0])
        fb = _encode(self.encoder_s2, s2, mask[:, 1])
        m1, m2 = (mask[:, i].to(fa[1].dtype).view(n, 1, 1, 1) for i in (0, 1))
        fused = [s1[:, :0]] + [f(a, b, m1, m2) for f, a, b in zip(self.fuse, fa[1:], fb[1:])]
        return self.segmentation_head(self.decoder(fused))
