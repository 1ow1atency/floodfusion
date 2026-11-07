"""Synthetic clouds for Sentinel-2 (numpy only, torch-free like chips/stats/viz/clouds).

A cloud layer is an opacity map ``alpha`` (H, W) in [0, 1], blended into the RAW S2 bands
(L1C DN = TOA reflectance x 10000, NaN = missing) before clipping and normalisation:

    s2_cloudy[b] = (1 - alpha) * s2[b] + alpha * cloud_dn[b]

Only S2 is changed: radar (S1) sees through clouds, and labels describe the ground, so
both stay untouched. NaN (S2 nodata) stays NaN.

Shape: the opacity comes from fractal value noise (Perlin-style fBm: 6 octaves of
smoothly interpolated random grids, each octave double the frequency and half the
amplitude of the previous one). Cloud cover is the area above a threshold chosen as a
quantile of the noise, so the covered fraction matches the target exactly. A pixel
counts as covered when its edge ramp is at least half way up, i.e. at least half the
layer's opacity before texture. Edges are soft (a smoothstep ramp across the threshold),
and a second noise field adds internal texture.

Two cloud types, picked per chip (``thin_p`` = probability of thin):
  * thick (cumulus-like): opacity 0.9-1.0, fairly sharp edges, mild texture, and a
    +-25% brightness relief across the cloud top;
  * thin (cirrus-like / haze): opacity 0.15-0.5, very soft edges, fibrous noise stretched
    3x along a random direction, strong texture. The ground stays visible through it.
    Measured on the hazy local chips (visible DN ~2000 over ground at ~1150): opacity ~0.2.
Coverage 0 returns the input unchanged; coverage 1 is complete overcast.

Cloud reflectance (DN at opacity 1, before a per-chip brightness factor of 0.85-1.15).
Measured on the cloud pixels of the 8 cloudiest local train chips, using the
s2cloudless mask from floodfusion.clouds (Ghana, 71-98% cloud, 1.5M pixels): the
brightest quarter has a visible DN of p50 ~2800-3000 and p90 ~4400-4600, B11 ~3300 / 5000,
B12 ~2600 / 4000, B9 ~1200 / 2200, B10 ~100 / 170. Those clouds are hazy and partly
transparent (the ground's NIR shows through), so an opaque cloud is set between the
median and p90 of the brightest quarter, close to spectrally flat from visible to NIR
(water droplets scatter all these wavelengths alike), and darker in SWIR, where water
and ice absorb:
  * B1-B5 4000-4300 (B1 slightly higher: Rayleigh scattering), B6-B8A 4500-4800;
  * B9 2000 (water-vapour absorption band);
  * B10 130: low and mid clouds sit under most of the water vapour that blocks 1375 nm,
    as in the measured chips;
  * B11 3600, B12 2700 ("somewhat bright in SWIR").
With the per-chip brightness factor (0.85-1.15) and the cloud-top relief of thick clouds
(+-25%), visible DN inside opaque cloud spans ~2500-5700, the range of the measured cloud
pixels; no brighter than real clouds, so brightness alone does not give synthetic clouds away.
Thin cirrus is ice cloud high above that vapour: B10 1000 (the cirrus band exists to
see it) and darker SWIR (B11 2800, B12 1900), and it is scaled down by its low opacity.
Not modelled: cloud shadows, and haze outside the cloud mask.

``cloudify(s2, coverage, seed)`` is deterministic: the same inputs and seed always give
the same clouds, so every model can be evaluated on identical cloudy inputs. Training
draws coverage and seed from torch's RNG (floodfusion.transforms.SyntheticClouds).
``FixedClouds`` derives the seed from the chip id alone (``chip_seed``), so a chip gets
the same clouds in every epoch, every run and every model, without touching any RNG.
"""

from __future__ import annotations

import zlib

import numpy as np

from .chips import S2_BANDS

_THICK = {"B1": 4300, "B2": 4100, "B3": 4000, "B4": 4000, "B5": 4100, "B6": 4500, "B7": 4700,
          "B8": 4600, "B8A": 4800, "B9": 2000, "B10": 130, "B11": 3600, "B12": 2700}
_THIN = {**_THICK, "B10": 1000, "B11": 2800, "B12": 1900}
THICK_CLOUD_DN = np.array([_THICK[b] for b in S2_BANDS], dtype=np.float32)
THIN_CLOUD_DN = np.array([_THIN[b] for b in S2_BANDS], dtype=np.float32)
THICK_OPACITY, THIN_OPACITY = (0.9, 1.0), (0.15, 0.5)
BRIGHTNESS = (0.85, 1.15)  # per-chip factor on the cloud spectrum
THIN_P = 0.3
EDGE_WIDTH = {False: 0.4, True: 1.5}  # thick / thin, in noise standard deviations
TEXTURE = {False: 0.15, True: 0.5}  # fraction of opacity modulated by a second noise field
SHADE = {False: 0.25, True: 0.1}  # +-variation of the cloud's own brightness (cloud-top relief)
PERSISTENCE = {False: 0.6, True: 0.5}  # thick: more small-scale detail (cumulus cells)
CIRRUS_STRETCH = 3  # thin clouds: features 3x longer along a random direction
OCTAVES, BASE_CELLS = 6, 2  # largest structures ~1/2 of the chip (~2.5 km at 10 m)
NOISE_DOWNSAMPLE = 4  # noise is generated at 1/4 resolution and upsampled (detail >= ~40 m)


def fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = OCTAVES,
                  base_cells: int = BASE_CELLS, persistence: float = 0.5, stretch: float = 1.0,
                  angle: float = 0.0) -> np.ndarray:
    """Fractal value noise (fBm), (h, w) float32 with zero mean and unit std.
    ``stretch`` > 1 makes features that many times longer along direction ``angle`` (radians).
    Computed on a grid NOISE_DOWNSAMPLE times coarser, then bilinearly upsampled (~5x faster)."""
    if NOISE_DOWNSAMPLE > 1 and min(h, w) >= 4 * NOISE_DOWNSAMPLE:
        small = _fractal_noise_full(-(-h // NOISE_DOWNSAMPLE), -(-w // NOISE_DOWNSAMPLE), rng, octaves,
                                    base_cells, persistence, stretch, angle)
        out = _upsample(small, h, w)
        out -= out.mean()
        return (out / max(float(out.std()), 1e-12)).astype(np.float32)
    return _fractal_noise_full(h, w, rng, octaves, base_cells, persistence, stretch, angle)


def _fractal_noise_full(h: int, w: int, rng: np.random.Generator, octaves: int, base_cells: int,
                        persistence: float, stretch: float, angle: float) -> np.ndarray:
    size = max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) / size
    u = xx * np.cos(angle) + yy * np.sin(angle)
    v = -xx * np.sin(angle) + yy * np.cos(angle)
    u, v = (u - u.min()) / stretch, v - v.min()
    out = np.zeros((h, w), dtype=np.float64)
    for o in range(octaves):
        g = base_cells * 2 ** o
        cu, cv = u * g + rng.uniform(0, 1), v * g + rng.uniform(0, 1)
        grid = rng.standard_normal((int(cv.max()) + 2, int(cu.max()) + 2))
        i0, j0 = cv.astype(int), cu.astype(int)
        fy, fx = _fade(cv - i0), _fade(cu - j0)
        top = grid[i0, j0] + (grid[i0, j0 + 1] - grid[i0, j0]) * fx
        bottom = grid[i0 + 1, j0] + (grid[i0 + 1, j0 + 1] - grid[i0 + 1, j0]) * fx
        out += persistence ** o * (top + (bottom - top) * fy)
    out -= out.mean()
    std = out.std()
    return (out / std if std > 0 else out).astype(np.float32)


def _upsample(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of (hs, ws) to (h, w), pixel centres aligned."""
    def coords(n_out, n_in):
        c = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        i0 = np.minimum(c.astype(int), n_in - 2) if n_in > 1 else np.zeros(n_out, int)
        return i0, (c - i0).astype(np.float32)
    y0, fy = coords(h, x.shape[0])
    x0, fx = coords(w, x.shape[1])
    rows = x[y0] * (1 - fy[:, None]) + x[np.minimum(y0 + 1, x.shape[0] - 1)] * fy[:, None]
    return rows[:, x0] * (1 - fx[None]) + rows[:, np.minimum(x0 + 1, x.shape[1] - 1)] * fx[None]


def _fade(t):
    return t * t * (3 - 2 * t)  # smoothstep: continuous slope across grid cells


def cloud_layer(h: int, w: int, coverage: float, rng: np.random.Generator,
                thin: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opacity map (h, w) float32 in [0, 1], covered mask (h, w) bool with
    ``mask.mean() == coverage`` (up to one pixel), and shade (h, w): the cloud's relative
    brightness (1 +- SHADE) that multiplies its spectrum."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0, 1], got {coverage}")
    lo, hi = THIN_OPACITY if thin else THICK_OPACITY
    opacity = rng.uniform(lo, hi)
    angle = rng.uniform(0, np.pi)
    shape_noise = fractal_noise(h, w, rng, persistence=PERSISTENCE[thin],
                                stretch=CIRRUS_STRETCH if thin else 1.0, angle=angle)
    texture = _unit(fractal_noise(h, w, rng, base_cells=4, stretch=CIRRUS_STRETCH if thin else 1.0, angle=angle))
    shade = (1 + SHADE[thin] * (2 * _unit(fractal_noise(h, w, rng, base_cells=4)) - 1)).astype(np.float32)
    if coverage <= 0.0:
        return np.zeros((h, w), np.float32), np.zeros((h, w), bool), shade
    if coverage >= 1.0:
        ramp = np.ones((h, w), np.float32)
        mask = np.ones((h, w), bool)
    else:
        n = h * w
        k = int(round(coverage * n))  # covered pixels
        order = np.argsort(shape_noise, axis=None, kind="stable")
        mask = np.zeros(n, bool)
        mask[order[n - k:]] = True
        mask = mask.reshape(h, w)
        # Threshold between the last uncovered and first covered value; ramp = 0.5 there.
        flat = shape_noise.reshape(-1)
        t = 0.5 * (flat[order[n - k - 1]] + flat[order[n - k]]) if 0 < k < n else flat[order[n - k]]
        ramp = np.clip((shape_noise - t) / EDGE_WIDTH[thin] + 0.5, 0.0, 1.0)
        ramp = _fade(ramp).astype(np.float32)
        ramp[mask] = np.maximum(ramp[mask], 0.5)  # ties at the threshold count as covered
    tex = TEXTURE[thin]
    alpha = opacity * ramp * ((1 - tex) + tex * texture)
    return alpha.astype(np.float32), mask, shade


def _unit(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / max(float(np.ptp(x)), 1e-6)  # rescale to [0, 1]


def apply_clouds(s2: np.ndarray, alpha: np.ndarray, cloud_dn: np.ndarray, shade: np.ndarray | None = None) -> np.ndarray:
    """Blend a cloud layer into raw S2 (13, H, W). NaN stays NaN."""
    _check_s2(s2)
    a = alpha[None].astype(np.float32)
    cloud = cloud_dn[:, None, None] * (1.0 if shade is None else shade[None])
    return ((1 - a) * s2 + a * cloud).astype(np.float32)


def cloudify(s2: np.ndarray, coverage: float, seed: int, thin: bool | None = None,
             thin_p: float = THIN_P, return_alpha: bool = False):
    """Deterministic synthetic clouds on raw S2 (all 13 bands, DN, NaN = missing).

    ``thin=None`` picks thin (cirrus-like) with probability ``thin_p`` from the seed;
    True / False forces the type. Returns the cloudy S2 (float32, same shape), plus the
    opacity map and covered mask if ``return_alpha``. Coverage 0 returns an unchanged copy.
    """
    _check_s2(s2)
    rng = np.random.default_rng(seed)
    is_thin = bool(rng.random() < thin_p) if thin is None else bool(thin)
    brightness = rng.uniform(*BRIGHTNESS)
    alpha, mask, shade = cloud_layer(s2.shape[1], s2.shape[2], coverage, rng, thin=is_thin)
    if coverage <= 0.0:
        out = s2.astype(np.float32, copy=True)
    else:
        spectrum = (THIN_CLOUD_DN if is_thin else THICK_CLOUD_DN) * np.float32(brightness)
        out = apply_clouds(s2, alpha, spectrum, shade)
    return (out, alpha, mask) if return_alpha else out


EVAL_CLOUD_SEED = 0  # base seed of the fixed evaluation clouds (independent of the run's seed)


def chip_seed(chip_id: str, base: int = EVAL_CLOUD_SEED) -> int:
    """Stable per-chip seed (CRC32 of base and chip id: the same in every process and Python
    version, unlike hash())."""
    return zlib.crc32(f"{base}:{chip_id}".encode())


class FixedClouds:
    """Dataset ``s2_augment`` hook with fixed clouds per chip: ``cloudify(s2, coverage,
    chip_seed(chip_id))``. Used for the information-only cloudy validation (val_iou_cloud50)."""

    def __init__(self, coverage: float, thin: bool | None = False, base_seed: int = EVAL_CLOUD_SEED):
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage must be in [0, 1], got {coverage}")
        self.coverage, self.thin, self.base_seed = float(coverage), thin, int(base_seed)

    def __call__(self, s2: np.ndarray, chip_id: str) -> np.ndarray:
        return cloudify(s2, self.coverage, chip_seed(chip_id, self.base_seed), thin=self.thin)


def _check_s2(s2: np.ndarray) -> None:
    if s2.ndim != 3 or s2.shape[0] != len(S2_BANDS):
        raise ValueError(f"synthetic clouds need raw S2 with all {len(S2_BANDS)} bands (13, H, W), "
                         f"got shape {s2.shape}")
