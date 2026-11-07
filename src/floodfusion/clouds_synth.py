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
  * thick: opacity 0.9-1.0, soft edges. Two scales, picked per chip
    (``CUMULUS_P`` = probability of the cumulus field):
      - "large": a few large clouds (noise structures up to half the chip);
      - "cumulus": a field of many small, broken cells (~20-85 px, 0.2-0.85 km), whose
        density varies across the chip (cf. Ghana_264787); at high coverage the cells merge
        into a deck, as in the cloudiest real chips (Ghana_161233, Ghana_154838).
    Inside, the cloud's brightness ("shade", multiplying its spectrum, always within
    1 +- SHADE = +-25%, so the calibrated DN range below holds) is the product of
      - depth into the cloud (0 at the edge, 1 at CORE_DEPTH noise std inside): greyer,
        slightly thinner edges and bright tops,
      - Perlin turbulence (sum over 4 octaves of |noise|, each octave at its own angle):
        "cauliflower" cell tops separated by darker creases at many scales,
      - a smooth multi-octave relief;
    opacity is also slightly lower (down to 1 - EDGE_THINNING) at edges and in creases;
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
``cloudify_as(s2, coverage, seed, kind)`` forces "large", "cumulus" or "thin" (figures, tests).
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
EDGE_WIDTH = {False: 0.6, True: 1.5}  # thick / thin, in noise standard deviations
TEXTURE = {False: 0.15, True: 0.5}  # fraction of opacity modulated by a second noise field
SHADE = {False: 0.25, True: 0.1}  # max +-variation of the cloud's own brightness (relief, edges)
PERSISTENCE = {False: 0.6, True: 0.5}  # thick: more small-scale detail
CUMULUS_P = 0.5  # thick clouds: probability of a cumulus field instead of a few large clouds
CUMULUS_CELLS = (12, 25)  # cumulus noise cells per chip side: ~20-43 px cells at 512 px (as Ghana_264787)
CORE_DEPTH = 1.0  # noise std above the cloud threshold at which a thick cloud is "deep" (bright top)
EDGE_THINNING = 0.15  # thick clouds: opacity in shallow edges / dark lanes down to (1 - this) of the core's
THICK_MODES = ("large", "cumulus")
CIRRUS_STRETCH = 3  # thin clouds: features 3x longer along a random direction
OCTAVES, BASE_CELLS = 6, 2  # largest structures ~1/2 of the chip (~2.5 km at 10 m)
NOISE_DOWNSAMPLE = 4  # noise is generated at 1/4 resolution and upsampled (detail >= ~40 m)


def fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = OCTAVES,
                  base_cells: int = BASE_CELLS, persistence: float = 0.5, stretch: float = 1.0,
                  angle: float = 0.0, downsample: int = NOISE_DOWNSAMPLE) -> np.ndarray:
    """Fractal value noise (fBm), (h, w) float32 with zero mean and unit std.
    ``stretch`` > 1 makes features that many times longer along direction ``angle`` (radians).
    Computed on a grid ``downsample`` times coarser, then bilinearly upsampled (much faster)."""
    if downsample > 1 and min(h, w) >= 4 * downsample:
        small = _fractal_noise_full(-(-h // downsample), -(-w // downsample), rng, octaves,
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


def cloud_layer(h: int, w: int, coverage: float, rng: np.random.Generator, thin: bool = False,
                mode: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opacity map (h, w) float32 in [0, 1], covered mask (h, w) bool with
    ``mask.mean() == coverage`` (up to one pixel), and shade (h, w): the cloud's relative
    brightness, within 1 +- SHADE, that multiplies its spectrum.

    Thick clouds: ``mode`` "large" or "cumulus"; None draws it (CUMULUS_P). The draw is made
    even when ``mode`` is given, so a forced mode gives exactly the clouds that a drawn
    mode would for the same RNG state.
    """
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0, 1], got {coverage}")
    if mode is not None and mode not in THICK_MODES:
        raise ValueError(f"mode must be one of {THICK_MODES} or None, got {mode!r}")
    lo, hi = THIN_OPACITY if thin else THICK_OPACITY
    opacity = rng.uniform(lo, hi)
    angle = rng.uniform(0, np.pi)
    drawn = "cumulus" if rng.random() < CUMULUS_P else "large"
    mode = None if thin else (mode or drawn)
    if thin:
        shape_noise = fractal_noise(h, w, rng, persistence=PERSISTENCE[True], stretch=CIRRUS_STRETCH, angle=angle)
    elif mode == "large":
        shape_noise = fractal_noise(h, w, rng, persistence=PERSISTENCE[False])
    else:  # cumulus field: small cells, density modulated at large scale
        cells = int(rng.integers(*CUMULUS_CELLS))
        small = fractal_noise(h, w, rng, octaves=3, base_cells=cells, persistence=0.4, downsample=2)
        density = fractal_noise(h, w, rng, octaves=3, base_cells=2, persistence=0.5)
        shape_noise = _standardise(small + 0.45 * density)
    texture = _unit(fractal_noise(h, w, rng, base_cells=4, stretch=CIRRUS_STRETCH if thin else 1.0, angle=angle))
    relief = _unit(fractal_noise(h, w, rng, base_cells=4, persistence=0.65))
    # Cloud cells separated by darker creases at many scales, also inside a closed deck
    # ("cauliflower" tops): Perlin turbulence, the sum over octaves of |noise|, which is ~0
    # along each octave's zero-crossings. 0 = crease / lane, 1 = cell top.
    cell_scale = (8, 16) if mode == "cumulus" else (4, 8)
    lanes = turbulence(h, w, rng, base_cells=int(rng.integers(*cell_scale)))
    if coverage <= 0.0:
        shade = (1 + SHADE[thin] * (2 * relief - 1)).astype(np.float32)
        return np.zeros((h, w), np.float32), np.zeros((h, w), bool), shade
    n = h * w
    flat = shape_noise.reshape(-1)
    order = np.argsort(flat, kind="stable")
    k = n if coverage >= 1.0 else int(round(coverage * n))  # covered pixels
    mask = np.zeros(n, bool)
    mask[order[n - k:]] = True
    mask = mask.reshape(h, w)
    # Threshold between the last uncovered and first covered value; ramp = 0.5 there.
    t = 0.5 * (flat[order[n - k - 1]] + flat[order[n - k]]) if 0 < k < n else float(flat[order[n - k]])
    if coverage >= 1.0:
        ramp = np.ones((h, w), np.float32)
    else:
        ramp = _fade(np.clip((shape_noise - t) / EDGE_WIDTH[thin] + 0.5, 0.0, 1.0)).astype(np.float32)
        ramp[mask] = np.maximum(ramp[mask], 0.5)  # ties at the threshold count as covered
    tex = TEXTURE[thin]
    alpha = opacity * ramp * ((1 - tex) + tex * texture)
    if thin:
        shade = 1 + SHADE[True] * (2 * relief - 1)
    else:
        # depth into the cloud: 0 at the threshold, 1 at CORE_DEPTH std above it
        core = _fade(np.clip((shape_noise - t) / CORE_DEPTH, 0.0, 1.0))
        depth = core ** 0.75
        top = depth * lanes ** 1.5 * (0.55 + 0.45 * relief)  # in [0, 1]: bright tops, dark creases
        alpha = alpha * ((1 - EDGE_THINNING) + EDGE_THINNING * depth * np.sqrt(lanes))  # thinner edges / creases
        shade = 1 + SHADE[False] * (2 * top - 1)  # 1 - SHADE in lanes and edges, 1 + SHADE on tops
    return alpha.astype(np.float32), mask, shade.astype(np.float32)


def turbulence(h: int, w: int, rng: np.random.Generator, base_cells: int, octaves: int = 4,
               persistence: float = 0.6) -> np.ndarray:
    """Perlin turbulence in [0, 1]: sum over octaves of |independent noise|, rescaled. Finer
    octaves are generated at higher resolution so creases stay a few pixels wide."""
    out = np.zeros((h, w), np.float32)
    for o in range(octaves):
        n = fractal_noise(h, w, rng, octaves=1, base_cells=base_cells * 2 ** o, angle=rng.uniform(0, np.pi),
                          downsample=4 if o < 2 else 2)  # own angle per octave: no grid-aligned creases
        out += persistence ** o * np.abs(n)
    return _unit(out)


def _standardise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return (x / max(float(x.std()), 1e-12)).astype(np.float32)


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
    return _cloudify(s2, coverage, seed, thin, thin_p, None, return_alpha)


def cloudify_as(s2: np.ndarray, coverage: float, seed: int, kind: str, return_alpha: bool = False):
    """``cloudify`` with a forced cloud kind: "large" or "cumulus" (thick), or "thin". For
    figures and targeted tests; training and evaluation use ``cloudify``."""
    if kind not in (*THICK_MODES, "thin"):
        raise ValueError(f"kind must be one of {(*THICK_MODES, 'thin')}, got {kind!r}")
    thin = kind == "thin"
    return _cloudify(s2, coverage, seed, thin, THIN_P, None if thin else kind, return_alpha)


def _cloudify(s2, coverage, seed, thin, thin_p, mode, return_alpha):
    _check_s2(s2)
    rng = np.random.default_rng(seed)
    is_thin = bool(rng.random() < thin_p) if thin is None else bool(thin)
    brightness = rng.uniform(*BRIGHTNESS)
    alpha, mask, shade = cloud_layer(s2.shape[1], s2.shape[2], coverage, rng, thin=is_thin, mode=mode)
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
