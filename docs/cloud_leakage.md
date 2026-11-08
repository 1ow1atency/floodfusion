# Ground signal under the synthetic evaluation clouds

Local sample: 12 chips with < 10% real cloud (s2cloudless), cloud seeds [0, 1, 2] of the robustness evaluation (thick clouds, large or cumulus). `thick` = the clouds as used in training; `opaque` = the same mask with opacity 1 inside it. All statistics are over the covered pixels (inside the mask).

## Opacity inside the mask (pooled over chips and seeds)

| clouds | coverage | min | p1 | p5 | p25 | p50 | mean | share < 0.9 | share < 0.8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| thick | 50% | 0.331 | 0.393 | 0.485 | 0.740 | 0.783 | 0.757 | 99.0% | 61.5% |
| thick | 100% | 0.664 | 0.733 | 0.754 | 0.787 | 0.812 | 0.813 | 98.5% | 37.8% |
| opaque | 50% | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0% | 0.0% |
| opaque | 100% | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0% | 0.0% |

## Correlation of clouded vs original S2 inside the mask (median over chips and seeds)

corr² is the share of the clouded band's variance (inside the cloud) that comes from the ground.

| clouds | coverage | B2 | B3 | B4 | B8 | B11 | B12 |
|---|--:|--:|--:|--:|--:|--:|--:|
| thick | 50% | 0.083 | 0.073 | 0.134 | 0.399 | 0.343 | 0.256 |
| thick | 100% | 0.093 | 0.105 | 0.143 | 0.340 | 0.324 | 0.325 |
| opaque | 50% | -0.008 | -0.019 | -0.009 | -0.014 | -0.011 | -0.023 |
| opaque | 100% | 0.003 | -0.002 | -0.002 | 0.009 | -0.026 | -0.017 |

## Water vs land inside the mask (median over chips and seeds)

AUC: 0.5 = index carries no water information, 1 = perfectly separates water and land. '9×9' = index averaged over the covered pixels of a 9×9 px window (spatial context, as a CNN has). Only (chip, seed) pairs with ≥ 200 water and land pixels under the cloud. Δ = mean index of water − land.

### NDWI (B3 − B8) / (sum)

| clouds | coverage | n | Δ clear | Δ clouded | AUC clear | AUC clouded | AUC clear 9×9 | AUC clouded 9×9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| thick | 50% | 27 | +0.391 | +0.055 | 0.985 | 0.984 | 0.981 | 0.974 |
| thick | 100% | 27 | +0.409 | +0.041 | 0.984 | 0.986 | 0.984 | 0.976 |
| opaque | 50% | 27 | +0.391 | +0.000 | 0.985 | 0.500 | 0.981 | 0.506 |
| opaque | 100% | 27 | +0.409 | -0.000 | 0.984 | 0.499 | 0.984 | 0.494 |

### MNDWI (B3 − B11) / (sum)

| clouds | coverage | n | Δ clear | Δ clouded | AUC clear | AUC clouded | AUC clear 9×9 | AUC clouded 9×9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| thick | 50% | 27 | +0.534 | +0.051 | 0.989 | 0.967 | 0.988 | 0.961 |
| thick | 100% | 27 | +0.537 | +0.030 | 0.979 | 0.973 | 0.974 | 0.965 |
| opaque | 50% | 27 | +0.534 | +0.000 | 0.989 | 0.500 | 0.988 | 0.512 |
| opaque | 100% | 27 | +0.537 | +0.000 | 0.979 | 0.500 | 0.974 | 0.503 |

