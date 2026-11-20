# Roadmap

## 1. Setup
- [ ] Repo skeleton, `pyproject.toml`, `.gitignore`
- [ ] Local `.venv` (Python 3.11) with `pip install -e ".[dev]"`
- [ ] GitHub repo created and pushed
- [ ] Colab launcher notebook: clone repo, install, mount Drive
- [ ] Weights & Biases project created

## 2. Data
- [ ] Sen1Floods11 v1.1 hand-labeled subset on Google Drive
- [ ] Dataset class reading S1 / S2 / label chips from split CSVs
- [ ] NaN handling for S1; label `-1` ignored throughout
- [ ] Normalisation stats computed from train split only and saved
- [ ] Synthetic GeoTIFF test fixtures + data tests

### Dataset statistics
From `docs/label_stats.json` (descriptive only) and `configs/band_stats.json` (train split,
252 chips, `allow_missing=false`, seed 0). Water fraction = water / labelled pixels (label `-1` excluded).

| Split   | Chips | Water fraction |
|---------|------:|---------------:|
| train   |   252 |          0.095 |
| valid   |    89 |          0.110 |
| test    |    90 |          0.125 |
| bolivia |    15 |          0.159 |

S1 clip limits (1st / 99th percentile of valid train pixels, subsample of 15.1% of pixels per chip):

| Band | clip_min (dB) | clip_max (dB) |
|------|--------------:|--------------:|
| VV   |       −23.861 |        −2.414 |
| VH   |       −34.152 |        −9.520 |

## 3. Baselines (U-Net)
- [ ] Training / evaluation loop, water-class IoU metric, fixed seeds, W&B logging
- [ ] SAR-only U-Net
- [ ] Optical-only U-Net
- [ ] Early-fusion U-Net (S1 + S2 channels concatenated)

### Results on real data (seed 0)
Seed 0, `best.pt` selected on valid, evaluated on full 512×512 chips. Cloud setting: real S2 as
acquired (no synthetic clouds). Metric: water-class IoU (label `-1` ignored).

| model               | valid IoU | test IoU | bolivia IoU | test precision | test recall |
|---------------------|----------:|---------:|------------:|---------------:|------------:|
| s1_only             |    0.6612 |   0.6567 |      0.6253 |         0.8168 |      0.7702 |
| s2_only             |    0.8297 |   0.8276 |      0.7908 |         0.9269 |      0.8854 |
| early_fusion        |    0.8379 |   0.8202 |      0.7922 |         0.9339 |      0.8707 |
| fusion              |    0.8375 |   0.8181 |      0.7971 |         0.9229 |      0.8781 |
| early_fusion_robust |    0.8271 |   0.8190 |      0.7818 |         0.9132 |      0.8881 |
| fusion_robust       |    0.8181 |   0.8130 |      0.7916 |         0.9159 |      0.8786 |

Early fusion does not beat optical-only on real (mostly clear) data. Hypothesis to test in
Stage 5: it relies on S2 and ignores S1, so it should collapse under clouds.

`fusion` = FusionNet (`configs/fusion.yaml`, cross-attention, no cloud/dropout training),
`best.pt` from epoch 74. On real (mostly clear) data it is within single-seed noise of
s2_only and early_fusion, as expected without cloud/dropout training.

Robust runs (`*_robust`: synthetic clouds + modality dropout in training, `best.pt` still
selected on the real valid split): `early_fusion_robust` from epoch 88, `fusion_robust` from
epoch 95. With cloud augmentation and modality dropout, early fusion's `val_iou_cloud50`
(~0.77) is close to FusionNet's (~0.78). Updated Stage 5 hypothesis: **does the architecture
still matter once both get the same robustness training?**

**Answered in Stage 5: yes, but only once the cloud is heavy.** Under opaque cloud the two
are within seed noise at 40% coverage (0.742 ± 0.008 vs 0.733 ± 0.003), and clearly apart at
80-100% (0.681 ± 0.008 vs 0.637 ± 0.021; 0.644 ± 0.007 vs 0.558 ± 0.037). The architecture
also buys seed stability: at 100% cloud early fusion's spread over training seeds is five
times FusionNet's.

**Synthetic clouds are semi-transparent (found 2026-11-08).** `s2_only_robust` reaches a
`val_iou_cloud50` of ~0.79, as high as `fusion_robust`, without any SAR. The thick clouds used
in training (and in `val_iou_cloud50`) are not opaque. Measured on the 12 local chips with
< 10% real cloud, evaluation clouds at 50% and 100% coverage, 3 cloud seeds
([docs/cloud_leakage.md](cloud_leakage.md), `scripts/cloud_leakage.py`):
- **Opacity:** median opacity inside the cloud mask is 0.78-0.81, and 99% of covered pixels
  are below 0.9 (nominal 0.9-1.0, reduced by texture, edge thinning and soft edges; minimum
  0.33 at cloud edges).
- **Ground signal:** the clouded bands still correlate with the ground (r ≈ 0.3-0.4 in B8
  and B11).
- **Water vs land:** NDWI still separates water from land under the cloud almost as well as
  in clear sky (AUC 0.98 vs 0.98). The contrast shrinks about 8×, but so does the noise.

A model trained on these clouds can learn to read water through them.

Training and the training-time clouds stay unchanged, because all robust models are already
trained with them. The robustness evaluation adds an **opaque** setting: the same masks, seeds
and coverage with opacity 1 inside the mask (AUC 0.50, no ground signal). **Opaque results are
the stricter test of cloud robustness; thick results show robustness to semi-transparent
cloud only.**

Model selection rule (all models, including the robust ones): `best.pt` is chosen on the
real valid split. Valid IoU with fixed synthetic clouds (`val_iou_cloud50`) is logged every
epoch for information only and never used for selection.

## 4. Fusion model
- [ ] Two-branch encoder (SAR branch, optical branch)
- [ ] Cross-attention fusion
- [ ] Train and compare against baselines on valid / test

## 5. Robustness evaluation — **done** (2026-11-21)
- [x] Synthetic cloud augmentation for S2
- [x] Modality dropout during training
- [x] Audit s2cloudless masks before using real-cloud bins: **done on the full data**
  (notebook 05; findings and decisions in [cloud_audit.md](cloud_audit.md)).
  - **Labels under cloud:** about half of the cloudy pixels are -1 and almost no water is
    labelled under cloud (test 60-100% bin: 0.7% of cloudy pixels are water). The real-cloud
    bins are **supporting evidence only**; the opaque synthetic sweep is the main experiment.
  - **Threshold:** keep 0.4 for the reported bins (`cloud_cover.csv`, fractions unchanged);
    `cloud_cover_t06.csv` holds the 0.6 fractions as a sensitivity check (full-data F1 vs -1:
    58.4% at 0.4 vs 62.3% at 0.6, a smaller gain than the local sample suggested, and 0.6
    shrinks the test 60-100% bin from 11 to 7 chips).
  - **Unreliable masks:** on 10 chips (none in bolivia) s2cloudless saturates at ~1.0 over
    plainly visible ground, driven by an elevated cirrus band B10; flagged
    `mask_unreliable = 1` in `cloud_cover.csv`, and the robustness evaluation reports the
    bins with and without them.
- [x] IoU vs cloud-cover curves for all models: thick (as trained) and opaque clouds, plus S2
  removed (`notebooks/04_evaluate.ipynb`). **Final run covers all 14 runs / 8 configurations**,
  with 3 training seeds for `s1_only`, `early_fusion_robust` and `fusion_robust` (results below).
- [x] Evaluation on held-out Bolivia split (evaluation only, never tuned on).

### Final robustness results (multi-seed)
Source: `docs/results/robustness.csv` (one row per run), `docs/results/robustness_by_model.csv`
(grouped by config) and `docs/results/summary.md`, from notebook 04 on Colab. 14 runs over 8
configurations; `s1_only`, `early_fusion_robust` and `fusion_robust` have training seeds 0-2,
the rest seed 0 only. Every run is its `best.pt`, selected on the real valid split. Metric:
pooled water IoU (label -1 ignored). Test has 90 chips, Bolivia 15.

Each run is first averaged over its 3 cloud seeds. For a 3-seed config, ± is then the sample
std **over training seeds**; for a 1-seed config it is the std **over cloud seeds** only, and
in clear sky (no cloud drawn) there is no spread to report.

**Main result: test split, opaque synthetic clouds** (no ground signal under the cloud):

| model | seeds | 0% | 40% | 60% | 80% | 100% | S2 removed |
|---|--:|--:|--:|--:|--:|--:|--:|
| s1_only | 3 | 0.656 ± 0.004 | 0.656 ± 0.004 | 0.656 ± 0.004 | 0.656 ± 0.004 | 0.656 ± 0.004 | 0.656 ± 0.004 † |
| s2_only_robust | 1 | 0.821 | 0.526 ± 0.016 | 0.383 ± 0.030 | 0.215 ± 0.007 | 0.064 ± 0.009 | n/a |
| early_fusion_robust | 3 | 0.817 ± 0.003 | 0.733 ± 0.003 | 0.692 ± 0.006 | 0.637 ± 0.021 | 0.558 ± 0.037 | 0.621 ± 0.019 |
| fusion_robust | 3 | 0.811 ± 0.004 | **0.742 ± 0.008** | **0.710 ± 0.008** | **0.681 ± 0.008** | 0.644 ± 0.007 | 0.641 ± 0.002 |
| fusion_concat_robust | 1 | 0.809 | 0.729 ± 0.003 | 0.686 ± 0.002 | 0.636 ± 0.004 | 0.590 ± 0.014 | 0.635 |

† unaffected: the model has no S2 input, so its clean result is shown.

- **40-100% coverage:** `fusion_robust` has the highest IoU of every model that uses the
  optical image. Its lead over `early_fusion_robust` grows with cloud cover: +0.010 (40%),
  +0.018 (60%), +0.044 (80%), +0.086 (100%). Against the training-seed spread, the 40% gap is
  about the size of the two models' own spreads (± 0.008 and ± 0.003) and should not be read
  as a difference; from 80% it is 2-12× either spread.
- **Against radar alone:** above `s1_only` up to 80% (0.681 vs 0.656, spreads ± 0.008 and
  ± 0.004). At 100% it is 0.644 vs 0.656 — 0.012 *below*, within the same order as the seed
  spread, so the two are level and fusion does not beat radar under total cloud.
- **Seed stability at 100%:** `fusion_robust` ± 0.007 vs `early_fusion_robust` ± 0.037, about
  five times tighter. Early fusion's average at heavy cloud depends strongly on which seed
  was trained.
- **Clear sky (0%):** `fusion_robust` 0.811 vs `early_fusion_robust` 0.817, a cost of 0.007;
  `s2_only_robust` is 0.821.
- **Cross-attention ablation (1 seed):** `fusion_concat_robust` keeps the two encoders but
  fuses by concatenation. It trails `fusion_robust` at every coverage above 0, by 0.013 (40%),
  0.024 (60%), 0.045 (80%) and 0.055 (100%) — at 80-100% several times `fusion_robust`'s seed
  spread. Indicative only: one training seed, and on Bolivia the ordering reverses (below).

**The thick (training) clouds leak ground signal.** `s2_only_robust` has no SAR input.
- Test: 0.791 at 100% thick cloud, 0.064 at 100% opaque.
- Bolivia: 0.783 at 100% thick, 0.051 at 100% opaque.

This confirms the leakage measured in [cloud_leakage.md](cloud_leakage.md): thick-cloud
results overstate optical robustness, so the opaque setting is the one to report.

**Bolivia (15 chips) still disagrees under heavy cloud — a small-sample caveat, not a result.**

| model | seeds | 0% | 40% | 60% | 80% | 100% |
|---|--:|--:|--:|--:|--:|--:|
| s1_only | 3 | 0.659 ± 0.043 | 0.659 ± 0.043 | 0.659 ± 0.043 | 0.659 ± 0.043 | 0.659 ± 0.043 |
| s2_only_robust | 1 | 0.789 | 0.559 ± 0.012 | 0.403 ± 0.035 | 0.280 ± 0.066 | 0.051 ± 0.016 |
| early_fusion_robust | 3 | 0.779 ± 0.015 | 0.692 ± 0.036 | 0.628 ± 0.060 | 0.598 ± 0.061 | 0.462 ± 0.155 |
| fusion_robust | 3 | 0.795 ± 0.007 | 0.720 ± 0.025 | 0.669 ± 0.027 | 0.640 ± 0.036 | 0.579 ± 0.034 |
| fusion_concat_robust | 1 | 0.802 | 0.733 ± 0.005 | 0.693 ± 0.014 | 0.679 ± 0.007 | 0.643 ± 0.010 |

- At 100% opaque `fusion_robust` (0.579 ± 0.034) is **below** `s1_only` (0.659 ± 0.043),
  though well above `early_fusion_robust` (0.462 ± 0.155 — the widest spread in the whole
  evaluation, i.e. one bad seed).
- The concat ablation scores *higher* than cross-attention on Bolivia at every coverage,
  reversing its test ordering. On 15 chips and one seed this is not separable from noise, but
  it does mean the ablation's conclusion is test-split evidence only.
- `fusion_robust` keeps the best clear-sky Bolivia score among the 3-seed models (0.795).

Bolivia answers the second research question (an unseen region) only weakly: 15 chips give
confidence intervals wider than the differences being compared.

**Real-cloud bins are supporting evidence only.** Labels are largely missing under real cloud
(see the audit above), and 2 test chips carry an unreliable s2cloudless mask
(`Paraguay_913449`, `Sri-Lanka_1049830`); the bins are reported with and without them, and
the cloudiest bin holds 11 chips (9 excluding the flagged two). In that 60-100% bin
`fusion_robust` leads the robust models (0.390 vs 0.366 early fusion, 0.254 optical-only,
0.267 radar-only), consistent with the synthetic sweep but on too few, partly unlabelled
chips to carry weight.

Figures: [iou_vs_coverage_test.png](figures/iou_vs_coverage_test.png),
[iou_vs_coverage_bolivia.png](figures/iou_vs_coverage_bolivia.png),
[real_cloud_bins.png](figures/real_cloud_bins.png),
[qualitative.png](figures/qualitative.png). In the qualitative figure, `India_747992` is the
clearest failure case: fragmented flooding in a vegetated, partly built-up landscape where
water is not cleanly radar-dark, and under opaque cloud every robust model — `fusion_robust`
included — recovers only scattered patches.

### Run plan (multi-seed and ablation) — completed
All runs below are trained, evaluated and included in the results above. `fusion_concat_robust`
was run at seed 0 only; seeds 1-2 were dropped for GPU budget, which is why the ablation is
reported as indicative.

| priority | config | seeds planned | seeds run |
|--:|---|---|---|
| 1 | `s1_only` | 1, 2 | 0, 1, 2 |
| 2 | `fusion_robust` | 1, 2 | 0, 1, 2 |
| 3 | `early_fusion_robust` | 1, 2 | 0, 1, 2 |
| 4 | `fusion_concat_robust` | 0, 1, 2 | 0 |

## 6. Stretch goals (only if time allows)
- [ ] _To be decided — check against CLAUDE.md scope first_

## 7. Report
- [ ] Figures and results tables
- [ ] Write-up answering the research question
- [ ] Code cleanup and reproducibility instructions in README
