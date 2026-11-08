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

## 5. Robustness evaluation
- [ ] Synthetic cloud augmentation for S2
- [ ] Modality dropout during training
- [ ] Audit s2cloudless masks before using real-cloud bins. Local-sample audit done
  ([cloud_audit.md](cloud_audit.md)); full-data run still TODO (`notebooks/05_cloud_audit.ipynb`).
  - **Finding:** threshold 0.4 over-flags haze where the ground is visible. It does not
    over-flag clear gaps (only a thin dilation rim).
  - **Recommendation, provisional:** threshold 0.6. It agrees best with the annotators' -1 on
    train chips (F1 80% vs 72%).
  - **Labels under cloud:** cloudy pixels are mostly labelled -1, and almost no water is
    labelled under cloud. The real-cloud bins therefore score the visible part of each chip,
    not water under cloud.
  - [ ] Run notebook 05 on the full data. Confirm 0.6 (F1 vs -1 on all train chips;
    Paraguay_215904 / Paraguay_36146; test and Bolivia bin sizes), then decide whether to
    regenerate `cloud_cover.csv`.
- [ ] IoU vs cloud-cover curves for all models: thick (as trained) and opaque clouds, plus S2
  removed (`notebooks/04_evaluate.ipynb`; code ready, not yet run on the trained models)
- [ ] Evaluation on held-out Bolivia split (evaluation only, never tuned on)

## 6. Stretch goals (only if time allows)
- [ ] _To be decided — check against CLAUDE.md scope first_

## 7. Report
- [ ] Figures and results tables
- [ ] Write-up answering the research question
- [ ] Code cleanup and reproducibility instructions in README
