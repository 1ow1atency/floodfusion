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

| model        | valid IoU | test IoU | bolivia IoU | test precision | test recall |
|--------------|----------:|---------:|------------:|---------------:|------------:|
| s1_only      |    0.6612 |   0.6567 |      0.6253 |         0.8168 |      0.7702 |
| s2_only      |    0.8297 |   0.8276 |      0.7908 |         0.9269 |      0.8854 |
| early_fusion |    0.8379 |   0.8202 |      0.7922 |         0.9339 |      0.8707 |
| fusion       |    0.8375 |   0.8181 |      0.7971 |         0.9229 |      0.8781 |

Early fusion does not beat optical-only on real (mostly clear) data. Hypothesis to test in
Stage 5: it relies on S2 and ignores S1, so it should collapse under clouds.

`fusion` = FusionNet (`configs/fusion.yaml`, cross-attention, no cloud/dropout training),
`best.pt` from epoch 74. On real (mostly clear) data it is within single-seed noise of
s2_only and early_fusion, as expected without cloud/dropout training.

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
- [ ] TODO: Audit s2cloudless masks before using real-cloud bins: several Paraguay train chips are
  marked 100% cloud but show clear ground or only haze (e.g. Paraguay_215904, Paraguay_36146).
  Check probability maps, a stricter threshold, and spot-check chips per bin.
- [ ] IoU vs cloud-cover curves for all models
- [ ] Evaluation on held-out Bolivia split (evaluation only, never tuned on)

## 6. Stretch goals (only if time allows)
- [ ] _To be decided — check against CLAUDE.md scope first_

## 7. Report
- [ ] Figures and results tables
- [ ] Write-up answering the research question
- [ ] Code cleanup and reproducibility instructions in README
