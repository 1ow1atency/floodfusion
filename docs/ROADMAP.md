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

## 4. Fusion model
- [ ] Two-branch encoder (SAR branch, optical branch)
- [ ] Cross-attention fusion
- [ ] Train and compare against baselines on valid / test

## 5. Robustness evaluation
- [ ] Synthetic cloud augmentation for S2
- [ ] Modality dropout during training
- [ ] IoU vs cloud-cover curves for all models
- [ ] Evaluation on held-out Bolivia split (evaluation only, never tuned on)

## 6. Stretch goals (only if time allows)
- [ ] _To be decided — check against CLAUDE.md scope first_

## 7. Report
- [ ] Figures and results tables
- [ ] Write-up answering the research question
- [ ] Code cleanup and reproducibility instructions in README
