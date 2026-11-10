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
  removed (`notebooks/04_evaluate.ipynb`). Run on the training-seed-0 models (results below);
  repeat once the runs in the run plan are done.
- [ ] Evaluation on held-out Bolivia split (evaluation only, never tuned on). Seed 0 done (below).

### Robustness results (training seed 0)
Source: `docs/results/robustness.csv` and `docs/results/summary.md` (notebook 04 on Colab). All
seven models are training seed 0, each its `best.pt` selected on valid. The numbers are water
IoU (label -1 ignored), averaged over 3 cloud seeds per coverage. Test has 90 chips and Bolivia 15.

**Main result: test split, opaque synthetic clouds** (no ground signal under the cloud):

| model | 0% | 20% | 40% | 60% | 80% | 100% | S2 removed |
|---|--:|--:|--:|--:|--:|--:|--:|
| s1_only | 0.657 | 0.657 | 0.657 | 0.657 | 0.657 | 0.657 | 0.657 |
| s2_only | 0.828 | 0.576 | 0.400 | 0.272 | 0.141 | 0.002 | n/a |
| s2_only_robust | 0.821 | 0.682 | 0.526 | 0.383 | 0.215 | 0.064 | n/a |
| early_fusion | 0.820 | 0.707 | 0.606 | 0.510 | 0.445 | 0.371 | 0.000 |
| early_fusion_robust | 0.819 | 0.777 | 0.730 | 0.688 | 0.616 | 0.530 | 0.635 |
| fusion | 0.818 | 0.738 | 0.664 | 0.607 | 0.566 | 0.546 | 0.000 |
| fusion_robust | 0.813 | **0.783** | **0.750** | **0.717** | **0.687** | 0.650 | 0.639 |

- **20–80% coverage:** `fusion_robust` has the highest IoU of all seven models at every
  coverage, and it stays above SAR alone (`s1_only` 0.657) up to 80% (0.687).
- **100% coverage:** it falls to 0.650, level with `s1_only` (0.657), not above it.
  - It is still the best of the models that use S2.
  - At the same 100% coverage, `early_fusion_robust` reaches 0.530 and `s2_only_robust` 0.064.
- **Clear sky (0%):** `fusion_robust` costs little: 0.813 vs 0.828 for `s2_only`.

**The thick (training) clouds leak ground signal.** `s2_only_robust` has no SAR input.
- Test: it keeps 0.791 at 100% thick cloud, but drops to 0.064 at 100% opaque.
- Bolivia: 0.783 at 100% thick vs 0.051 at 100% opaque.

This confirms the leakage measured in [cloud_leakage.md](cloud_leakage.md): thick-cloud results
overstate optical robustness, so the opaque setting is the one to report.

**Bolivia (15 chips) disagrees at 100% opaque. Treat it as a small-sample caveat.**
- At 100% opaque, `fusion` (no robustness training) scores 0.687, above `fusion_robust`
  (0.618) and `s1_only` (0.625).
- From 20% to 80% opaque, `fusion_robust` is again the best model on Bolivia (0.767, 0.741,
  0.696, 0.677).
- `fusion`'s Bolivia curve is not monotonic: 0.625 at 80%, then 0.687 at 100%.
- With S2 removed, `fusion` scores 0.000 on Bolivia.

So its 100% opaque score on 15 chips is not evidence of real robustness.

**Caveat: single training seed.** Every model is one training run (seed 0). The ± values in
`summary.md` are std over the 3 cloud seeds only (for example ± 0.0009 for `fusion_robust` at
100% opaque on test), so they do not measure training variance. Gaps such as 0.650 vs 0.657
cannot be called real until the runs below exist.

### Run plan (multi-seed and ablation)
Run each with notebook 03 (set `CONFIG` and `SEED`). Notebook 04 then groups the runs by
config name and reports mean ± std over training seeds, keeping the per-run tables.
`configs/fusion_concat_robust.yaml` is `fusion_robust.yaml` with `fusion: concat`, a test
checks that this is the only difference. It tests whether cross-attention or just the separate
encoders makes the difference.

| priority | config | seeds | rough T4 time per run | total |
|--:|---|---|--:|--:|
| 1 | `s1_only` | 1, 2 | 25–40 min | 1–1.5 h |
| 2 | `fusion_robust` | 1, 2 | 45–100 min | 1.5–3.5 h |
| 3 | `early_fusion_robust` | 1, 2 | 35–60 min | 1–2 h |
| 4 | `fusion_concat_robust` | 0, 1, 2 | 45–90 min | 2.5–4.5 h |

Total about 6–11.5 h of T4 time, plus a few minutes per session for the data copy.
- **Times are estimates, not measured.**
  - Per-epoch guesses: 10–15 s for the S2-free U-Net and 15–25 s for the early-fusion U-Net.
  - FusionNet: about 1.5–2× early fusion.
  - Plus about 5–10 s per epoch for `val_iou_cloud50`.
  - Replace them with the `epoch_time_s` column of the seed-0 runs' `metrics.csv` on Drive.
- **Priority order:**
  - `s1_only` comes first: it is the reference that `fusion_robust` meets at 100% opaque
    cloud, and it is the cheapest run.
  - The concat ablation comes last: it answers a follow-up question, not the main one.
- **Drive space:** each finished FusionNet run keeps `last.pt` (~590 MB) and `best.pt`
  (~196 MB); a U-Net run keeps ~294 MB + ~98 MB.
  - The plan adds 5 FusionNet runs and 4 U-Net runs, about 5.5 GB.
  - Only `best.pt` and `eval_*/` are needed for evaluation. A finished run's `last.pt` can be
    deleted once you are sure it will not be resumed. Empty the Bin after each run.
- **Notebook 04:** re-run it once at the end over all runs. Its runtime grows with the number
  of runs (16 instead of 7).

## 6. Stretch goals (only if time allows)
- [ ] _To be decided — check against CLAUDE.md scope first_

## 7. Report
- [ ] Figures and results tables
- [ ] Write-up answering the research question
- [ ] Code cleanup and reproducibility instructions in README
