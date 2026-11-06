# floodfusion — cloud-robust flood mapping (final-year project)

Fuse Sentinel-1 SAR and Sentinel-2 optical imagery with deep learning for flood segmentation.

**Research question:** Does fusing SAR with optical imagery keep flood segmentation accurate
(1) when the optical image is covered by clouds, and (2) when the model is tested on a region
it never saw in training?

## Scope
- U-Net baselines: SAR-only, optical-only, early fusion (channel concatenation).
- Two-branch encoder (one per modality) with cross-attention fusion.
- Robustness training: modality dropout and synthetic cloud augmentation on S2.
- Evaluation: water IoU vs cloud cover, and on the held-out Bolivia split.

## Out of scope unless I explicitly ask
Self-supervised pretraining, change detection, extra datasets, big hyperparameter sweeps.
If a request or idea drifts toward these, **warn me before doing it** and tie it back to the
research question.

## Dataset: Sen1Floods11 v1.1, hand-labeled subset
- Root: `data/sen1floods11/{S1Hand,S2Hand,LabelHand,splits}`
- Files: `<chip_id>_S1Hand.tif`, `<chip_id>_S2Hand.tif`, `<chip_id>_LabelHand.tif`
- S1: 2 bands (VV, VH) in dB; **may contain NaN** — handle explicitly, never let NaN reach the model.
- S2: 13 bands (B1–B12 incl. B8A), Level-1C (top-of-atmosphere).
- Labels: `1` water, `0` not water, `-1` no data. **`-1` is always ignored in loss and metrics.**
- Splits (in `splits/`): `flood_train_data.csv`, `flood_valid_data.csv`,
  `flood_test_data.csv`, `flood_bolivia_data.csv`.

### Data rules (non-negotiable)
- **Never train, tune, early-stop or select models on Bolivia.** It is evaluation-only.
- Model selection and tuning use the validation split only.
- Normalisation statistics are computed from the **train split only**, saved to a file, and
  reused for every other split.

## Where code runs
- Written and tested locally on CPU (Mac). Keep everything runnable on CPU for small inputs.
- Training on Google Colab GPU: clone the GitHub repo, install, run scripts.
- Data and checkpoints live on Google Drive; paths come from config/CLI, never hard-coded.
- Experiments logged to Weights & Biases. No API keys or secrets in the repo.
- Data, checkpoints and W&B run dirs are git-ignored.

## Conventions
- Package `floodfusion` in `src/` (src layout); thin scripts in `scripts/` that only parse
  args/config and call package code.
- YAML configs in `configs/`; every experiment is reproducible from its config.
- Tests with pytest in `tests/`, using **tiny synthetic GeoTIFFs generated in the test**
  (e.g. via `tmp_path`). Tests must **never** read the real dataset.
- Notebooks are thin Colab launchers only — no logic that isn't in `src/`.
- Fixed seeds (Python, NumPy, PyTorch) set from config for every run.
- **Primary metric: IoU of the water class** (ignoring `-1`). Other metrics are secondary.

## How to work with me
- Run `pytest` after code changes; keep tests fast on CPU.
- Prefer small, reviewable changes; explain design choices that affect the experiments.
- When results are reported, state the split, cloud-cover setting and seed.
