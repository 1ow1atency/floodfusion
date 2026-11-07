# floodfusion

Cloud-robust flood mapping by fusing Sentinel-1 SAR and Sentinel-2 optical imagery with deep learning (Sen1Floods11).

> **Interim README.** This project is in progress. The results below are preliminary.

Optical satellite images (Sentinel-2) show flood water clearly, but they are often blocked by
clouds, and clouds are especially common during floods. Radar (Sentinel-1 SAR) sees through clouds,
but water is harder to tell apart from other surfaces in radar. This project fuses both
modalities in deep segmentation models and asks: **does fusing SAR with optical imagery keep
flood segmentation accurate (1) when the optical image is covered by clouds, and (2) when the
model is tested on a region it never saw in training?** The second question uses the held-out
Bolivia split of Sen1Floods11, which is used for evaluation only.

## Status

- **Done:** data pipeline, normalisation statistics (train split only) and cloud cover of
  every chip.
- **Done:** U-Net baselines (SAR-only, optical-only, early fusion), trained.
- **Done:** FusionNet, trained without cloud or dropout training. It has two encoders (one per
  modality) with cross-attention fusion.
- **In progress:** robustness training (synthetic clouds on Sentinel-2, modality dropout).
- **In progress:** evaluation of water IoU against cloud cover, and on Bolivia.

## Preliminary results

**Preliminary: single seed (0), real data only.** These numbers are not final: there is
no repeat over seeds, no synthetic-cloud evaluation, and no robustness-trained models yet.

- `best.pt` was selected on the valid split and evaluated on full 512×512 chips.
- Sentinel-2 is used as acquired, with no synthetic clouds.
- Metric: IoU of the water class, with no-data pixels (label `-1`) ignored.
- Source: [docs/ROADMAP.md](docs/ROADMAP.md).

| model        | valid IoU | test IoU | bolivia IoU | test precision | test recall |
|--------------|----------:|---------:|------------:|---------------:|------------:|
| s1_only      |    0.6612 |   0.6567 |      0.6253 |         0.8168 |      0.7702 |
| s2_only      |    0.8297 |   0.8276 |      0.7908 |         0.9269 |      0.8854 |
| early_fusion |    0.8379 |   0.8202 |      0.7922 |         0.9339 |      0.8707 |
| fusion       |    0.8375 |   0.8181 |      0.7971 |         0.9229 |      0.8781 |

`fusion` is FusionNet without cloud or dropout training (`best.pt` from epoch 74). On this
real, mostly clear data it is within single-seed noise of `s2_only` and `early_fusion`, as
expected without cloud or dropout training. Early fusion does not beat optical-only either. The hypothesis
still to be tested is that early fusion relies on Sentinel-2 and ignores Sentinel-1, so it
should fail under clouds.

## Repository layout

```
src/floodfusion/   the package (all logic lives here)
  data.py, chips.py      Sen1Floods11 dataset; missing-value, clipping and normalisation rules
  stats.py               normalisation statistics (train split only)
  clouds.py              real cloud cover per chip (s2cloudless, optional `clouds` extra)
  clouds_synth.py        synthetic clouds for Sentinel-2 (robustness training)
  transforms.py          crops/flips/rotations, synthetic clouds, modality dropout
  models.py              U-Net baselines and FusionNet
  losses.py, metrics.py  cross-entropy + Dice loss; water IoU from pooled confusion counts
  training.py, evaluation.py   training loop (checkpoints, resume, W&B) and evaluation
  viz.py                 figures
scripts/           thin command-line entry points (train.py, evaluate.py, compute_stats.py, ...)
configs/           one YAML per experiment, plus band_stats.json (normalisation statistics)
notebooks/         thin Colab launchers: 00 download data, 01 stats, 02 cloud cover, 03 train
tests/             pytest suite on tiny synthetic GeoTIFFs (never the real dataset)
docs/              ROADMAP.md, label and cloud-cover tables, figures
```

Data, checkpoints and W&B run folders are not in git. They live on Google Drive, and their
paths come from the configs or the command line.

## Run locally (CPU)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests are fast on a CPU and need no data.

A **smoke run** checks that the full training path works on a few local chips. It needs a
local sample of the dataset under `data/sen1floods11/`:

```bash
python scripts/train.py --config configs/early_fusion.yaml --root data/sen1floods11 \
    --out outputs/runs --smoke --no-wandb
```

The smoke run trains for one epoch of 2 batches and "validates" on 2 train chips. It prints
a warning because its numbers are not a result. Any config in `configs/` works.

## Train on Colab

Training runs on a Colab GPU (T4) with `notebooks/03_train.ipynb`.

1. Do the one-time setup in [Colab setup](#colab-setup) below (GitHub token secret).
2. Run `00_download_data.ipynb` once to put Sen1Floods11 on Google Drive.
3. Run `01_compute_stats.ipynb` once to compute the normalisation statistics.
4. In notebook 03, change only the settings cell (config and seed). It trains, resumes
   automatically after a disconnect, and evaluates `best.pt` on valid, test and bolivia.

Never change hyperparameters based on test or bolivia results.

## Dataset and citation

This project uses the hand-labelled subset of **Sen1Floods11** (v1.1):

> Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). Sen1Floods11: a
> georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.
> In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
> Workshops (CVPRW)*, pp. 210–211.

```bibtex
@inproceedings{bonafilia2020sen1floods11,
  title     = {Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1},
  author    = {Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {210--211},
  year      = {2020}
}
```

## Colab setup

Notebooks in `notebooks/` are thin Colab launchers. Each one clones (or pulls) this repo into
`/content/floodfusion` with the shared cell in `notebooks/colab_clone_cell.py`.

### GitHub token (needed while the repo is not publicly visible)

1. On GitHub: Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token. Repository access: *Only select repositories* → `floodfusion`;
   Permissions: *Contents: Read-only*. Set an expiry date.
2. In Colab: click the **key icon** (Secrets) in the left sidebar → **Add new secret** →
   Name `GITHUB_TOKEN`, Value = the token.
3. Switch **Notebook access** on for that secret. Colab asks again for each new notebook.
4. Re-run the clone cell.

The token is passed to git only through environment variables. It is never printed (it is
masked as `***` in error output), never put on the command line and never written to
`.git/config`. If the secret is missing, the cell prints how to add it and tries a plain
public clone instead.

### Opening a notebook when the GitHub link does not work

1. Locally: `git pull` so your copy of the repo is up to date.
2. In Colab: **File → Upload notebook** → pick e.g. `notebooks/02_cloud_cover.ipynb` from your
   local repo.

The uploaded notebook is a copy saved in your Drive (`Colab Notebooks/`). It does not update
when the repo changes, so upload it again after notebook changes. The code it runs is always
the latest, because the clone cell pulls the repo.
