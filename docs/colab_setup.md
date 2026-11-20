# Running on Colab

Training and the robustness evaluation run on a Colab GPU; the cloud-mask audit runs on a
Colab CPU runtime. Notebooks in `notebooks/` are thin launchers: each one clones (or pulls)
this repo into `/content/floodfusion` with the shared cell in `notebooks/colab_clone_cell.py`,
and all the logic lives in the `floodfusion` package.

Local, data-free setup (tests and a smoke run) is in the
[README](../README.md#run-locally-cpu) instead.

## One-time setup

1. Add the `GITHUB_TOKEN` secret if your fork is private (below). The repo is public, so for
   this repo the plain clone works and no token is needed.
2. Run `00_download_data.ipynb` once to put Sen1Floods11 on Google Drive.
3. Run `01_compute_stats.ipynb` once to compute the normalisation statistics (train split
   only), which every other notebook reuses.

### GitHub token (only needed if you make your fork private)

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

## Train

Training runs on a Colab GPU (T4) with `notebooks/03_train.ipynb`. Change only the settings
cell (config and seed). It trains, resumes automatically after a disconnect, and evaluates
`best.pt` on valid, test and bolivia.

**Never change hyperparameters based on test or bolivia results.** Model selection and tuning
use the validation split only; Bolivia is evaluation-only.

## Robustness evaluation

After all runs are trained and evaluated, `notebooks/04_evaluate.ipynb` runs the robustness
evaluation (`scripts/robustness_eval.py`) over every run folder and writes `robustness.csv`
(one row per run), `robustness_by_model.csv`, `summary.md` and figures to
`floodfusion/results/` on Drive. Runs of one config with different seeds
(`fusion_robust_seed0`, `_seed1`, ...) are grouped: tables and figures show mean ± std over
training seeds, with the per-run tables kept. It sweeps cloud coverage for thick clouds (as
trained) and opaque clouds (the stricter test), and also evaluates with S2 removed. Expect
about 15-25 min on a T4. This is final reporting only.

## Cloud-mask audit

`notebooks/05_cloud_audit.ipynb` (CPU runtime) audits the s2cloudless masks behind the
real-cloud bins (`scripts/audit_clouds.py`); this has been run on the full data, with the
findings in [cloud_audit.md](cloud_audit.md). `scripts/flag_cloud_masks.py` turns its
`chip_audit.csv` into the `mask_unreliable` flags in `docs/cloud_cover.csv` and the 0.6
sensitivity file `docs/cloud_cover_t06.csv`.
