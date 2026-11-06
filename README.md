# floodfusion

Cloud-robust flood mapping by fusing Sentinel-1 SAR and Sentinel-2 optical imagery with deep learning (Sen1Floods11).

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
