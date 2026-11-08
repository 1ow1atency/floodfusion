"""Tests for the shared Colab clone cell (notebooks/colab_clone_cell.py).

The cell is executed exactly as it appears in the notebooks, with REPO_URL / REPO_DIR
pointed at a local git repo and a fake google.colab.userdata. No network is used.
"""

from __future__ import annotations

import base64
import re
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL = (REPO / "notebooks" / "colab_clone_cell.py").read_text().rstrip("\n")
URL_LINE = "REPO_URL = 'https://github.com/1ow1atency/floodfusion.git'"
DIR_LINE = "REPO_DIR = Path('/content/floodfusion')"
TOKEN = "ghp_TESTtoken1234567890abcdef"
B64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()


def _clone_cells() -> dict[str, str]:
    cells = {}
    for nb_path in sorted((REPO / "notebooks").glob("*.ipynb")):
        for c in json.loads(nb_path.read_text())["cells"]:
            src = "".join(c["source"])
            if c["cell_type"] == "code" and "REPO_URL" in src:
                cells[nb_path.name] = src
    return cells


def test_notebook_clone_cells_match_canonical():
    cells = _clone_cells()
    assert {"01_compute_stats.ipynb", "02_cloud_cover.ipynb"} <= set(cells)
    for name, src in cells.items():
        assert src == CANONICAL, f"{name}: clone cell differs from notebooks/colab_clone_cell.py"
    assert URL_LINE in CANONICAL and DIR_LINE in CANONICAL


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    _git("-c", "init.defaultBranch=main", "init", cwd=src)
    (src / "a.txt").write_text("1")
    _git("add", ".", cwd=src)
    _git("commit", "-m", "first", cwd=src)
    return src


@pytest.fixture
def colab(monkeypatch):
    """Fake google.colab.userdata; returns a setter for the secret's behaviour."""
    state = {"get": lambda key: TOKEN}

    class SecretNotFoundError(Exception):
        pass

    userdata = types.ModuleType("google.colab.userdata")
    userdata.SecretNotFoundError = SecretNotFoundError
    userdata.get = lambda key: state["get"](key)
    colab_mod = types.ModuleType("google.colab")
    colab_mod.userdata = userdata
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)

    def missing(key):
        raise SecretNotFoundError(key)

    state["missing"] = missing
    return state


@pytest.fixture
def calls(monkeypatch):
    """Record argv and env of every subprocess.run call made by the cell."""
    seen = []
    real = subprocess.run

    def spy(args, *a, **kw):
        seen.append((list(args), dict(kw.get("env") or {})))
        return real(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", spy)
    return seen


def run_cell(url: Path | str, dest: Path) -> dict:
    src = CANONICAL.replace(URL_LINE, f"REPO_URL = {str(url)!r}").replace(DIR_LINE, f"REPO_DIR = Path({str(dest)!r})")
    assert src != CANONICAL
    ns: dict = {}
    exec(compile(src, "colab_clone_cell", "exec"), ns)
    return ns


def _no_secret_in(text: str) -> bool:
    return TOKEN not in text and B64 not in text


def test_clone_with_token_keeps_config_and_argv_clean(source, tmp_path, colab, calls, capsys):
    dest = tmp_path / "clone"
    run_cell(source, dest)
    out = capsys.readouterr().out
    assert "first" in out and _no_secret_in(out)
    assert _git("-C", str(dest), "remote", "get-url", "origin") == str(source)
    assert _no_secret_in((dest / ".git" / "config").read_text())
    assert "extraheader" not in _git("-C", str(dest), "config", "--list", "--local").lower()
    assert all(_no_secret_in(" ".join(argv)) for argv, _ in calls)
    clone_env = next(env for argv, env in calls if argv[:2] == ["git", "clone"])
    assert clone_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert clone_env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {B64}"
    assert clone_env["GIT_TERMINAL_PROMPT"] == "0"


def test_rerun_pulls_new_commits_and_cleans_remote(source, tmp_path, colab, capsys):
    dest = tmp_path / "clone"
    run_cell(source, dest)
    (source / "a.txt").write_text("2")
    _git("commit", "-am", "second", cwd=source)
    # Simulate an old clone whose remote URL had a token embedded (file:// so a regression
    # fails locally instead of contacting github.com).
    _git("-C", str(dest), "remote", "set-url", "origin", f"file:///nonexistent/x-access-token:{TOKEN}@github.com/x/y.git")
    capsys.readouterr()
    run_cell(source, dest)
    out = capsys.readouterr().out
    assert "second" in out and _no_secret_in(out)
    assert (dest / "a.txt").read_text() == "2"
    assert _git("-C", str(dest), "remote", "get-url", "origin") == str(source)
    assert _no_secret_in((dest / ".git" / "config").read_text())


def test_token_is_seen_by_git_but_masked(source, tmp_path, colab, capsys):
    ns = run_cell(source, tmp_path / "clone")
    # git reads the env-provided header (proves the mechanism works) and the output is masked.
    assert ns["_git"]("config", "--get", "http.https://github.com/.extraheader", token=TOKEN) == \
        "Authorization: Basic ***"
    with pytest.raises(RuntimeError) as e:
        ns["_git"]("-C", str(tmp_path / "clone"), "checkout", TOKEN, token=TOKEN)
    assert "git checkout failed" in str(e.value) and "***" in str(e.value) and _no_secret_in(str(e.value))


def test_missing_secret_falls_back_to_plain_clone(source, tmp_path, colab, calls, capsys):
    colab["get"] = colab["missing"]
    run_cell(source, tmp_path / "clone")
    out = capsys.readouterr().out
    assert "SecretNotFoundError" in out and "GITHUB_TOKEN" in out and "Notebook access" in out
    assert (tmp_path / "clone" / "a.txt").exists()
    assert all("GIT_CONFIG_COUNT" not in env for _, env in calls)


def test_failed_clone_without_token_hints_at_secret(tmp_path, colab, capsys):
    colab["get"] = colab["missing"]
    with pytest.raises(RuntimeError, match="add the GITHUB_TOKEN secret"):
        run_cell(tmp_path / "does-not-exist", tmp_path / "clone")


# --- notebook 03 (training) and general notebook hygiene ------------------------------

def _code_cells(nb_name: str) -> list[str]:
    nb = json.loads((REPO / "notebooks" / nb_name).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_all_notebook_code_cells_compile():
    import re

    for nb_path in sorted((REPO / "notebooks").glob("*.ipynb")):
        for i, src in enumerate(_code_cells(nb_path.name)):
            compile(re.sub(r"^(\s*)!.*$", r"\1pass", src, flags=re.M), f"{nb_path.name}[{i}]", "exec")


def test_train_notebook_contract():
    nb = json.loads((REPO / "notebooks" / "03_train.ipynb").read_text())
    top = "".join(nb["cells"][0]["source"])
    assert ("Only change the settings cell. Re-running after a disconnect resumes automatically. "
            "Never change hyperparameters based on test or bolivia results.") in top
    code = _code_cells("03_train.ipynb")
    settings = code[0]
    ns: dict = {}
    exec(settings, ns)
    assert set(k for k in ns if not k.startswith("__")) == {"CONFIG", "SEED", "USE_WANDB"}
    assert (REPO / ns["CONFIG"]).is_file()
    listed = set(re.findall(r"\w+", settings))
    assert {c.stem for c in (REPO / "configs").glob("*.yaml")} <= listed  # every config is offered
    joined = "\n".join(code)
    # torch and s2cloudless must never meet: nothing from the clouds extra.
    assert "clouds" not in joined and "s2cloudless" not in joined
    assert '--resume' in joined and '--no-wandb' in joined and "/content/data/sen1floods11" in joined
    assert all(s in joined for s in ("'valid'", "'test'", "'bolivia'"))


def test_evaluate_notebook_contract():
    nb = json.loads((REPO / "notebooks" / "04_evaluate.ipynb").read_text())
    top = "".join(nb["cells"][0]["source"])
    assert "Final reporting only" in top and "select models" in top
    code = _code_cells("04_evaluate.ipynb")
    ns: dict = {}
    exec(code[0], ns)
    assert set(k for k in ns if not k.startswith("__")) == {"RUNS_DIR", "RESULTS_DIR", "CACHE_DIR"}
    assert ns["RUNS_DIR"] == "/content/drive/MyDrive/floodfusion/runs"
    assert ns["RESULTS_DIR"] == "/content/drive/MyDrive/floodfusion/results"
    assert ns["CACHE_DIR"].startswith("/content/") and "drive" not in ns["CACHE_DIR"]  # local disk, not Drive quota
    joined = "\n".join(code)
    assert "scripts/robustness_eval.py" in joined and "/content/data/sen1floods11" in joined
    assert ".[clouds]" not in joined and "s2cloudless" not in joined  # base install only
    copy_03 = next(c for c in _code_cells("03_train.ipynb") if "LOCAL_DATA = Path(" in c)
    copy_04 = next(c for c in code if "LOCAL_DATA = Path(" in c)
    assert copy_04 == copy_03.replace("RUNS_DIR = Path('/content/drive/MyDrive/floodfusion/runs')\n", "")


def test_cloud_audit_notebook_contract():
    nb = json.loads((REPO / "notebooks" / "05_cloud_audit.ipynb").read_text())
    code = _code_cells("05_cloud_audit.ipynb")
    ns: dict = {}
    exec(code[0], ns)
    assert ns["OUT_DIR"] == "/content/drive/MyDrive/floodfusion/results/cloud_audit"
    assert ns["DATA_ROOT"] == "/content/drive/MyDrive/floodfusion/data/sen1floods11"
    joined = "\n".join(code)
    assert '[clouds]' in joined and "scripts/audit_clouds.py" in joined
    assert "--allow-missing" not in joined  # full data: missing files are an error
    assert "import torch" not in joined and nb["metadata"].get("accelerator") is None  # CPU runtime
