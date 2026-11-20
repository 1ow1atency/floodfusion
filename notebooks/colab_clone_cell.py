# Clone or update the repo in Colab. Shared cell: paste verbatim into every notebook
# (tests/test_notebooks.py checks the copies match notebooks/colab_clone_cell.py).
# Uses the Colab secret GITHUB_TOKEN if available (see docs/colab_setup.md). The token
# reaches git only through environment variables: never on the command line, never in
# .git/config (the remote URL stays token-free), and it is masked in any output.
import base64, os, subprocess
from pathlib import Path

REPO_URL = 'https://github.com/1ow1atency/floodfusion.git'
REPO_DIR = Path('/content/floodfusion')


def _github_token():
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
    except Exception as e:  # secret missing, notebook access off, or not in Colab
        print(f'GITHUB_TOKEN secret not available ({type(e).__name__}); trying a plain public clone.\n'
              'To add it: key icon in the left sidebar -> "Add new secret" -> name GITHUB_TOKEN, value = a\n'
              'GitHub token with read access to this repo -> switch "Notebook access" on -> re-run this cell.')
        return None
    return token.strip() or None


def _git(*args, token=None):
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}  # fail instead of waiting for a password
    secrets = []
    if token:
        b64 = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0=f'Authorization: Basic {b64}')
        secrets = [token, b64]
    r = subprocess.run(['git', *args], env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    for s in secrets:
        out = out.replace(s, '***')
    if r.returncode != 0:
        hint = '' if token else '\n(No token used. If the repo is not publicly visible, add the GITHUB_TOKEN secret.)'
        raise RuntimeError(f'git {args[0] if args[0] != "-C" else args[2]} failed:\n{out}{hint}')
    return out


def _clone_or_pull():
    version = tuple(int(x) for x in _git('--version').split()[2].split('.')[:2])
    if version < (2, 31):  # GIT_CONFIG_COUNT needs git >= 2.31
        raise RuntimeError(f'git >= 2.31 needed, found {version}')
    token = _github_token()
    if (REPO_DIR / '.git').exists():
        _git('-C', str(REPO_DIR), 'remote', 'set-url', 'origin', REPO_URL)  # keep the remote token-free
        print(_git('-C', str(REPO_DIR), 'pull', '--ff-only', token=token))
    else:
        print(_git('clone', REPO_URL, str(REPO_DIR), token=token))
    print('at', _git('-C', str(REPO_DIR), 'log', '--oneline', '-1'))


_clone_or_pull()
