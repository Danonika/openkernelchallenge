#!/usr/bin/env bash
# User-invoked environment preparation only. No root/system changes or driver install.
set -euo pipefail
: "${PYTHON313:?Set PYTHON313 to an existing Python3.13.15 executable}"
: "${VENV_DIR:?Set VENV_DIR to a NEW absolute venv directory}"
[[ "$VENV_DIR" = /* && ! -e "$VENV_DIR" ]] || { echo 'VENV_DIR must be an absent absolute path; nothing deleted' >&2; exit 2; }
"$PYTHON313" -c 'import sys; assert sys.version_info[:3]==(3,13,15), "Exact Python3.13.15 required; no fallback"'
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
"$PYTHON313" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements-kaggle.txt"
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" "$ROOT/check_kaggle_environment.py" --out "$VENV_DIR/kaggle-compatibility.json"
"$VENV_DIR/bin/python" -m pip freeze > "$VENV_DIR/installed-freeze.txt"
echo 'Pinned package preparation complete; GPU, driver, CPU affinity, task and baseline gates still required.'
