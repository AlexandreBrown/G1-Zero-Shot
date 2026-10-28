#!/usr/bin/env bash
# Installs only; never starts a robot controller or sends API requests.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${SONIC_UPSTREAM:-$(dirname "$ROOT")/GR00T-WholeBodyControl}"
if [[ $# -gt 1 || ($# -eq 1 && "$1" != --gpu) ]]; then
    echo 'Usage: bash gpt_astra/setup.sh [--gpu]' >&2
    exit 2
fi
if ! git lfs version >/dev/null 2>&1; then
    echo 'Install git-lfs first (Ubuntu: sudo apt install git-lfs), then rerun.' >&2
    exit 1
fi
source "$ROOT/scripts/laptop/source.sh"
ensure_sonic_source
python3 -m venv "$ROOT/gpt_astra/.venv"
"$ROOT/gpt_astra/.venv/bin/python" -m pip install -r "$ROOT/gpt_astra/requirements.txt"
if [[ ${1:-} == --gpu ]]; then
    export SONIC_UPSTREAM="$UPSTREAM"
    bash "$ROOT/scripts/laptop/setup.sh"
fi
echo 'Ready. See gpt_astra/README.md. OPENAI_API_KEY is read at run time.'
