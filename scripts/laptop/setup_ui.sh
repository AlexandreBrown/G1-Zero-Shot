#!/usr/bin/env bash
# LOCAL: offline UI/import dependencies only; no models, GPU setup or robot commands.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
sudo apt-get install -y python3-venv python3-tk fonts-dejavu-core git git-lfs
# Ubuntu's Tk uses antialiased system fonts. Some bundled Python Tk builds do not.
/usr/bin/python3 -m venv "$ROOT/.venv_ui"
"$ROOT/.venv_ui/bin/python" -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2'
UPSTREAM="${SONIC_UPSTREAM:-$(dirname "$ROOT")/GR00T-WholeBodyControl}"
source "$ROOT/scripts/laptop/source.sh"
ensure_sonic_source
echo 'LOCAL: run bash scripts/laptop/library.sh (offline) or demo.sh.'
