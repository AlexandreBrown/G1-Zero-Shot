#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/Documents/g1_zero/GR00T-WholeBodyControl"
source .venv_sim/bin/activate
exec python -u "$SCRIPT_DIR/sonic_headless.py"
