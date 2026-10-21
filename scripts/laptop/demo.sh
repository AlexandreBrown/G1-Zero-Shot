#!/usr/bin/env bash
# LOCAL offline UI demo. No DDS, GPU, model download or robot process.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for candidate in "$ROOT/.venv_ui/bin/python" "$ROOT/.venv_laptop/bin/python" "$(dirname "$ROOT")/GR00T-WholeBodyControl/.venv_sim/bin/python" python3; do
 if command -v "$candidate" >/dev/null && "$candidate" -c 'import tkinter' 2>/dev/null; then
  exec "$candidate" -m sonic_laptop.app --demo
 fi
done
echo 'LOCAL: install Python Tk support (Ubuntu: sudo apt-get install python3-tk), then rerun.' >&2
exit 1
