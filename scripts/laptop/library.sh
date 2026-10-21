#!/usr/bin/env bash
# LOCAL offline library workspace. Cannot start a controller or arm hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for candidate in "$ROOT/.venv_ui/bin/python" "$ROOT/.venv_laptop/bin/python" "$(dirname "$ROOT")/GR00T-WholeBodyControl/.venv_sim/bin/python" python3; do
 if command -v "$candidate" >/dev/null && "$candidate" -c 'import tkinter,numpy,scipy' 2>/dev/null; then
  exec "$candidate" -m sonic_laptop.app --library
 fi
done
echo 'LOCAL: use Python with tkinter, numpy and scipy (or the laptop setup environment).' >&2
exit 1
