#!/usr/bin/env bash
# LOCAL: offline conversion only, no robot commands or GPU inference.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for candidate in "$ROOT/.venv_laptop/bin/python" "$(dirname "$ROOT")/GR00T-WholeBodyControl/.venv_sim/bin/python" python3; do
 if command -v "$candidate" >/dev/null && "$candidate" -c 'import numpy, scipy' 2>/dev/null; then
  exec "$candidate" -m sonic_laptop.import_kimodo "$@"
 fi
done
echo 'LOCAL: run scripts/laptop/setup.sh or install numpy and scipy in your Python environment.' >&2
exit 1
