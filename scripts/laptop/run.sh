#!/usr/bin/env bash
# LOCAL GPU laptop. UI opens disarmed; only the ARM button launches motor control.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifacts/laptop/env.sh"
cd "$ROOT"
exec python -m sonic_laptop.app "$@"
