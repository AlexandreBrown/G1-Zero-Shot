#!/usr/bin/env bash
# LOCAL GPU laptop. UI opens disarmed; only the ARM button launches motor control.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/laptop/runtime.sh"
require_sonic_environment
if [[ ! -x "$ROOT/.runtime/sonic/deploy/target/release/g1_deploy_onnx_ref" || ! -f "$ROOT/.runtime/sonic/deploy/binary-manifest.json" ]]; then
 echo 'LOCAL: SONIC setup/build is incomplete. Run bash scripts/laptop/build.sh; see setup.sh if dependencies are missing.' >&2
 exit 1
fi
cd "$ROOT"
exec python -m sonic_laptop.app "$@"
