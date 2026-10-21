#!/usr/bin/env bash
# Sourced by LOCAL launch/build scripts. Never installs anything or contacts a robot.
require_sonic_environment() {
 local env_file="$ROOT/.runtime/sonic/env.sh"
 if [[ ! -f "$env_file" ]]; then
  cat >&2 <<'EOF'
SONIC is not installed on this laptop yet.

LOCAL GPU laptop — run once from the G1-Zero-Shot repository:
  export CUDA_HOME=/usr/local/cuda
  bash scripts/laptop/setup.sh

Setup downloads dependencies/models, writes .runtime/sonic/env.sh and builds
this laptop's controller. env.sh is generated locally; do not copy it from another PC.

LOCAL CPU/GPU offline preview instead:
  bash scripts/laptop/setup_ui.sh
  bash scripts/laptop/demo.sh
EOF
  return 1
 fi
 if [[ ! -x "$ROOT/.venv_laptop/bin/python" ]]; then
  echo 'LOCAL: Python environment missing. Rerun bash scripts/laptop/setup.sh on this laptop.' >&2
  return 1
 fi
 source "$env_file"
}
