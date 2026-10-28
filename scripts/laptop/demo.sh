#!/usr/bin/env bash
# LOCAL demo. After GPU setup: MuJoCo simulator + the real SONIC panel (run.sh --sim, interface lo)
# with the motions/sim_demo clips; you operate the simulated robot. Nothing is sent to hardware.
# Without GPU setup, or with --offline: UI-only preview, no DDS, GPU, model download or robot process.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

offline_preview() {
 for candidate in "$ROOT/.venv_ui/bin/python" "$ROOT/.venv_laptop/bin/python" "$(dirname "$ROOT")/GR00T-WholeBodyControl/.venv_sim/bin/python" python3; do
  if command -v "$candidate" >/dev/null && "$candidate" -c 'import tkinter' 2>/dev/null; then
   exec "$candidate" -m sonic_laptop.app --demo
  fi
 done
 echo 'LOCAL: install Python Tk support (Ubuntu: sudo apt-get install python3-tk), then rerun.' >&2
 exit 1
}

if [[ "${1:-}" == --offline ]]; then offline_preview; fi
if [[ ! -f "$ROOT/.runtime/sonic/env.sh" || ! -x "$ROOT/.runtime/sonic/deploy/target/release/g1_deploy_onnx_ref" ]]; then
 echo 'LOCAL: no SONIC GPU setup on this laptop (scripts/laptop/setup.sh); opening the offline UI preview.' >&2
 offline_preview
fi
source "$ROOT/scripts/laptop/runtime.sh"
require_sonic_environment
SIM_PYTHON="${SONIC_SIM_PYTHON:-$SONIC_UPSTREAM/.venv_sim/bin/python}"
if [[ ! -x "$SIM_PYTHON" ]]; then
 cat >&2 <<EOF
LOCAL: the MuJoCo simulator environment is missing. Install it once:
  cd $SONIC_UPSTREAM && bash install_scripts/install_mujoco_sim.sh
or set SONIC_SIM_PYTHON to a Python with gear_sonic[sim] and unitree_sdk2py installed.
EOF
 exit 1
fi

# Fresh import each run, so the demo clips pass the same checks and limits as any other library.
DEMO="$(mktemp -d "$ROOT/.runtime/sonic/demo-XXXXXX")"
python -m sonic_laptop.import_kimodo motions/sim_demo/idle/idle_apose.csv --source-fps 30 \
 --upstream "$SONIC_UPSTREAM" --output "$DEMO/idle"
python -m sonic_laptop.import_kimodo motions/sim_demo/clips --source-fps 30 \
 --upstream "$SONIC_UPSTREAM" --output "$DEMO/motions"

# The simulator resolves its scene relative to the upstream checkout, and must use its gear_sonic.
(cd "$SONIC_UPSTREAM" && PYTHONPATH="$SONIC_UPSTREAM" exec "$SIM_PYTHON" "$ROOT/scripts/laptop/sim.py" --idle "$DEMO/idle") \
 >"$DEMO/sim.log" 2>&1 &
SIM=$!
trap 'kill "$SIM" 2>/dev/null || true' EXIT

cat <<EOF

SONIC MuJoCo demo: simulated G1 on interface lo. Simulator log: $DEMO/sim.log
 1. Wait for the MuJoCo window, then click "1  Suspended setup" in the panel and confirm.
 2. When the panel shows CHECKS PASSED, click into the MuJoCo window and press 7 once:
    the band sets the feet on the floor and still carries the weight.
 3. Click "2  Feet down / start" and confirm, then press 9 in the MuJoCo window to release
    the band: SONIC balances the robot and holds the idle.
 4. Motions tab: select a clip and click Play once, or Play all.
 5. Keyboard tab: Enable keyboard control, keep the panel focused, hold Space + W/A/S/D.
    Keyboard mode cannot return to motions; close the panel and rerun demo.sh to restart.
 Esc or the red button latches damping. Close the panel to end; the simulator closes with it.
 Simulator keys: 7/8 lower/raise the band, 9 toggles it, Backspace resets the robot.

EOF
sleep 2
kill -0 "$SIM" 2>/dev/null || { tail -20 "$DEMO/sim.log" >&2; echo 'LOCAL: the simulator exited; see the log above.' >&2; exit 1; }
bash "$ROOT/scripts/laptop/run.sh" --sim --interface lo --motions "$DEMO/motions" --idle "$DEMO/idle" "$@"
