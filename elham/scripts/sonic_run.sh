#!/usr/bin/env bash
# Launch the GEAR-SONIC deploy program (g1_deploy_onnx_ref) for ONE trial, in MuJoCo sim2sim or on the real G1.
#
# Why not upstream deploy.sh: it silently treats any option it does not know as the network interface (so the
# CSV-logging flags never reach the program), falls back to the first interface it finds in `real` mode, and
# defaults to the Hugging Face ONNX files (a different network from sonic_release/last.pt). This wrapper makes
# the policy, motions, interface and logging explicit, refuses unsafe defaults, and writes a manifest per trial.
#
#   scripts/sonic_run.sh --sim  --policy artifacts/policies/base --motions artifacts/motions/single/squat_lv1 \
#       --session 2026-10-22_sim --trial base_squat_1
#   scripts/sonic_run.sh --real --iface enp3s0 --policy ... --motions ... --session ... --trial ...
#
# Keys once running (manager input, keyboard): ] start policy | T play | R back to frame 0 | N/P next/prev motion
# | O STOP (damping, then exit) | @ switch to the robot's remote (Start, A play, B frame 0, Select STOP).
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_ROOT="${SONIC_ROOT:-}"
MODE="" IFACE="" POLICY="" MOTIONS="" SESSION="" TRIAL="" PLANNER="" KP_SCALE="" KD_SCALE=""
INPUT_TYPE="manager" DRY_RUN=0 ALLOW_HF=0 IGNORE_TRT=0 LOG_ROOT="$REPO/logs" RECORDER=1 RECORDER_PYTHON=""

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Required: --sim | --real, --policy DIR, --motions DIR, --session NAME, --trial NAME, and --iface NIC with --real.
Options:  --sonic-root DIR (or $SONIC_ROOT)   --input-type manager|keyboard|gamepad (default manager)
          --planner FILE (default: none; the kinematic planner is off, reference playback only)
          --kp-scale SPEC --kd-scale SPEC (per-motor gain scaling, e.g. 0-28=1.0; hardware indices)
          --log-root DIR (default <repo>/logs)   --dry-run (print the command, write nothing)
          --allow-hf-policy (run a policy dir without our manifest.json)   --ignore-trt-version
          --recorder-python PY (python with unitree_sdk2py; default $SONIC_ROOT/.venv_sim/bin/python)
          --no-recorder (skip the full-rate LowState/LowCmd recording and the 2 s damping hold after exit;
                         the program's 50 Hz logs are still written)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim) MODE=sim; shift ;;
    --real) MODE=real; shift ;;
    --iface) IFACE="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --motions) MOTIONS="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --trial) TRIAL="$2"; shift 2 ;;
    --sonic-root) SONIC_ROOT="$2"; shift 2 ;;
    --input-type) INPUT_TYPE="$2"; shift 2 ;;
    --planner) PLANNER="$2"; shift 2 ;;
    --kp-scale) KP_SCALE="$2"; shift 2 ;;
    --kd-scale) KD_SCALE="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-hf-policy) ALLOW_HF=1; shift ;;
    --ignore-trt-version) IGNORE_TRT=1; shift ;;
    --recorder-python) RECORDER_PYTHON="$2"; shift 2 ;;
    --no-recorder) RECORDER=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done

[[ -n "$MODE" ]] || die "choose --sim or --real"
[[ -n "$SONIC_ROOT" ]] || die "set --sonic-root or SONIC_ROOT to the GR00T-WholeBodyControl checkout"
[[ -n "$POLICY" && -n "$MOTIONS" ]] || die "--policy and --motions are required"
[[ -n "$SESSION" && -n "$TRIAL" ]] || die "--session and --trial are required (they name the log folder)"
[[ "$SESSION$TRIAL" =~ ^[A-Za-z0-9._/-]+$ ]] || die "--session/--trial may only use letters, digits and . _ - /"

SONIC_ROOT="$(cd "$SONIC_ROOT" && pwd)"
POLICY="$(cd "$POLICY" && pwd)"
MOTIONS="$(cd "$MOTIONS" && pwd)"
DEPLOY="$SONIC_ROOT/gear_sonic_deploy"
BINARY="$DEPLOY/target/release/g1_deploy_onnx_ref"
[[ -x "$BINARY" ]] || die "$BINARY not built: cd $DEPLOY && source scripts/setup_env.sh && just build"

for f in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
  [[ -f "$POLICY/$f" ]] || die "$POLICY/$f missing (a SONIC policy folder: encoder, decoder, observation config)"
done
if [[ ! -f "$POLICY/manifest.json" ]]; then
  ((ALLOW_HF)) || die "$POLICY has no manifest.json, so it was not exported and parity-checked from a checkpoint (the \
Hugging Face files are a different network from sonic_release/last.pt). Pass --allow-hf-policy only for a deliberate stack check."
  warn "running a policy without manifest.json (--allow-hf-policy)"
fi
[[ -w "$POLICY" ]] || die "$POLICY must be writable: TensorRT caches its engines next to the ONNX files"
if [[ "$MODE" == real && -f "$POLICY/manifest.json" ]]; then
  python3 - "$POLICY" <<'PY' || die "the policy folder did not pass its checks (see above); fix it before using the robot"
import json, sys
from pathlib import Path
policy = Path(sys.argv[1])
problems = json.loads((policy / "manifest.json").read_text()).get("problems", [])
parity = policy / "parity.json"
ok = parity.exists() and json.loads(parity.read_text()).get("pass") is True
if problems:
    print("manifest.json lists export problems:", problems)
if not ok:
    print(f"{parity} is missing or not passing: the ONNX files were not checked against the checkpoint")
sys.exit(0 if ok and not problems else 1)
PY
fi

mapfile -t MOTION_DIRS < <(find -L "$MOTIONS" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/joint_pos.csv' ';' -print | sort)
((${#MOTION_DIRS[@]} > 0)) || die "$MOTIONS has no motion subfolders (each needs joint_pos.csv); see scripts/export_motions.py"
if ((${#MOTION_DIRS[@]} > 1)); then
  warn "${#MOTION_DIRS[@]} motions in $MOTIONS: the program loads them in an unspecified order and N/P switch \
without blending. Read the index -> name lines in the startup log before pressing T."
fi

case "$(uname -m)" in
  x86_64) TRT_WANT="10.13" ;;
  aarch64) TRT_WANT="10.7" ;;
  *) TRT_WANT="" ;;
esac
TRT_HEADER="${TensorRT_ROOT:-/nonexistent}/include/NvInferVersion.h"
[[ -f "$TRT_HEADER" ]] || TRT_HEADER="/usr/include/$(uname -m)-linux-gnu/NvInferVersion.h"
TRT_HAVE="unknown"
if [[ -f "$TRT_HEADER" ]]; then
  # NV_TENSORRT_MAJOR is either a number or another macro (TRT_MAJOR_ENTERPRISE in newer headers): resolve once.
  TRT_HAVE="$(awk '/^#define/ {v[$2]=$3} END {ma=v["NV_TENSORRT_MAJOR"]; if (ma in v) ma=v[ma];
    mi=v["NV_TENSORRT_MINOR"]; if (mi in v) mi=v[mi]; print ma"."mi}' "$TRT_HEADER")"
fi
if [[ "$TRT_HAVE" != "$TRT_WANT" ]]; then
  msg="TensorRT $TRT_HAVE found, $TRT_WANT required on $(uname -m) (a wrong version gives silently wrong actions)"
  ((IGNORE_TRT)) && warn "$msg" || die "$msg; pass --ignore-trt-version only if you have checked it yourself"
fi

if pgrep -x g1_deploy_onnx_ref >/dev/null; then
  die "another g1_deploy_onnx_ref is running (pid $(pgrep -x g1_deploy_onnx_ref | tr '\n' ' ')): two programs would both command the robot"
fi

ARGS=()
if [[ "$MODE" == real ]]; then
  [[ -n "$IFACE" && "$IFACE" != lo ]] || die "--real needs --iface <the NIC wired to the robot> (not lo)"
  [[ "$INPUT_TYPE" == manager ]] || die "--real needs --input-type manager: keyboard-only drops the remote's Select stop, \
gamepad-only drops the keyboard O stop"
  if ((DRY_RUN == 0)); then
    [[ -n "${TMUX:-}" ]] || die "--real must run inside tmux (tmux new -s sonic): a lost plain terminal leaves no way to press O"
  fi
  ip link show "$IFACE" >/dev/null 2>&1 || die "interface $IFACE does not exist (see: ip -br addr)"
  ip -4 addr show "$IFACE" | grep -q 'inet 192\.168\.123\.' || die "$IFACE has no 192.168.123.x address \
(sudo ip addr add 192.168.123.222/24 dev $IFACE)"
  if ((DRY_RUN == 0)); then
    ping -c 3 -W 1 192.168.123.161 >/dev/null 2>&1 || die "the robot (192.168.123.161) does not answer ping on $IFACE"
  fi
else
  IFACE=lo
  ARGS+=(--disable-crc-check)
fi

LOG_DIR="$LOG_ROOT/$SESSION/$TRIAL"
[[ ! -e "$LOG_DIR" ]] || die "$LOG_DIR exists; use a new --trial name (logs are never overwritten)"
[[ "$LOG_DIR" != *" "* ]] || die "the log path may not contain spaces: $LOG_DIR"

RECORDER_PYTHON="${RECORDER_PYTHON:-$SONIC_ROOT/.venv_sim/bin/python}"
# setsid: the recorder survives a dying terminal (SIGHUP) long enough for the EXIT trap to stop it cleanly.
RECORDER_CMD=(setsid nice -n 10 "$RECORDER_PYTHON" "$REPO/scripts/session_recorder.py" --iface "$IFACE" --out "$LOG_DIR/hw"
  --parent-pid "$$")
DAMPING_CMD=("$RECORDER_PYTHON" "$REPO/scripts/damping_hold.py" --iface "$IFACE" --seconds 2)
if ((RECORDER && DRY_RUN == 0)); then
  "$RECORDER_PYTHON" -c "import unitree_sdk2py" 2>/dev/null || die "$RECORDER_PYTHON cannot import unitree_sdk2py \
(run install_scripts/install_mujoco_sim.sh, pass --recorder-python, or --no-recorder to skip the full-rate recording)"
fi

ARGS+=(--obs-config "$POLICY/observation_config.yaml" --encoder-file "$POLICY/model_encoder.onnx")
ARGS+=(--input-type "$INPUT_TYPE" --output-type all --zmq-host localhost)
ARGS+=(--enable-csv-logs --logs-dir "$LOG_DIR/state")
ARGS+=(--target-motion-logfile "$LOG_DIR/target.csv" --policy-input-logfile "$LOG_DIR/obs.csv")
ARGS+=(--record-input-file "$LOG_DIR/inputs.csv")
[[ -n "$PLANNER" ]] && ARGS+=(--planner-file "$(cd "$(dirname "$PLANNER")" && pwd)/$(basename "$PLANNER")")
[[ -n "$KP_SCALE" ]] && ARGS+=(--motor-kp-scale "$KP_SCALE")
[[ -n "$KD_SCALE" ]] && ARGS+=(--motor-kd-scale "$KD_SCALE")
CMD=("$BINARY" "$IFACE" "$POLICY/model_decoder.onnx" "$MOTIONS" "${ARGS[@]}")

if ((DRY_RUN)); then
  ((RECORDER)) && { printf 'recorder: '; printf '%q ' "${RECORDER_CMD[@]}"; echo; }
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

mkdir -p "$LOG_DIR/state"
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)  arch: $(uname -m)  tensorrt: $TRT_HAVE"
  echo "mode: $MODE  iface: $IFACE  input_type: $INPUT_TYPE  planner: ${PLANNER:-none}"
  echo "kp_scale: ${KP_SCALE:-none}  kd_scale: ${KD_SCALE:-none}"
  echo "sonic_git: $(git -C "$SONIC_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "this_repo_git: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "policy: $POLICY"
  (cd "$POLICY" && md5sum model_encoder.onnx model_decoder.onnx observation_config.yaml) | sed 's/^/  /'
  echo "motions: $MOTIONS"
  for d in "${MOTION_DIRS[@]}"; do echo "  $(basename "$d") $(md5sum < "$d/joint_pos.csv" | cut -d' ' -f1)"; done
  printf 'command: '; printf '%q ' "${CMD[@]}"; echo
} > "$LOG_DIR/run_manifest.txt"
[[ -f "$POLICY/manifest.json" ]] && cp "$POLICY/manifest.json" "$LOG_DIR/policy_manifest.json"

if [[ "$MODE" == real ]]; then
  cat <<EOF

================ REAL ROBOT ================
The program moves the robot as soon as it connects: it ramps every joint to the stand pose over 3 s.
Confirm all of these (docs/CHECKLIST.md, section D):
  - robot HANGING on the gantry, feet clear of the floor, in debug mode (L2+R2), joints damped
  - rope handler, remote holder and observer in place; 3 m clear zone; cameras recording
  - you know the stops: the LETTER O (not zero) in THIS pane, Select on the remote (after @), hoist.
    A stop prints 'EMERGENCY STOP triggered'. Remote B only returns to frame 0 (the policy keeps running).
  - run inside tmux: losing this terminal does NOT stop the robot safely
  - right after 'Init Done' (robot still hanging) press @ so the remote works: Start, A play, Select STOP
  - from typing 'hanging' until the robot stands on the floor, HANDS OFF the keyboard: an early ] or T is
    remembered and would start the policy in the air (if you see 'transitioning to CONTROL' before pressing ],
    press O and relaunch)
Policy:  $POLICY
Motions: $(for d in "${MOTION_DIRS[@]}"; do printf '%s ' "$(basename "$d")"; done)
Gains:   kp scale ${KP_SCALE:-none}, kd scale ${KD_SCALE:-none}; planner ${PLANNER:-off}
Logs:    $LOG_DIR
EOF
  read -r -p "Type 'hanging' to start: " answer
  [[ "$answer" == hanging ]] || { echo "aborted"; rm -rf "$LOG_DIR"; exit 1; }
fi

REC_PID=""
DEPLOY_STARTED=0
after_exit() {
  # Whatever ended the program (O, Select, a self-stop, a crash, Ctrl+C), hold damping for 2 s from here too:
  # the program itself sends a single damping packet, or none at all if it was killed.
  if ((DEPLOY_STARTED && RECORDER)); then
    "${DAMPING_CMD[@]}" 2>&1 | grep -v multicast || true
  fi
  if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    sleep 1  # keep recording what the robot does right after the stop
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
}
trap after_exit EXIT
if ((RECORDER)); then
  # A launcher killed without its exit trap leaves its recorder running; it only listens, so stop it here.
  STALE_RECORDERS=$(pgrep -f 'scripts/session_recorde[r].py' || true)
  if [[ -n "$STALE_RECORDERS" ]]; then
    warn "stopping a session_recorder left over from an earlier launch (pid ${STALE_RECORDERS//$'\n'/ })"
    # shellcheck disable=SC2086
    kill -INT $STALE_RECORDERS 2>/dev/null || true
  fi
  "${RECORDER_CMD[@]}" > "$LOG_DIR/recorder.log" 2>&1 &
  REC_PID=$!
  sleep 1
  kill -0 "$REC_PID" 2>/dev/null || die "session_recorder exited at start; see $LOG_DIR/recorder.log"
fi

cd "$DEPLOY"
set +e +o pipefail  # upstream's environment script has commands that fail harmlessly (deploy.sh does the same)
# shellcheck disable=SC1091
source scripts/setup_env.sh
set -eo pipefail
# drop anything typed so far (e.g. after 'hanging'), so no stale ] or T reaches the program
python3 -c 'import sys, termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)' 2>/dev/null || true
DEPLOY_STARTED=1
"${CMD[@]}"
