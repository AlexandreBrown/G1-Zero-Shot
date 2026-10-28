#!/usr/bin/env bash
# GR00T N1.7 VLA + SONIC whole-body control, following
# https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html
#
#   bash run.sh sim  ["prompt"]   # MuJoCo G1 on this laptop, nothing touches hardware
#   bash run.sh real ["prompt"]   # real G1 over Ethernet; needs pc2_camera_server.py running on PC2
#   bash run.sh stop              # kill the policy server and the inference tmux session
#
# The policy server runs in tmux session "groot_policy", the tutorial's 4-pane launcher in
# "sonic_inference" (plus window "sim" in sim mode). Both live on a private tmux socket:
#   TMUX_TMPDIR=<this dir>/.tmux tmux attach -t sonic_inference
set -eo pipefail
MODE=${1:?usage: run.sh sim|real|stop [prompt]}
PROMPT=${2:-grab the bottle}
source "$(dirname "$(readlink -f "$0")")/env.sh"
export TMUX_TMPDIR=$GROOT_VLA/.tmux
mkdir -p "$TMUX_TMPDIR" "$GROOT_VLA/logs"

if [[ $MODE == stop ]]; then
  tmux kill-session -t sonic_inference 2>/dev/null || true
  tmux kill-session -t groot_policy 2>/dev/null || true
  echo "stopped"; exit 0
fi

# Every G1 sim/deploy on this laptop shares the default DDS domain, so two at once cross-talk.
if ! tmux has-session -t sonic_inference 2>/dev/null; then
  OTHERS=$(pgrep -af "g1_deploy_onnx_re[f]|run_sim_loo[p]|sonic_lapto[p]|scripts/laptop/si[m]\.py" || true)
  if [[ -n $OTHERS ]]; then
    echo "Another G1 sim/deploy is running; stop it first:"; echo "$OTHERS"; exit 1
  fi
fi

if ! tmux has-session -t groot_policy 2>/dev/null; then
  : > "$GROOT_VLA/logs/policy_server.log"
  tmux new-session -d -s groot_policy -x 200 -y 50 \
    "cd $ISAAC_GROOT && PYTHONUNBUFFERED=1 GR00T_BACKBONE_FROM_CONFIG=1 GR00T_BACKBONE_LOCAL_DIR=$GR00T_BACKBONE_LOCAL_DIR HF_HUB_OFFLINE=1 $GROOT_VLA/.venv/bin/python \
     gr00t/eval/run_gr00t_server.py --model-path $VLA_CKPT --embodiment-tag UNITREE_G1_SONIC \
     --device cuda:0 --host 127.0.0.1 --port 5550 2>&1 | tee $GROOT_VLA/logs/policy_server.log; \
     echo 'policy server exited' | tee -a $GROOT_VLA/logs/policy_server.log; read"
fi
echo "Waiting for the GR00T policy server (first load takes ~1 min)..."
until ss -ltn | grep -q "127.0.0.1:5550 " || grep -qE "Traceback|policy server exited" "$GROOT_VLA/logs/policy_server.log"; do sleep 2; done
if ! ss -ltn | grep -q "127.0.0.1:5550 "; then
  tail -20 "$GROOT_VLA/logs/policy_server.log"; exit 1
fi

cd "$WBC"
if [[ $MODE == sim ]]; then
  ARGS=(--sim)
else
  ARGS=(--camera-host 192.168.123.164)
fi
exec "$GROOT_VLA/.venv/bin/python" gear_sonic/scripts/launch_inference.py "${ARGS[@]}" \
  --no-data-exporter --policy-host 127.0.0.1 --policy-port 5550 --prompt "$PROMPT"
