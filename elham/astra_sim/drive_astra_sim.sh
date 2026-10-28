#!/usr/bin/env bash
# Drive gpt_astra/run.py --backend sim through its operator prompts (README step 4), in a private
# user+net+mount namespace (own loopback/DDS/ZMQ ports, lock file, /dev/shm) so it can run beside
# another G1 sim without cross-talk. Everything but the sim (CPU 3) and controller (CPU 2) stays on
# the E-cores 12-19.
# Usage: [NET=1] drive_astra_sim.sh <log-name> [extra run.py args...]   (NET=1: internet via slirp4netns)
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
WS=$ROOT/.runtime/astra_sim

export TMUX_TMPDIR=$WS/.tmux
mkdir -p "$TMUX_TMPDIR"
NAME=$1; shift
LOG=$WS/$NAME.log
: > "$LOG"
tmux kill-session -t astra 2>/dev/null
printf 'nameserver 10.0.2.3\n' > "$WS/resolv.conf"
tmux new-session -d -s astra -x 220 -y 50 \
  "taskset -c 12-19 unshare -rnm --propagation private bash -c 'ip link set lo up; mount -t tmpfs tmpfs /dev/shm; exec bash'"
sleep 1
if [ "${NET:-0}" = 1 ]; then
  PANE_PID=$(tmux display -p -t astra '#{pane_pid}')
  slirp4netns --configure --mtu=65520 --disable-host-loopback "$PANE_PID" tap0 > "$WS/slirp.log" 2>&1 &
  sleep 2
  tmux send-keys -t astra "mount --bind $WS/resolv.conf /etc/resolv.conf" C-m
  sleep 1
fi
tmux send-keys -t astra "cd $ROOT && source .runtime/sonic/env.sh && export SONIC_CONTROL_CPU=2 GPT_ASTRA_BALLOON=hover && OPENAI_API_KEY=\"\$(cat ~/.config/gpt_astra/openai_key)\" gpt_astra/.venv/bin/python gpt_astra/run.py --backend sim --build $ROOT/.runtime/sonic/deploy-sim $* 2>&1 | tee $LOG" C-m

fail() { echo "DRIVER STOP: $1"; tail -25 "$LOG"; exit 1; }
wait_for() {  # pattern timeout_s
  for _ in $(seq 1 "$2"); do
    grep -qE "$1" "$LOG" && return 0
    grep -qE "Traceback|RuntimeError|ValueError|run.py: error" "$LOG" && fail "error while waiting for '$1'"
    sleep 1
  done
  fail "timeout waiting for '$1'"
}
enter() { tmux send-keys -t astra "" C-m; echo "[driver] ENTER after: $1"; }

wait_for "Waiting up to 30s for fresh images" 120
SIMPID=$(pgrep -f "$ROOT/gpt_astra/sim\.py" | head -1)
[ -n "$SIMPID" ] && taskset -a -p -c 3 "$SIMPID" >/dev/null && echo "[driver] pinned sim $SIMPID to CPU 3"
wait_for "CAMERA CHECK" 180;   sleep 3; enter "camera check"
wait_for "SUSPENDED SETUP" 120; enter "suspended setup"
wait_for "CHECKS PASSED" 600;  enter "checks passed"
wait_for "LOWERING" 60
RUNDIR=$(grep -m1 "^Run folder: " "$LOG" | sed 's/^Run folder: //')
[ -d "$RUNDIR" ] || fail "run folder not found"
# Gantry order through the fixture's run-folder triggers (keys to an off-screen viewer are ignored):
# lower one band step so the feet carry weight with the band still loaded, settle, start, then release.
if [ "${ORDER:-gantry}" = readme ]; then
  # README order: release suspension (9) at the lowering prompt, THEN Enter starts SONIC.
  touch "$RUNDIR/band_lower"; echo "[driver] band_lower trigger"
  sleep 3; touch "$RUNDIR/band_release"; echo "[driver] band_release trigger (before start, README order)"
  sleep 5; enter "lowering"
  wait_for "standing_reference_active" 60
else
  touch "$RUNDIR/band_lower"; echo "[driver] band_lower trigger"
  sleep 6; enter "lowering"
  wait_for "standing_reference_active" 60
  touch "$RUNDIR/band_release"; echo "[driver] band_release trigger"
fi
wait_for "Session finished" 900; sleep 2; enter "session finished"
wait_for "\[finished\]" 120
echo "DRIVER DONE"
pkill -f "slirp4netns --configure --mtu=65520 --disable-host-loopback [0-9]" 2>/dev/null
tail -5 "$LOG"
