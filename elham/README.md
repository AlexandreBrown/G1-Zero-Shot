# elham/ — run SONIC on the G1 with full logging

Elham's tools for running NVIDIA's released SONIC weights on our G1 from the deploy laptop: a launcher for NVIDIA's
stock deploy program (`g1_deploy_onnx_ref`, unchanged), a read-only pre-flight and live monitor, a full-rate
recorder, log analysis, dataset packing, and a MuJoCo preview that runs the policy the way the robot program does.

Everything lives in this folder; nothing outside `elham/` is changed. **Alex's controller panel (`sonic_laptop/`,
`scripts/laptop/`) is a separate path** with its own guarded build. Never run both at once: only one program may
command the robot.

**Our robot:** G1 EDU U4 (29 DoF + Dex3-1 hands), Machine Type 5 (old 7520-14 hip-pitch motor, grey pelvis),
driven from the deploy laptop over Ethernet.

| Doc | What |
|---|---|
| **This README** | How to run the robot: what to type in which terminal, the stops, what the messages mean |
| [docs/CHECKLIST.md](docs/CHECKLIST.md) | Printable checklist for the day: room, bring-up, per trial, aborts, after a fall |
| [docs/MOTIONS.md](docs/MOTIONS.md) | Which motions in which order, the simulation evidence, the expected standing lean per motion |
| [docs/HARDWARE_NOTES.md](docs/HARDWARE_NOTES.md) | Every hardware fact used here, with its source |

## Results so far (2026-10-21, released SONIC weights, robot on a gantry)

| Motion (trial) | Tracking error, robot / sim | Max tilt, robot / sim | Standing lean on frame 0, robot / sim | Hottest motor |
|---|---|---|---|---|
| step_forward_back_lv2 (`base_stand_6`) | 0.112 / 0.110 rad | 11.4 / 11.7° | −0.1 / −1.4° | 64 °C after 3 min |
| squat_lv1, played twice (`base_squat_2`) | 0.096 / 0.095 rad | 16.1 / 16.6° | 13.4 / 11.4° | 50 °C |

Sim = `scripts/preview_mujoco.py` with 5 ms delay. Both trials pass the gate (no fall, no saturation, no joint at
its limit). The robot stands 1–2° more forward than the simulation, far from the ~5° reported for other
grey-pelvis robots. The largest torque came at `Start` (left ankle pitch 52 Nm for 0.06 s, the known start-up
transient); during the squats the ankles stay below ~19 Nm. Loop timing: robot-state age 5–8 ms, policy 0.4 ms.

---

## Run the robot

### The three terminals

| Terminal | What runs there | Never |
|---|---|---|
| **A**: tmux left pane | `scripts/sonic_run.sh`, the program that drives the robot. **The only place where the letter `O` stops it.** | Type anything else here while it runs |
| **B**: tmux right pane | `scripts/lowstate_monitor.py`, read-only: temperatures, joint limits, rate. Leave it running all session. | |
| **C**: any other terminal | Spare: emergency commands, checks, analysis after the trial | |

Every terminal starts with:
```bash
cd ~/Desktop/my_code/G1-Zero-Shot/elham
export SONIC_ROOT=~/Desktop/my_code/sonic_hs1
source $SONIC_ROOT/.venv_sim/bin/activate
```

tmux in one minute:
- `tmux new -s sonic` opens the session;
- `Ctrl+b`, release, then `%` splits it (or type `tmux split-window -h`);
- `Ctrl+b`, release, then an arrow key moves between panes (or `tmux set -g mouse on` and click);
- with the mouse on, select text with **Shift + drag** and copy with `Ctrl+Shift+C`;
- `tmux capture-pane -p -S -1000 > ~/Desktop/pane.txt` saves a pane's history to a file.

### Before the session

**Laptop** (terminal C):
```bash
systemctl --user stop read-aloud.service   # no other GPU programs during a session (one froze this laptop)
nvidia-smi                                 # nothing else on the GPU
ip -br addr show enp58s0                   # UP 192.168.123.222/24; if not: sudo nmcli con up g1
ping -c 20 192.168.123.161                 # 0 % loss
```
The `g1` network profile is created once (`sudo nmcli con add type ethernet ifname enp58s0 con-name g1
ipv4.method manual ipv4.addresses 192.168.123.222/24`). Running it again makes a duplicate; delete that one by
its UUID.

**Robot:** waist unlocked, battery ≥ 80 %, hanging on the gantry with the **feet clear of the floor**, damping
(`L2+B` on V1.0.4), then debug mode (`L2+R2`). The team has read the stops below.

### One trial

| Step | Where | Do | You should see |
|---|---|---|---|
| 1 | B | `python scripts/lowstate_monitor.py --iface enp58s0 --check 5` | after ~10 s: `PRE-FLIGHT: PASS` (a yellow warning for warm motors is fine) |
| 2 | B | `python scripts/lowstate_monitor.py --iface enp58s0` | a status line every second, ending `last message 0.0 s ago` |
| 3 | A | the launch command below, **on one line**, with a new `--trial` name | the safety banner, then `Type 'hanging' to start:` |
| 4 | A | type `hanging`, then **hands off the keyboard** | 20–40 s of loading; the robot does not move yet |
| 5 | — | wait | the robot moves to the stand pose over 3 s, hands close and open, `Init Done` |
| 6 | A | press `@` (Shift+2), robot still in the air | `Switched to: GAMEPAD`; the remote now works |
| 7 | rope | lower the robot until **both feet are flat**, rope near-taut but not carrying it | do not wait here: the robot cannot balance yet |
| 8 | remote | **`Start`** | `transitioning to CONTROL`; a quick adjustment of arms and knees; it stands on frame 0, then `Loop timing` lines every second |
| 9 | rope | give 5–10 cm of slack | the robot stands by itself, leaning about as MOTIONS.md says for this motion |
| 10 | remote | **`A`** to play the motion (skip it for a stand test) | `Playing motion 0 …`, then `… completed. Reset to frame 0.` |
| 11 | rope + remote | re-tension the rope, then **`Select`** | `Stop`, `Program exiting normally`, then `damping_hold: sent 1000 damping commands`; the joints go soft |
| 12 | C | `python scripts/analyze_logs.py logs/<date>_g1/<trial>` | the trial's numbers and `gate True` |

Launch command (step 3), one line, `squat_lv1` replaced by the motion:
```bash
scripts/sonic_run.sh --real --iface enp58s0 --policy artifacts/policies/base --motions artifacts/motions/single/squat_lv1 --session $(date +%F)_g1 --trial base_squat_3
```
- **Stand test:** use `step_forward_back_lv2` (its frame 0 is nearly level) and skip step 10.
- `artifacts/policies/base` is the released SONIC policy folder (encoder, decoder, observation config, plus the
  `manifest.json` / `parity.json` of its export from `sonic_release/last.pt`); `artifacts/motions/single/<motion>` are
  the motion folders from `scripts/export_motions.py`. Both are git-ignored; on the deploy laptop they link to
  `zerosho/outputs/zerosho/sonic_g1_deploy/`.
- **Keep each launch under ~2 min**, and cool down hanging and damped when the monitor reports 70 °C.
- **Only motions with ≤ 0.5 m of travel** on a fixed hoist (MOTIONS.md); walking needs a trolley.
- **The first time each day:** the two stop tests in the air before any floor trial. Launch, `Init Done`, press
  `O` → the joints go soft. Relaunch, `Init Done`, `@`, `Select` → the joints go soft.

### Stops

| Stop | Works | You see |
|---|---|---|
| **The letter `O`** (not zero) in terminal A | always, also after `@` | `EMERGENCY STOP triggered`, `Stop`, the damping line |
| **`Select`** on the remote | after `@` | `Stop`, the damping line |
| Terminal C: `pkill -x g1_deploy_onnx_ref` | if A does not respond | the launcher then sends 2 s of damping |
| Terminal C: `python scripts/damping_hold.py --iface enp58s0 --seconds 5` | if the robot is still stiff after a stop | `damping_hold: sent …` |
| Power button (short press, then hold > 2 s), rope tight | last resort | |

After any stop the rope handler takes the weight; hoist at once only if the robot is already falling (hoisting a
robot whose policy still runs unloads its feet, and the policy then shakes). **Not stops:** the digit `0`,
remote `B` (back to frame 0, the policy keeps running), `@`, Unitree's `L2+B` / `L1+A`, `Ctrl+C` (kills the program
without its damping command; the launcher's hold still runs).

**Never:** `]` or `Start` with the feet in the air (the policy shakes violently without load on its feet), `A`
during a stand test, `T` / `N` / `P` / `Enter` on the keyboard.

### What the messages mean

| Message | Meaning | Do |
|---|---|---|
| `ERROR: logs/... exists; use a new --trial name` | trial logs are never overwritten | new `--trial` name |
| `zsh: permission denied: artifacts/policies/base` | the pasted command broke over two lines | paste it as one line |
| `PRE-FLIGHT: FAIL … <joint> +0.0xx rad from its limit` | a limp leg rests against a joint stop | robot hanging, move that joint off its stop by hand, check again |
| monitor `LowState stopped arriving` | no robot state reached the laptop for 0.5 s | `ip -br addr show enp58s0`, `ping`; reseat the cable |
| `Loop timing - LowState age: N ms` | how old the robot state is when the policy uses it; 5–8 ms is normal here | abort if above ~10 ms more than once |
| `G1 type: 5` | Machine Type 5: old hip-pitch motor | expected |
| `TensorRT_ROOT is not set`, `ROS2 not found` | from NVIDIA's environment script | harmless |
| `[ErrorMonitor]`, `CRC`, `HighTemp`, `Lost LowState` | a robot fault | stop (see docs/CHECKLIST.md aborts) |

### After the session

Terminal C, for every trial of the day:
```bash
python scripts/analyze_logs.py logs/<date>_g1/* --json logs/<date>_g1_summary.json
python scripts/build_dataset.py logs/<date>_g1/*      # -> <trial>/episode.npz + episode.json
systemctl --user start read-aloud.service
```

What a trial folder holds:
- `state/*.csv`: the program's 50 Hz log of every control tick (joints, IMUs, actions, tokens, temperatures);
- `obs.csv`, `target.csv`, `inputs.csv`: policy inputs, the reference, operator inputs;
- `hw/`: every robot message at full rate (LowState ~1 kHz, LowCmd 500 Hz, torso IMU);
- `run_manifest.txt`: command, file hashes, versions.

The 1 kHz `hw/` recordings of 2026-10-21 are partial (the first recorder kept ~40 %; fixed since); the 50 Hz logs
are complete.

---

## Setup (once per laptop)

- **SONIC and TensorRT:** follow the repository README (TensorRT 10.13 on x86). These tools use NVIDIA's stock
  deploy program from a GR00T-WholeBodyControl checkout, `$SONIC_ROOT`, built with upstream's own
  `cd $SONIC_ROOT/gear_sonic_deploy && source scripts/setup_env.sh && just build`. On the deploy laptop it is already
  built: `SONIC_ROOT=~/Desktop/my_code/sonic_hs1` (commit `b042411`).
- **Python:** the upstream simulator environment `$SONIC_ROOT/.venv_sim` (MuJoCo, unitree_sdk2py) plus
  `uv pip install --python $SONIC_ROOT/.venv_sim/bin/python onnxruntime pytest` for the preview and the tests.
- **Network:** the `g1` profile above.

## What is in this folder

| Script | Does |
|---|---|
| `scripts/sonic_run.sh` | Launches NVIDIA's deploy program for one trial (sim or real); refuses unsafe setups; logs; records; holds damping 2 s after every exit |
| `scripts/lowstate_monitor.py` | Read-only pre-flight check (`--check 5`) and live temperature / limit alerts |
| `scripts/lowstate_sources.py` | Read-only: who publishes the robot state, at what rate, with which tick steps |
| `scripts/session_recorder.py` | Read-only recorder of every LowState / LowCmd / torso-IMU message during a trial (started by the launcher) |
| `scripts/g1_cdr.py`, `scripts/dds_reader.py` | Raw DDS reading and bulk numpy decoding of the G1 messages (~10 µs per LowState instead of ~3 ms) |
| `scripts/damping_hold.py` | Sends the program's own damping command at 500 Hz for a few seconds |
| `scripts/analyze_logs.py` | Per trial: torque, joint speed and margins, tilt, standing lean, tracking error; gate pass/fail |
| `scripts/build_dataset.py` | Packs a trial into `episode.npz` (states, actions, motor targets, policy inputs, reference, full-rate streams) |
| `scripts/preview_mujoco.py` | Runs an ONNX policy on a motion in MuJoCo the way the robot program does: video + numbers; `--delay-ms`, `--revision old/new` |
| `scripts/export_motions.py` | ASAP motions (SONIC PKL format) → the deploy program's 50 Hz CSV folders, plus a report (travel, tilt, speeds, margins) |
| `scripts/g1_limits.py`, `scripts/deploy_logs.py`, `scripts/rotations.py`, `scripts/sonic_tree.py` | Shared helpers |

`configs/` holds the motion ladder, the G1 joint table (limits with sources) and the deploy gains.
`templates/trial_sheet.csv` is the per-trial record. Tests (offline, a few seconds):
`$SONIC_ROOT/.venv_sim/bin/python -m pytest tests`.

Weights, motions and logs are not in the repo (SONIC's weights are under the NVIDIA Open Model License).
