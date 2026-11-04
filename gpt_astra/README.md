# GPT Astra / Sol → SONIC

Images + task → **selected GPT model** → bounded arm-tap selection → pretrained SONIC.
No GR00T model or fine-tuning. Reuses this repo's guarded controller and joint checks.

**Experimental:** the catching task offers stationary left, right or two-arm
reaches, hold and stop. Fingers and walking are not controlled yet, so a grasp
or catch is not guaranteed. Replies older than 2 seconds are discarded.

## 1. Install — from the G1-Zero-Shot repo root

Ubuntu, Python 3.10+, git-lfs. Fetches the pinned sibling WholeBodyControl checkout
if missing; an incompatible existing checkout is rejected without overwriting it.

```bash
bash gpt_astra/setup.sh
source gpt_astra/.venv/bin/activate
```

## 2. CPU smoke test — no GPU, key or robot

```bash
python gpt_astra/run.py --backend mock --policy fake --auto-confirm --max-decisions 4
```

Exercises the complete stage/command/acknowledgement loop. Controller and camera
are **synthetic**, with no SONIC inference, physics or balloon-contact simulation.

## 3. Actual GPT + fake controller

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY
export OPENAI_MODEL=gpt-6.1-sol  # Or gpt-6-astra
python gpt_astra/run.py --backend mock --max-decisions 3
```

Press Enter at the prompts. Optional: `--image /path/to/photo.jpg` for a static-image
API test. Images/task go to OpenAI; requests are billable. Your API project needs
access to the selected model. No model fallback; keys are never written to logs.

**Model selection:** `--model gpt-6-astra` overrides `OPENAI_MODEL`; when neither is
set, the default remains `gpt-6-astra`. The `astra` policy name uses either model.
Both accept images and structured commands with low reasoning effort. Sol costs
less; compare `latency_s` and stale-decision rejects in logs to measure actual speed.
Other model IDs must support Responses, images, strict JSON and `reasoning.effort=low`.

## 4. GPU laptop — real SONIC in MuJoCo

```bash
export CUDA_HOME=/usr/local/cuda
bash gpt_astra/setup.sh --gpu
(cd ../GR00T-WholeBodyControl && bash install_scripts/install_mujoco_sim.sh)
source .runtime/sonic/env.sh
gpt_astra/.venv/bin/python gpt_astra/run.py --backend sim --max-decisions 3
```

The runner opens a **live camera window**: verify the image updates, then press
Enter in the terminal. No images within 30 seconds causes an error; a stream stall
or closing the preview stops the run. Use a graphical desktop (`setup.sh --gpu`
installs Tk). Add `--preview` to a mock run to try the window on CPU.

The runner starts the simulator, camera publisher and guarded SONIC. At the
lowering prompt, press **9 in the viewer**, then Enter in the terminal. The robot
starts in a standing pose; SONIC holds the ready reference before GPT decisions.
Once the loop starts, a red sphere spawns slightly above the head and alternates
between dropping and moving toward the robot. It enters the downward-facing ego
camera view as it falls. MuJoCo gravity and collisions determine its motion. An
attempt resets after ground contact or five seconds after a detected two-hand
catch; SONIC continues controlling the robot during the reset.
[Preview the simulated scene](examples/catch_balloon_preview.mp4) (CPU test with a
fixed robot pose; GPT and SONIC inference were not running).

## 5. Real G1 — keep gantry and physical stop available

Follow the [camera server quick start](../camera_server/README.md) to set up the
camera and synchronize the robot and laptop clocks. Close competing controllers.
Replace the interface (`ip -br addr`) and camera IP:

```bash
source .runtime/sonic/env.sh
gpt_astra/.venv/bin/python gpt_astra/run.py --backend robot \
  --interface enp3s0 --camera-host 192.168.123.164 --camera-key ego_view \
  --task-name catch_balloon --max-decisions 20
```

1. Confirm the live camera window with Enter; API preflight then runs before actuation.
2. **Feet suspended, limbs clear:** Enter starts the ready-pose ramp.
3. Joint error ≤0.12 rad and speed ≤0.3 rad/s must hold for 2 seconds.
   After **CHECKS PASSED**, Enter advances to lowering; autonomy stays disabled.
4. Lower onto both feet; Enter confirms ground support and starts the loop.
   Ground contact is **operator-confirmed**, not automatically sensed.
5. On completion SONIC holds ready. Support the robot, then Enter to exit.

**Ctrl+C, SPACE or `q` (no ENTER needed) requests latched damping, which can let the robot collapse.**
After faults, the controller stays alive until support is confirmed. A runner crash
expires its native lease. Never kill the surviving controller while unsupported.
Read the existing [guard limitations](../docs/sonic-laptop-safety.md); initial
clearance checks may reject the stock hanging pose. Do not bypass failed checks.

Follow mode (`--follow --task-name follow_balloon`, Claude choosing the intent, a local loop walking and grabbing)
refuses to run on the robot until the controller build declares `reference` and `fine-move` (CAPABILITIES) and
`gpt_astra/calibration/robot.json` is measured. Steps and tools: [ROBOT_FOLLOW_CHECKLIST.md](ROBOT_FOLLOW_CHECKLIST.md).

## Logs / extensions / tests

Each run prints its persistent `gpt_astra/runs/…` folder: `events.jsonl` (latency,
commands, rejects), `frames/` (GPT input images), `task/`, `controller.log`,
`simulator.log`, `latest.jpg` and staged references. The folder is ignored by git.
Default references are generated and validated; the repo's invalid bundled idle is unused.

Everything carries the laptop's wall-clock time (`time`, seconds), so a run can be reconstructed:
- `events.jsonl`: GPT requests (`observation`: images, tracker, previous action) and replies (`decision`: action,
  confidence, reason, `latency_s`), controller acks, catches, let-gos, limits.
- `trace.jsonl`: every follow-loop tick (balloon estimate, mode, intent, still, step and its reason, dead-reckoned
  position), every controller mode change (state, clip, planner, fault, reason) and, on the robot, every camera
  frame (`camera/NNNNNN.jpg`, with the robot's capture time).
- Robot runs: `robot_state.csv` (joints, IMU, speeds, torques) and `robot_state_cmd.csv` (SONIC's joint targets and
  gains) at 50 Hz, recorded automatically from before actuation until the controller has stopped; `controller.log`.

Sim runs also save `simulation.mp4` (third-person video), `trajectory.csv` and
`trajectory_meta.json`. Replay a trajectory on the same laptop:

```bash
../GR00T-WholeBodyControl/.venv_sim/bin/python gpt_astra/replay.py gpt_astra/runs/RUN_FOLDER
```

Robot runs have no trajectory. Record one in a second terminal, started before `run.py` and stopped with Ctrl+C
after it; it only listens (joints, IMU orientation, speeds, torques, and the controller's targets and gains in
`<name>_cmd.csv`, at 50 Hz):

```bash
SIMPY="env PYTHONPATH=../GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python:. ../GR00T-WholeBodyControl/.venv_sim/bin/python"
$SIMPY -m gpt_astra.tools.record_lowstate --interface enp58s0 --out gpt_astra/runs/robot_logs/$(date +%Y%m%d-%H%M%S).csv
```

Later, turn the log into a trajectory and replay it in MuJoCo. The log has no base position, so the lower foot is
kept on the floor (gantry hanging looks like standing, walking replays in place):

```bash
$SIMPY -m gpt_astra.tools.robot_to_sim gpt_astra/runs/robot_logs/LOG.csv
$SIMPY gpt_astra/replay.py gpt_astra/runs/robot_logs/LOG_replay --speed 0.5
```

The robot's camera server (GR00T `composed_camera`) JPEG-encodes RGB images with OpenCV, which assumes BGR, so a
red balloon arrives blue. `run.py --backend robot` and `tools.check_camera` swap red and blue back by default
(`--no-camera-swap-red-blue` turns it off if the server is ever fixed); check at CAMERA CHECK that red looks red.

The default prompt is [tasks/catch_balloon/task.txt](tasks/catch_balloon/task.txt). To add a
task, create `tasks/NAME/task.txt`, then use `--task-name NAME`; optional
`tasks/NAME/sim.py` can define `prepare_scene(source_scene, run_directory)` and
`TaskSimulation(model, data, start_file)` with a `step()` method. Optional
`tasks/NAME/motions.py` can define `generate_library(destination, model)`.
Without motion definitions a new task holds the standing pose.
`--task 'custom instruction'` overrides the selected prompt for one run.

```bash
python gpt_astra/run.py --prepare-only /tmp/astra-motion-library
python -m unittest discover -s gpt_astra/tests -v
```

`--library PATH`: provide a stationary `ready/`, compatible SONIC CSV motion
directories and `actions.json` mapping their names to descriptions. All existing
limit/clearance/endpoint checks apply. Walking and finger commands are not exposed.
Custom upstream path: `SONIC_UPSTREAM` during setup, `--upstream` during runs.

Tests cover mock/API-wire/camera/kinematic/native command paths. Optional MuJoCo
and C++ tests need their packages/headers. Actual API access, TensorRT, balance
and balloon contact still need target-laptop validation.
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) /
[Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) API references.
