# follow_balloon on the real G1: checklist

Follow mode works in simulation (21/21 grabs, no body contact, Claude deciding). On the robot it refuses to start until
the controller build declares the commands it needs and `gpt_astra/calibration/robot.json` has every measured section
dated. Do the steps in order; each one writes or checks one part. Commands run from the `G1-Zero-Shot` folder.

`SIMPY` below is the upstream sim interpreter, which has MuJoCo and the Unitree SDK:
`PYTHONPATH=../GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python:. ../GR00T-WholeBodyControl/.venv_sim/bin/python`

## 0. Before anything moves

- **Controller (Alex).** The robot build lacks the `reference` command and the fine walking speed. The change is drafted
  in `gpt_astra/robot_patch/` (patch, README, tests pass). Alex must also decide its open question (b): with the current robot
  guard, the hand/body clearance check most likely latches about 0.6 s after entering planner (walking) mode, with or
  without the patch, because SONIC's planner stance hangs the hands against the thighs (-33 to -59 mm by the guard's own
  model, 25 mm required). No walking test can run until that is settled: the runner refuses `--follow` and
  `--measure walk` on the robot until the build's CAPABILITIES file declares `planner-stance` (Alex adds it), the
  two commands, and walking speeds equal to `robot.json`'s. Also note that every sim result ran under a much looser
  sim guard (see the patch README); behaviour under the strict robot guard is untested.
- **Machine.** Original charger, laptop cool, no video player open. After any standing session:
  `gpt_astra/.venv/bin/python -m gpt_astra.tools.timing_report gpt_astra/runs/<run>` must say OK (robot state p95 under
  7 ms, observation to command p95 under 1 ms, no guard latch other than the normal end). Ten clean Starts in a row.

- **Hands.** The robot build sends the Dex3 fingers nothing while running (the stock per-cycle finger command and
  the open/close keys are removed) and only damping (kp 0, kd 0.1, no torque) after a guard fault. Close every other
  program that could command the hands (Unitree app, hand services). Gloves are fine: keep them thin (the guard's
  25 mm hand/body clearance is computed for bare hands) and not red (the balloon detector looks for red).

## 1. Camera and balloon (robot standing still; nothing moves)

Use a red balloon about 15 cm across; tape its diameter. Mark 9 floor spots from the point under the pelvis (0.4 /
0.55 / 0.7 m ahead, 0.12 m left / centre / right) and measure the pelvis height.

```bash
gpt_astra/.venv/bin/python -m gpt_astra.tools.check_camera --camera-host <robot camera host> \
    --balloon-diameter <m> --pelvis-height <m> --red-object --write
```

Pass: no detection in the empty view or on red clothing; every spot within 3 cm after the fit; the suggested
thresholds find the balloon in at least 90% of the spot frames. It writes the camera pose and field of view and the
balloon thresholds (colour, circle fit, height range) into `robot.json`, dated. The fit cannot see a grid that is
misplaced as a whole (the camera pose absorbs it), so measure the grid's origin under the pelvis carefully.

## 2. Grab poses (gantry, robot standing; arms only)

Terminal A records the joints: `$SIMPY -m gpt_astra.tools.record_lowstate --interface <robot NIC> --out joints.csv`.
Terminal B plays 10 grab poses (2 reaches x waist yaw -0.32..0.32 rad), each held 2 s:

```bash
gpt_astra/.venv/bin/python gpt_astra/run.py --backend robot --interface <robot NIC> --task-name static_balloon \
    --policy none --measure grab
$SIMPY -m gpt_astra.tools.grab_report gpt_astra/runs/<run> --joints joints.csv --write
```

Pass: every pose has a cradle region (the hands can hold a balloon of the measured size). It writes where each reach's
hands meet and how well the waist twist tracks. The finger joints are not recorded (they are taken as straight), so
the manual check is the real one: a person holds the balloon between the hands of a held grab pose; it must neither
pop nor slip out. The tool refuses to write simulation runs into `robot.json`. In simulation this no-balloon
measurement put the cradle centres ~2.5 cm closer than where 100 grabs of a real balloon succeeded (reach 2: 0.38-0.52 m,
centre ~0.445; the tool said 0.418): after the first follow runs, move `grab.meeting_x_m` out by what the robot's own
held/missed grabs show.

## 3. Walking (needs step 0's controller; floor, gantry slack, balloon on a stand 0.9 m ahead, centre 0.15 m below the pelvis)

```bash
gpt_astra/.venv/bin/python gpt_astra/run.py --backend robot --interface <robot NIC> --task-name static_balloon \
    --policy none --measure walk
gpt_astra/.venv/bin/python -m gpt_astra.tools.walk_report gpt_astra/runs/<run> --write
```

It walks 6 kinds of short steps 3 times each (fast and fine forward/back, sideways). Before every probe the robot
walks back to that probe's start spot (closed loop, like the follower); a probe toward the stand starts only if its
commanded distance plus a generous coast keeps the balloon at least 0.3 m away, and stops early when the balloon comes
within that margin or is lost. The balloon's apparent shift gives the robot's position for the 1 m fence. Pass: all six
moves measured, none stalls every time, none walks farther than commanded plus its coast (the fence relies on that).
It writes how far each move keeps going after the last request (the follower's stop margins) and how far the robot
settles back when it leaves walking mode.

## 4. Follow

```bash
gpt_astra/.venv/bin/python gpt_astra/run.py --backend robot --interface <robot NIC> --follow \
    --task-name follow_balloon --policy claude --follow-seconds 60 --min-confidence 0.6
```

Or GPT deciding: `export OPENAI_API_KEY=$(cat ~/.config/gpt_astra/openai_key)` in your terminal, then the same
command with `--policy astra --model gpt-6.1-terra` (reasoning effort low; in a 12-observation replay every effort from
none to xhigh made the same 12/12 choices in ~2 s).

Start with 60 s runs, gantry slack, the operator at the physical stop. Safety limits from `robot.json` (`safety`): no
walk longer than 4 s, none beyond 1 m from the start, stop within 0.3 s of losing the balloon. The 1 m is dead-reckoned
at the commanded speeds plus coasts, an upper bound that in simulation read 2.8x the real distance, so expect the
robot to stop following after roughly 0.4 m; raise `fence_m` only after watching it. If Claude fails twice in a row, or the camera is stale for 2 s, the loop pauses
with the robot standing (or holding), never in damping; SPACE or `q` (no ENTER) requests latched damping as before. A controller
fault still latches damping, as it should. Check each run with `timing_report` and
`python -m gpt_astra.tools.analyze_follow gpt_astra/runs/<run>`.
