# First G1 robot runs, 2026-11-04 (gpt_astra + SONIC, GPT gpt-6.1-terra deciding)

Compact copies of the run folders (`python -m gpt_astra.tools.compact_run <run> <out>`): every log and GPT input
frame as recorded; `robot_state.csv.gz` (joints, IMU, speeds, torques) and `robot_state_cmd.csv.gz` (SONIC's joint
targets and gains), both 50 Hz; `camera.mp4` with `camera_frames.csv` (MP4 frame k = row k: laptop receive time and
the robot's capture time). All times are the laptop's wall clock. Left out: the staged motion library (`motions/`,
`generated/`), which the code regenerates.

Reconstruct: `events.jsonl` (GPT requests/replies with latency, controller acks, catches, let-gos, limits),
`trace.jsonl` (every follow tick: balloon estimate, mode, intent, walk step and its reason; controller mode changes;
camera frames). Replay the joints in MuJoCo: `gunzip -k robot_state.csv.gz`, then
`gpt_astra/tools/robot_to_sim.py robot_state.csv` and `gpt_astra/replay.py robot_state_replay`.

| Run | Task | What happened | Change made after it |
|---|---|---|---|
| 144051 | catch_balloon | Camera and GPT preflight OK; operator stopped at the suspended-setup prompt | |
| 144302 | catch_balloon | Controller could not start: CUDA unavailable after the laptop slept (kernel Xid 154) | `nvidia_uvm` reloaded; runs under `systemd-inhibit` |
| 144749 | catch_balloon | Guard latch at setup: hand/body clearance < 25 mm (hanging hands on the thighs) | clearance check removed |
| 150031 | catch_balloon | Guard latch when SONIC took over: target step > 0.20 rad, left hip (robot leaning 11 deg) | |
| 152755 | catch_balloon | Same, left knee (leaning 8 deg); standalone recorder log `20261104-152714` | guard motion limits removed |
| 153507 | catch_balloon | First run standing on SONIC; GPT held 20/20 (balloon on the floor / at the edge, "outside reach"); 8 replies older than 2 s rejected | `tasks/catch_balloon/reach_toward.txt`, `--max-decision-age 3` |
| 154312 | catch_balloon | With reach_toward.txt: 13 reaches (left 2, right 7, both 4) | |
| 160314 | follow_balloon | Stopped before actuation: camera stale while the recorder started | fixed (camera read during the wait) |
| 160724 | follow_balloon | GPT catch at 7 s; balloon detected in 20% of ticks (cord in front split the red disk); sidesteps < 1 s; 1 m fence stopped walking at 21 s | trial: circle test 0.45, fence 5 m, keep walking 1 s |
| 161329 | follow_balloon | Operator stopped after the suspended checks | |
| 161620 | follow_balloon | Detected 85%; 0.2 m/s steps stepped in place; never "still" (2 cm); 5 m fence (dead reckoned) hit at 38 s | trial: fence off, fast walking only, stillness off |
| 162812 | follow_balloon | First robot grab (pose 1_7 at 0.42 m) at 59.6 s, run ended at its 60 s limit; backed off twice from a balloon < 0.3 m | `--follow-seconds 300` |
| 163511 | follow_balloon | Operator stopped after the suspended checks | |
| 163750 | follow_balloon | 3 grabs (2_2, 2_7, 2_4) and 3 releases when pulled; runner ended during the third release | recorder exits with the runner |
| 164410 | follow_balloon | Walked backward: the balloon and red labels on equipment were fitted as one huge circle reading 0.2 m away (526 of 1083 detections); 6 grab attempts skipped | coverage check (fit must be mostly red) |
| 165249 | follow_balloon | 3 grabs, 2 releases (before the coverage check) | |

The trial calibration (`gpt_astra/calibration/robot_trial.json`) is unmeasured; see its `trial_changes`.
