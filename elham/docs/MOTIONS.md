# Motions: what to run on the robot, in which order, and why

All motions are ASAP clips retargeted to the G1 and converted to SONIC's format. They are 30 fps PKLs; the
exporter turns them into 50 Hz deploy CSVs. The ladder is in `configs/motion_stages.yaml`, and
`scripts/export_motions.py` builds it.

## Ladder

Everything here is for NVIDIA's released SONIC weights, in two sessions. Root travel decides the gantry: with a
fixed hoist point, run only motions that travel ≤ 0.5 m; longer travel needs a trolley that a person moves along
with the robot.

| Session | Policy | Motions (travel) | Why this order |
|---|---|---|---|
| 1 | released base | `step_forward_back_lv2` stand test: start the policy, no play | Stack check: the robot stands on the policy, both feet down. Its frame 0 is nearly level, so a lean on the robot shows (next section) |
| 1 | released base | step_forward_forward_lv1 (0.47 m), squat_lv1 (0.30), step_forward_back_lv1 (0.17), kick_lv2 (0.27) | Low joint speed, clean in every simulator, fixed hoist OK |
| 1 | released base | single_foot_balance_lv1 (0.14), step_forward_back_lv2 (0.47), kick_lv3 (0.34), APT_lv1 (0.40) | Retained motions that fit a fixed hoist |
| 2, trolley | released base | walk_lv1-4 (1.8–2.2 m), step_forward_forward_lv2/3 (0.71/0.97), step_forward_back_lv3 (0.62), lebron1_lv1 (1.25), shoot_lv1 (1.95), TigerWoods_lv1 (1.39) | Long travel |
| 2 | released base | Bolt_lv1 (0.41, fixed hoist), lebron2_lv1 (1.06, trolley) | Not in SONIC's training motions; the base tracks them loosely (Isaac success 0.18 and 0.00) |

**Never run without a new simulation gate:**

| Motion(s) | Why |
|---|---|
| SpiderMan_lv2 | The base never learned it. The pelvis goes down to 0.45 m and the reference floats 10–18 cm. |
| All jumps, CR7_lv1-2, Kobe_lv1, single_foot_jump_lv2 | Flight phases of 0.2–0.57 s. The base succeeds on 0 of 510 jump episodes. |
| single_foot_balance_lv4, squat_lv3 | The base already falls on them in simulation. |
| single_foot_balance_lv2 | The reference is broken (root height glitch). |
| Any motion marked ✗ in the table below | Failed the preview's 20 ms screening; run it only after it passes. |

## Expected lean while the policy holds frame 0

After `Start` the robot holds the motion's frame 0, and many frame-0 poses lean on purpose (the clips start
mid-movement). Judge a standing lean against this table, not against zero. Positive = pelvis pitched forward
(nose down), as the pelvis IMU reads it; `analyze_logs.py` prints the robot's value as `lean held`.

| Session | Motion | frame-0 reference ° | held in simulation ° |
|---|---|---|---|
| 1 (stand test) | step_forward_back_lv2 | -5.5 | -1.4 |
| 1 | step_forward_forward_lv1 | +9.8 | +8.0 |
| 1 | squat_lv1 | +14.2 | +11.4 |
| 1 | step_forward_back_lv1 | -11.8 | -7.2 |
| 1 | kick_lv2 | +3.6 | +1.6 |
| 1 | single_foot_balance_lv1 | -6.6 | -4.5 |
| 1 | kick_lv3 | +1.2 | +0.1 |
| 1 | APT_lv1 | +8.9 | +8.3 |
| 2 | walk_lv1 | +12.1 | +9.3 |
| 2 | walk_lv2 | +9.0 | +7.8 |
| 2 | walk_lv3 | +0.3 | +1.4 |
| 2 | walk_lv4 | +0.5 | +0.6 |
| 2 | step_forward_forward_lv2 | +4.5 | +3.9 |
| 2 | step_forward_forward_lv3 | +0.6 | +0.2 |
| 2 | step_forward_back_lv3 | -4.9 | -1.7 |
| 2 | lebron1_lv1 | +3.4 | +3.1 |
| 2 | shoot_lv1 | +5.9 | +5.2 |
| 2 | TigerWoods_lv1 | +8.4 | +5.1 |
| 2 | Bolt_lv1 | +18.8 | +15.4 |
| 2 | lebron2_lv1 | +0.2 | +0.9 |

Simulation: `scripts/preview_mujoco.py --revision old --delay-ms 5 --hold-s 15`, released base, mean over 5–15 s
of the hold; old hip-pitch motor. The robot holds about 70–100 % of the reference lean. With the new motor the
values change by less than 0.3°.

## Evidence per motion

The `base` column reads: **Isaac success / MuJoCo joint error / preview verdict**.

- **Isaac success:** native SONIC evaluation. 170 episodes per new motion and 28 per retained motion; 0 falls in
  every cell shown.
- **MuJoCo joint error:** mean |q − reference| in rad, from our zerosho harness. 10 episodes per motion, deploy
  torque limits (88 Nm hips), 0 falls in every cell.
- **Preview verdict:** the dynamics gate, i.e. `scripts/preview_mujoco.py --delay-ms 20` run the deploy way
  (zero-padded start-up history, end-of-motion re-anchor, old 88 Nm hip pitch, 20 ms loop delay). The reasons are
  given when it fails.

The numbers come from the following files:
- Isaac: `zerosho/outputs/zerosho/hs1_sonic/eval*/`;
- MuJoCo harness: `.../sim2sim/p35_g1deploy/`;
- previews: `.../sonic_g1_deploy/previews/quick_delay20/`; without delay `quick/`; hip-pitch fallback
  `quick_delay20_hip075/`.

| Stage | Motion | s | travel m | ref tilt ° | peak ref speed / limit | base |
|---|---|---|---|---|---|---|
| stage1 | squat_lv1 | 5.24 | 0.3 | 19.1 | 0.09 | 1.00 / 0.093 / ✓ |
| stage1 | step_forward_forward_lv1 | 3.64 | 0.47 | 15.4 | 0.1 | 1.00 / 0.079 / ✓ |
| stage1 | step_forward_back_lv1 | 3.94 | 0.17 | 15.9 | 0.12 | 1.00 / 0.117 / ✓ |
| stage1 | kick_lv2 | 3.34 | 0.27 | 21.1 | 0.16 | 1.00 / 0.088 / ✓ |
| stage2 | single_foot_balance_lv1 | 7.24 | 0.14 | 29.9 | 0.14 | 1.00 / 0.111 / ✓ |
| stage2 | step_forward_forward_lv2 | 4.34 | 0.71 | 11.3 | 0.12 | 1.00 / 0.074 / ✓ |
| stage0, stage2 | step_forward_back_lv2 | 4.64 | 0.47 | 18.7 | 0.13 | 1.00 / 0.110 / ✓ |
| stage2 | kick_lv3 | 4.54 | 0.34 | 37.8 | 0.16 | 1.00 / 0.117 / ✓ |
| stage2 | walk_lv1 | 5.04 | 1.77 | 14.0 | 0.14 | 1.00 / 0.097 / ✓ |
| stage2 | walk_lv2 | 3.44 | 2.0 | 15.0 | 0.17 | 1.00 / 0.095 / ✓ |
| stage2 | walk_lv3 | 4.74 | 2.23 | 9.0 | 0.19 | 1.00 / 0.075 / ✓ |
| stage2 | walk_lv4 | 4.44 | 2.08 | 11.7 | 0.24 | 1.00 / 0.075 / ✓ |
| stage2 | step_forward_forward_lv3 | 4.44 | 0.97 | 16.1 | 0.12 | 0.96 / 0.082 / ✓ |
| stage2 | step_forward_back_lv3 | 5.04 | 0.62 | 13.9 | 0.16 | 0.93 / 0.118 / ✓ |
| stage2 | lebron1_lv1 | 5.04 | 1.25 | 22.0 | 0.27 | 1.00 / 0.099 / ✓ |
| stage2 | APT_lv1 | 7.64 | 0.4 | 18.4 | 0.14 | 1.00 / 0.134 / ✓ |
| stage2 | shoot_lv1 | 4.74 | 1.95 | 14.8 | 0.23 | 1.00 / 0.084 / ✗ torque saturated 0.12 s in a row; torque saturated on 3.2% of steps |
| stage2 | TigerWoods_lv1 | 3.94 | 1.39 | 22.8 | 0.23 | 1.00 / 0.140 / ✓ |
| stage3 | Bolt_lv1 | 4.34 | 0.41 | 25.9 | 0.18 | 0.18 / 0.149 / ✓ |
| stage3 | lebron2_lv1 | 6.94 | 1.06 | 26.1 | 0.24 | 0.00 / 0.096 / ✓ |

How to read the table:
- **Success is not comparable across simulators.** In MuJoCo the base "succeeds" on Bolt and lebron2 (0.8 and
  1.0 with the harness's terminations), while Isaac scores it 0.18 and 0.00 with SONIC's stricter native
  terminations. **On the robot, judge by tracking error and falls (`analyze_logs.py`), not by a success flag.**
- **The preview starts from the stand pose,** as the robot does. The Isaac and MuJoCo evaluations start at
  frame 0. The largest torque of a trial usually comes in the first 0.2 s after `]`.
- **The released base passes every session-1 motion with 20 ms of loop delay.** Its only failure on the ladder is
  shoot_lv1, a session-2 trolley motion: torque saturated for 0.12 s. Without delay it passes everything.
- **Delay tolerance (released base, squat_lv1):** stands at 0, 10, 20 and 30 ms; falls at 40 ms. The real G1's loop
  is a few ms.
- **The old hip-pitch motor.** The previews use 88 Nm and the old rotor inertia at hip pitch, as on our
  grey-pelvis robot. It does not change the lean: held still on a level reference, the pelvis settles at −5.3°
  with the old motor and −5.3° with the new one (`--revision old/new`); on the session-1 motions the robot-minus-
  reference lean differs by at most 0.2°. Hip pitch needs about 1–15 Nm while standing, far below 88 Nm.
- **Grey-pelvis fallback** (hip pitch Kp and Kd × 0.75, with 20 ms delay; released base):
  - passes 19 of 22 motions, including every session-1 motion;
  - fails TigerWoods_lv1 (saturation) and lebron2_lv1 (0.027 rad past a limit);
  - falls on NVIDIA's deep squat (squat_001__A359, not in our ladder).
- **An earlier version of the preview** did not re-anchor the heading when a motion ended. On TigerWoods_lv1 that
  made the robot whip back about 95° in the tail (hip yaw about 30 rad/s), so TigerWoods was wrongly marked unsafe.
  Fixed: TigerWoods peaks at about 6 rad/s.

## Preview videos

On the deploy laptop, `zerosho/outputs/zerosho/sonic_g1_deploy/previews/` holds the base's videos (not in the
repo): `video_base_5ms/` has every ladder motion, and `sonic_weights_session1.mp4` / `sonic_weights_session2.mp4`
are the labelled session reels.

Each video shows 2 s standing on frame 0, then the motion, then 2 s back on frame 0. Make more with
`scripts/preview_mujoco.py`.
