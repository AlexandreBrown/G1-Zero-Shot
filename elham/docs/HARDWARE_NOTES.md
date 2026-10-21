# Hardware notes and sources

Where each hardware fact in these docs comes from. "Code" means GR00T-WholeBodyControl @ `b042411`,
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`. Issue numbers are on
https://github.com/NVlabs/GR00T-WholeBodyControl/issues (read 2026-10-21).

## The deploy program

| Fact | Source |
|---|---|
| Control 50 Hz, LowCmd writer 500 Hz, input 100 Hz, planner 10 Hz | code `src/g1_deploy_onnx_ref.cpp:13-29` |
| On connect: releases Unitree's motion service, then INIT ramps all joints to `default_angles` over 3 s at policy gains | `:2280-2287`, `:2189`, `:2762-2794` |
| `]` starts the policy on frame 0 of the current motion (paused); `T` plays; at the end the reference returns to frame 0 | `:2303-2312`, `:3405-3428` |
| Stop (`O`, `Select`, any error) = one damping command kp 0 / kd 8, then exit; no SIGINT handler | `:2719-2752` (grep finds no signal handler) |
| Self-stops: LowState older than 500 ms; joint velocity > +35 rad/s (one-sided, skipped in sim); failed step | `:299`, `:2797-2812`, `:2865` |
| Over-temperature (≥ 90 °C) only warns | `:2876-2925` |
| No action clipping (training clipped at ±20) | `:3137-3167`; training `action_clip_value: 20.0` |
| Gains: Kp = armature·(2π·10 Hz)², Kd = 2·2·armature·ω (×2 on ankle pitch/roll and waist roll/pitch); action scale 0.25·effort/Kp; hip pitch and roll use 7520-22 values ("old is 7520_14" for hip pitch) | `include/policy_parameters.hpp:40-207`; numbers in `configs/sonic_deploy_constants.yaml` |
| The state history starts empty at `]` (the logger is fed only in CONTROL); missing entries are zero, and a zero quaternion gives gravity (0, 0, +1): for 9 ticks the policy sees a zero-padded history | `src/state_logger.cpp` `GetLatest` / `makeZeroEntry_`; `g1_deploy_onnx_ref.cpp` `GatherHisGravityDir`, CONTROL case |
| At the end of a motion: frame 0, paused, heading re-anchored to the robot (`reinitialize_heading_`) | `g1_deploy_onnx_ref.cpp:3413-3426`, `UpdateHeadingState` `:559-585` |
| `]` / `T` pressed early are remembered (`operator_state.start/play` never cleared) | `include/input_interface/keyboard_handler.hpp:306,408-414`; WAIT_FOR_CONTROL `:3858-3869` |
| The remote is read only after `@` (manager mode); after `@` keyboard `]`/`T` are ignored, `O` stays global; in gamepad mode `B` = frame 0, `L1`/`R1` = previous/next motion (so `L2+B`, `L1+A` move the reference) | `include/input_interface/interface_manager.hpp:81-171`, `gamepad.hpp:251-268,434` |
| `F` (temperature report) and the 90 °C warning work only in CONTROL | `g1_deploy_onnx_ref.cpp:3873-3929`, `:2876-2925` |
| "robot data late" (LowState > 50 ms old) and motor-error warnings are spoken (text-to-speech), not printed | `:298`, `:3020`, `src/audio_thread/audio_thread.cpp` |
| The torso IMU (`rt/secondary_imu`) is required from the first CONTROL tick; without it the program stops into damping | `:2847-2850` |
| Motion folders are loaded in `directory_iterator` order (unsorted); N/P switch with no blending | `include/motion_data_reader.hpp:685-692`, `include/input_interface/keyboard_handler.hpp:367-427` |
| `deploy.sh` turns unknown options into the interface name; `real` falls back to the first interface | `deploy.sh:342-346`, `:148-171` |
| TensorRT must be exactly 10.13 (x86) / 10.7 (Orin); a wrong version gives silently wrong outputs | `docs/source/getting_started/installation_deploy.md:8-26`, troubleshooting #9 |
| CSV logs (`--enable-csv-logs`): 50 Hz, CONTROL state only, joints in hardware order, `q` absolute | `src/state_logger.cpp:30-130, 280-330` |

## The robot

| Fact | Source |
|---|---|
| SONIC targets the black-pelvis G1 with 7520-22 hip-pitch motors; older (white/silver/grey pelvis) robots have 7520-14 hip pitch and are reported to lean and overheat; the maintainers suggest fine-tuning with the old motor's parameters | issues #174 (maintainer, 06-04 and 06-08), #217, #225 |
| Hip pitch Kp/Kd × 0.75: "no problems, slightly worse tracking" (one user, against hip heating; the robot turned out to be a new one) | #85 (DavidDobas, 2026-03-16) |
| Revision by `mode_machine` (app: Device → Data → Robot → Machine Type; printed as `G1 type:`): 1–6, 13, 14 = hip pitch 14.3:1 (old; 1–3 also hip roll 14.5:1); 10, 11, 12, 15, 16, 18 = 22.5:1 (new; G1+ `g1_pl` is mode 18 with 14.3). Unitree's URDFs for the two differ only in the hip-pitch limits (88 Nm / 32 rad/s vs 139 Nm / 20 rad/s) and the pelvis mesh; masses and IMU frame are the same | unitree_ros `robots/g1_description/README.md`; URDF diff of `g1_29dof_rev_1_0` vs mode 15 |
| SONIC trains the old body (rev_1_0) with the new hip-pitch motor; it randomises joint zero offsets by only ±0.01 rad and does not randomise gains, armature or torque limits | `gear_sonic/envs/manager_env/robots/g1.py:244-275`; release `config.yaml` events |
| Old vs new hip-pitch motor in the preview: standing lean changes by < 0.3°, hip pitch needs 1–15 Nm standing | this repo, `preview_mujoco.py --revision old/new` (2026-10-21) |
| Motor constants: 7520-14 armature 0.010177520, 7520-22 0.025101925; SONIC's MuJoCo scene uses armature 0.01 on every joint | `policy_parameters.hpp:40-43`; `scene_43dof.xml` (checked with MuJoCo) |
| Ankle roll reaches 90–113 °C in 5–10 min under SONIC, even standing | #174 |
| Policy shakes when active in the air; start only with feet loaded | #106; `docs/source/user_guide/teleoperation.md:53` |
| Needs 29 DoF with the waist unlocked | `include/robot_parameters.hpp:108-124`; #57/#61 maintainer replies |
| Offboard Ethernet: occasional LowState drop-out → damping stop | #226 (open) |
| Router in the path → "robot data late", growing latency; use a direct cable | #265 |
| Factory PC2 image is JetPack 5.1.1; onboard SONIC needs JetPack 6.2 (reflash, robot opened) | #147; `docs/source/references/jetpack6.md` |
| Deploy heading differs from training when the torso is tilted | open PR #285 (2026-10-15), linked to #269 / #231 |
| Debug mode `L2+R2` (hanging, damped); damping `L2+B` (V1.0.4) or `L1+A` (V1.0.2) | G1 user manual (MyBotShop PDF, firmware V1.0.2/V1.0.4, p.25/p.41); unitree_rl_gym `deploy/deploy_real/README.md` |
| IPs: motion control unit 192.168.123.161, PC2 192.168.123.164 | G1 manual p.9; LeRobot G1 docs |
| No hardware e-stop button on the G1 | NVIDIA GR00T real-robot safety page |
| Unitree SDK protective thresholds: casing > 85 °C, winding > 120 °C, SOC < 20 % | `unitree_sdk2/include/unitree/robot/g1/common/terminations.hpp` |

## Joint limits

`configs/g1_joint_limits.yaml` (hardware order) merges Unitree's rev-1.0 MJCF and URDF shipped in SONIC with the
Isaac training limits. The two disagree at the hips:

| Joint | Isaac training | Unitree rev-1.0 model | Used by the gate |
|---|---|---|---|
| hip pitch | 139 Nm | 88 Nm (7520-14) | 88 Nm |
| hip roll | 139 Nm | 139 Nm | 139 Nm |
| ankles, waist roll/pitch | 50 Nm | 50 Nm (MJCF) / 35 Nm (URDF) | 50 Nm (the URDF value is reported, not gated: NVIDIA's released base exceeds 35 Nm at the ankle on a gentle squat in simulation) |

If the robot has 7520-22 hip-pitch motors (black pelvis), the real hip-pitch limit is 139 Nm and the gate is
conservative there. The deploy program's own MuJoCo simulator caps hip pitch **and** roll at 88 Nm
(`gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml:241`).

## Unverified — check on the robot (while hanging)

- What the motors do when commands stop without a damping packet (phase-A kill test).
- That `O` and `Select` leave the joints soft (phase-A tests; the last LowCmd rows in `hw/` should show kp = 0).
- The `G1 type` (mode_machine) value of an unlocked 29-DoF robot (write it down on day 1).
- The debug-mode entry sequence on this firmware (`L2+R2`, or `L2+R2`, `L2+A`, `L2+B`).
- The power-off sequence (short press, then hold > 2 s, as for power-on) and that the robot goes fully limp.
- Exiting debug mode requires a reboot.
- Behaviour without Dex3 hands (#222 unanswered).
- Button combinations on firmware newer than V1.0.4.
- Battery state of charge over DDS (the hg `BmsState_` message exists; its topic name is unknown): read the app.
