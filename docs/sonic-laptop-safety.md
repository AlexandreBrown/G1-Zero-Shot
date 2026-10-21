# Scope and commissioning limits

This is commissioning code, not a validated safety controller. It reuses NVIDIA's
pretrained SONIC encoder, decoder, planner and Unitree DDS transport. It modifies
an isolated deployment copy under `.runtime/sonic/deploy`; the upstream checkout
is untouched. GPU engines must be built on the laptop that will run inference.
No Mila or Orin service is needed. Hand publishers are initialized for damping on a latched stop; no finger motion commands are published.

## What the controls actually do

- **Open panel:** validate/snapshot the library; no DDS or robot process yet.
- **Suspended setup:** start the guarded process, load/build models, release the existing Unitree
  motion controller, then ramp to the supplied idle reference. **This moves motors.**
- Setup holds the idle joint targets without running the neural controller. The
  final command gate must see joint error ≤0.12 rad and joint speed ≤0.3 rad/s
  continuously for two seconds before enabling the next step. Existing state,
  command, limits and hand-envelope checks remain active.
- **Feet down / start:** manually lower both feet to ground support and confirm;
  start neural inference from the idle reference. No automatic foot-contact or
  balance check is claimed. Playback and keyboard enable remain separate actions.
- **Play once / folder:** use the upstream motion reader in sorted order; queue
  progression uses acknowledged controller state, not a timer. At completion,
  stop advancing the reference and keep inference running at its final standing
  frame. Do not reset heading, jump to frame zero, exit, or remove stiffness.
- **Clear remaining queue:** finish the current clip, then hold its final frame.
- **Keyboard:** one-way switch from reference hold to upstream planner. Space plus
  W/S/A/D requests slow forward/back/left/right movement at 0.15 m/s. Release or
  focus loss requests IDLE, superseding a pending move. Key-release uses a 25 ms
  X11 autorepeat filter. Each timestamped movement request expires after 150 ms;
  the next input iteration forces planner IDLE without momentum decay. Stale
  requests cannot renew that deadline. If the input loop itself stalls >150 ms,
  the writer latches damping. Enabling keyboard clears held keys. This is a request, not a guaranteed physical speed
  or instantaneous stop. Return to clips requires a supported restart.
- **Esc / red EMERGENCY STOP:** the writer latches body damping (`kp=0`, `kd=8`, feedforward torque
  zero) and both-hand damping (`kp=0`, `kd=0.1`, desired velocity and feedforward torque zero); no rearming in that session. This is **not torque-off** and **not standing**.
  The gantry must support/catch the robot. Close likewise latches damping before
  asking whether the robot is supported and the process may terminate.

A GUI crash leaves the independent controller writer alive to latch damping on
lease expiry. An inherited process lock blocks a second panel until that controller
exits. A controller crash, power loss, DDS loss, frozen OS, or severed Ethernet
cannot be made safe by this app. Keep the physical stop and mechanical support.
Do not kill a live controller to resolve a lock while the robot is unsupported.
Session directories contain its PID and log for recovery after supporting the robot.

## Motion acceptance profiles

The UI's **Acceptance limits** editor applies only before setup. Its speed, travel,
tilt, velocity-consistency and hand-clearance settings are shared by importing and
library validation. Changing a profile requires a fresh library validation; each
session saves the exact values. See [ranges and CLI use](laptop-advanced.md#acceptance-limits).
The limits below describe default reference checks and fixed runtime checks.

## Enforced checks

- 29 body joints, using the upstream hardware/IsaacLab permutation explicitly.
- Reference and commanded joint limits from upstream `g1_29dof.xml`, with 0.03 rad
  inward margin. Reject violations rather than silently clipping learned actions.
- Runtime checks include finite values, measured joint bounds, measured speed
  <=8 rad/s, desired speed <=4 rad/s, target step <=0.20 rad per publication,
  nonnegative gains (`kp<=400`, `kd<=20`), and feedforward torque <= model limits.
  These do **not** cap the complete PD-generated motor torque or validate dynamics.
- Writer latches damping for command/state age >100 ms or UI heartbeat >300 ms.
  Polling adds up to 20 ms to the heartbeat threshold; OS scheduling can add delay.
  WAIT_FOR_CONTROL refreshes its standing command at control cadence.
- Dex3 publishers are reserved for latched damping only. The damping path bypasses
  normal gesture buffers and position smoothing. Finger open/close calls are removed.
  Hand mode uses the existing driver's enabled joint mode and its default damping
  gain of 0.1. This has been checked at command construction, not on physical hands.
  Disable competing hand command publishers before commissioning; otherwise commands
  can conflict. Damping does not guarantee collision avoidance or retain a grasp.
- The input thread remains responsive during planner startup. Movement requests are
  rejected until startup completes; a five-second startup timeout latches damping.
  Setup handover exceptions also latch the guard instead of terminating the process.
- Before the controller finishes loading, the UI reports stop as requested, not
  acknowledged. A cancelled lease prevents the subsequent body-control handover.
- A preliminary geometric envelope checks both commanded and measured poses:
  120 mm radius spheres offset 80 mm forward of each wrist against torso, head,
  pelvis, thighs, shins, and each other, requiring 25 mm modeled clearance.
  This uses body kinematics from upstream, **not measured Dex3 finger geometry**.
  It does not cover every link pair, accessories, obstacles or swept trajectories.
  It is not proof of collision freedom. Confirm it encloses your real hand/finger
  configuration and calibrate against U4 before hardware commissioning.

The conservative envelope rejects the stock default arm pose in local tests.
Do not weaken it merely to make a clip run. Supply a validated standing reference
with hands clear of the body, and put the supported robot into a clear starting
configuration before arming. No generated “safe wave” or unchecked standing motion
is shipped. The UI's preset buttons are your first four validated clips.

## Motion library contract

Each clip is a directory of headered CSV files: `joint_pos.csv`, `joint_vel.csv`
(29 columns, **IsaacLab order**, radians/radians per second), `body_pos.csv`
(world xyz per body), `body_quat.csv` (world **wxyz** per body), and `metadata.txt`
with `Body part indexes:` followed by bracketed, space-separated indices. Root
index 0 must come first. Use SONIC's native **50 Hz** data; NPZ, BVH and video need
upstream retargeting/conversion first. Extra data files are copied unchanged.

The initial idle clip must be stationary and its lower body within 0.15 rad of
SONIC's nominal stance. Other clips must have stationary first/last 0.5 seconds,
match idle endpoints within 0.10 rad and root pose within 5 cm / 0.10 rad.
Root tilt is restricted to 0.20 rad and horizontal excursion to 15 cm.
Reference speed is limited to 4 rad/s; position/velocity disagreement >1 rad/s
is rejected. Terminal joint velocities are zeroed in the session snapshot.
These are deliberately narrow commissioning limits, not general SONIC abilities.

Holding a learned standing reference avoids intentional shutdown after a clip;
it cannot guarantee balance or prevent a fall. Validate each motion, transition,
keyboard start/stop, timeout and stop behavior in MuJoCo before supported hardware.

## Validation and known limits

CPU tests compile and exercise the actual C++ publish gate (NaN, stale data,
limits, target discontinuity, heartbeat expiry and latching), reject bad motion
files, and compare envelope forward kinematics with MuJoCo. Source patch anchors
and hashes reject incompatible upstream files. An offline virtual-display UI test also checks button gating, deadman Space,
playlist progression and latched stop without launching a controller. Full C++ deployment compilation,
TensorRT inference, DDS/network latency, physical G1/Dex3 clearance and stability
must still be checked on the target setup. The source hash pin is intentional;
updating upstream requires reviewing the patch, not bypassing the lock file.

The panel uses a private local directory for commands/acknowledgements, with no
remote command listener. Debug ZMQ output is upstream's normal read-only stream.
A future VLM/VLA integration should submit approved high-level actions through
these same state/limit checks, not directly publish joint commands or latent tokens.
