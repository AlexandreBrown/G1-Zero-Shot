# LOCAL validation — 21 October 2026

**Local checks pass; hardware deployment is not yet validated.** No robot connection,
DDS publisher, or SONIC inference process was launched during this validation.
The local machine has no available NVIDIA GPU/CUDA compiler or TensorRT headers,
so full deployment compilation and closed-loop SONIC simulation remain outstanding.

## Checks completed

- 17 automated tests passed: CSV/NPZ input handling, time resampling, joint ordering,
  velocity calculations, malformed input, reference rejection, immutable library
  snapshots, and MuJoCo forward-kinematics agreement.
- The real patched upstream keyboard handler and generated IPC block were compiled
  and executed on CPU. Checked setup gating, nonblocking planner initialization,
  all four directions at **0.15 m/s requested speed**, idle, 150 ms command expiry,
  stale requests and the five-second planner-start timeout.
- The native final command guard was compiled and tested for nonfinite values,
  stale state/commands, target jumps, expired heartbeat, input-loop stalls, limits,
  latched faults, and setup settling/reset behavior.
- Both-hand damping command construction was tested using the actual Unitree
  `HandCmd_` type: seven joints, enabled joint IDs, zero position gain, desired
  velocity and feed-forward torque, with `kd=0.1`. The patched Dex3 driver passed
  C++ syntax checking with the bundled Unitree SDK. This does not test transport,
  firmware acceptance or physical finger behavior.
- Both offline Tk UI suites passed: initialization gating, disabled unsafe actions,
  Space not clicking setup, held-key autorepeat lockout, release/focus-loss idle,
  queue completion, stop, import results and offline arm lockout.
- The actual supplied CSV was imported through the offline UI at **30 FPS**, yielding
  300 frames at 50 Hz and **Needs review**. It was not enabled for robot execution.
- Python compilation, shell syntax and Git whitespace checks passed. The UI was
  visually inspected with Ubuntu's antialiased Tk runtime.

## Problems corrected

1. Upstream planner initialization blocked the input thread, conflicting with the
   150 ms stall watchdog. Initialization now progresses asynchronously while input
   and stop handling remain live; requests to move are rejected until ready.
2. Upstream clamped the requested 0.15 m/s to 0.20 m/s. The guarded build now permits
   the requested 0.15 m/s within the upstream slow-walk mode's 0.1–0.8 range.
3. Keyboard autorepeat could recreate keys held before enabling control. Those
   keys now require release before they can become movement input again.
4. Handover exceptions now latch the guard rather than escape the setup function.
   Heartbeat checks shared by setup and the writer are serialized.
5. Stop now includes a dedicated Dex3 damping-only publish path, independent of
   normal hand gesture buffers; no automatic finger open/close actions remain.
6. Setup activity and joint-settling values are visible in the UI, with a large
   persistent emergency-stop button. GPU build/loading output is read incrementally.

## Supplied motion: move_forward_reach_forward.csv

User confirmed the original six-second duration. The file has 180 rows, consistent
with 30 FPS; the first-to-last sample span is 179/30 = 5.9667 seconds.
SHA-256: `9ca6fcc93937421206998f36e6ab8e7ce767caf1609f0ece6b1b0b24769fbf41`.

| Check | Result |
| --- | --- |
| File structure, finite values, unit root quaternions | Pass |
| Joint-angle limits including margin | Pass |
| Root tilt ≤0.20 rad | Pass: maximum 0.111 rad |
| Horizontal root displacement ≤0.15 m | Pass: maximum 0.081 m |
| Stationary first and last 0.5 seconds | Pass |
| Joint speed ≤4 rad/s | **Fail: right knee 6.652 rad/s around 1.30 s** |
| Position/velocity consistency threshold | **Fail** |
| Hand envelope clearance ≥25 mm | **Fail: minimum −84.6 mm around 3.46 s** |

Negative envelope clearance indicates overlap in the preliminary geometric model;
it is not a measured physical collision. The source was not clipped, slowed or
otherwise changed to pass. This clip remains blocked from deployment. Matching
an actual validated idle reference is also still required.

Detailed results: `artifacts/laptop/kimodo-review/move_forward_reach_forward-audit-30fps.json`.
Converted review copy: `artifacts/laptop/kimodo-review/confirmed-30fps-ui/`.
Earlier reports marked timing unconfirmed are superseded by this report.

## Remaining GPU / robot validation

1. **LOCAL GPU laptop:** rebuild the new `sonic-laptop-v3` controller (README step 8).
2. **LOCAL GPU laptop:** run closed-loop MuJoCo tests with a reference that passes
   all checks: suspended setup, lowering, idle inference, motion completion, keyboard
   release, focus loss, GUI heartbeat loss, and stop. CPU tests do not establish
   physical stopping time, tracking stability or gantry clearance.
3. **LOCAL / ROBOT commissioning:** verify the U4 hand envelopes against the actual
   fingers, remove competing low-level/hand publishers, and measure body and hand
   damping response with the robot supported and a physical-stop operator present.
   The UI reports requests, not hardware acknowledgements. Damping can allow collapse
   and cannot hold a grasp or guarantee collision avoidance.

**LOCAL — repeat offline checks:**

```bash
python -m unittest discover -s tests -v
xvfb-run -a python -m tests.ui_smoke
xvfb-run -a python -m tests.ui_import_smoke
python -m sonic_laptop.audit_kimodo /path/to/motion.csv --source-fps 30 --output /path/to/audit.json
```

Use an environment with the laptop requirements, Tk, and the C++ development
headers from setup. Audit exit status 2 means reference checks failed.
