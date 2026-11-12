# Follow-mode controls for the guarded SONIC build (for Alex's review)

`prepare_follow_controls.patch` changes `sonic_laptop/prepare.py`, so the generated robot tree gains
the three controls the `follow_balloon` task needs. It also extends the native keyboard test (new
speeds, the `reference` command and the exit blend). The robot binary is not rebuilt. Please review
it before anyone builds or runs it on the robot. **Read question (b) first: it concerns the current
robot build too.**

## What it adds and why

1. **`reference` command**: leaves planner mode and goes back to the reference clips. It is accepted only
   in CONTROL, with the planner running and its startup finished, and with the deadman expired (150 ms
   after the last `move`; the robot may still be finishing its last step). While leaving, a static
   `leaving_planner` flag stops the per-tick
   `r`/walk key, because outside the planner `r` restarts the clip.
2. **Exit blend (keyboard_handler.hpp)**: when the planner is disabled, the controller plays a copy of
   clip 0 (the stationary idle/ready clip) whose joints blend smoothly from the planner's current
   frame into clip 0's first frame. The blend uses smootherstep and lasts one clip length (50 frames
   = 1 s). It then holds that pose like any finished clip (`current_motion_index_ = 0`,
   `play = true`). Upstream jumps straight to the last-selected clip instead. In sim that jump
   (planner arms down, clips arms forward) made SONIC command target steps of 1.8 rad.
3. **`move` 0-7**: 0-3 are forward/back/left/right at `WALK_SPEED`. 4-7 are the same directions at
   `FINE_WALK_SPEED`, used for the last centimetres before a grab. The deadman gets `argument%4`.
   The speeds are named constants in prepare.py, with no environment variable.
4. **`CAPABILITIES` file** in the generated tree: `reference`, `fine-move`, `walk-speed 0.4` and
   `fine-walk-speed 0.2`. The runner refuses robot follow mode (and `--measure walk`) unless the build
   declares both commands, speeds equal to the robot calibration's, **and `planner-stance`**, which the
   patch deliberately does not write: add that line in prepare.py once question (b) is resolved.

Sim evidence: at 0.15 m/s SONIC shuffled in place. A fine speed of 0.3 m/s coasted 0.18 m past the
stop and bumped the balloon into the body 21 times. A fine speed of 0.2 m/s grabbed 9/9 times without
bumps. With all three controls, `follow_balloon` completes 21/21 grabs in sim.

**All of that sim evidence ran under the sim build's much looser guard**, which this patch does not
carry over. Compared with your robot guard (`native/guard.hpp`), the sim guard:
- allows 1.5 rad target steps per 20 ms (robot: 0.20 rad);
- caps joint speed at 80% of SONIC's trained limits, 16-29.6 rad/s (robot: 8 rad/s on every joint);
- allows targets up to 1.5 rad beyond the joint limits (robot: within the limits);
- checks hand/body clearance on measured joints only, against 0 mm (robot: 25 mm on targets and
  measured joints), skips it in planner mode, for 2 s after `reference` and for 1 s after Start;
- uses x8 setup gains on the waist and arms, ramps from the pose at handover (`ramp_start`) and pins
  the control thread (`SONIC_CONTROL_CPU`).
How SONIC's walking, planner exits and grabs fare under the strict robot guard is therefore untested.
One number that is known: in 64 sim grabs (after leaving the planner) the measured hand/body
clearance during the grab and hold never fell below 37 mm, 12 mm above the robot limit. A grab
started 2 s after Start, while SONIC was still swaying, dipped to -44 mm.

## Checks done

- `tests/test_laptop.py` passes with the patch applied (9 tests, on a scratch copy). The native
  keyboard test compiles the generated command block and the patched keyboard handler with g++ and
  checks: `move` 0-3 at 0.4 m/s and 4-7 at 0.2 m/s with direction `argument%4`; `move 8` is refused;
  `reference` is refused while a walk is live; it otherwise leaves the planner through a blend that
  starts at the planner's frame, ends at clip 0's first frame and steps < 0.02 rad per frame on the test
  joint; no walk or stop key follows it; `planner` is refused until the blend has finished.
  The full controller was not rebuilt (no other source changes).
- Unpatched prepare.py regenerated `src/` byte-identical to the current robot tree
  (`.runtime/sonic/deploy`, binary a4d688170fca2e00).
- The patched `src/` differs from that baseline only in the three hunks above (generated into scratch
  trees, since deleted; rerun `python -m sonic_laptop.prepare` to inspect them). `sonic_guard.hpp`,
  `guard-model.json`, `CMakeLists.txt` and `GUARDED_BUILD` (`sonic-laptop-v3`) are unchanged.
- Those hunks match the sim build's logic. The only differences are comments, the fixed fine speed
  in place of the environment variable, and the sim-only guard calls listed above.

## Apply

```bash
git apply gpt_astra/robot_patch/prepare_follow_controls.patch   # from the G1-Zero-Shot root
mv .runtime/sonic/deploy .runtime/sonic/deploy-v3-a4d6   # prepare.py refuses an existing tree
python -m sonic_laptop.prepare --upstream ../GR00T-WholeBodyControl --output .runtime/sonic/deploy
bash scripts/laptop/build.sh
python -m unittest tests.test_laptop                 # native keyboard test: 0.4/0.2, move 8 refused, reference
```

Runner side (not in this patch): `session.py` must accept `reference` in its command whitelist.

## Open questions for you

a. **Walking speeds.** The robot build walks at 0.15 m/s. The proposal is `WALK_SPEED = 0.4` and
   `FINE_WALK_SPEED = 0.2` (the patched keyboard clamps SLOW_WALK to 0.1-0.8 on mode keys). Please confirm or pick hardware values.

b. **Hand/body envelope in planner mode.** The robot guard needs clearance of at least 25 mm on both
   targets and measured joints every tick. The sim build checked measured joints only, against 0 mm,
   and skipped the check in planner mode and for 2 s after leaving it (see above). I applied the guard's
   own FK model (`sonic_laptop.model.clearance`) to
   joints logged in 75 sim planner entries:
   - Measured clearance drops below 25 mm 0.52-0.72 s after `planner`. In the planner stance it is
     -67 to -98 mm, because the arms hang down and the hands overlap the hip/thigh boxes. Even
     upstream `default_angles` scores -37 mm.
   - Unless the real planner stance differs a lot, **the current robot build probably latches about
     0.6 s into planner mode, with or without this patch**. SONIC targets lead inward, so the target
     check may trip even sooner.
   - After `reference`, measured clearance is back above 25 mm 0.47-0.55 s later (70 exits), so the
     exit blend would also trip in the first ~0.5 s.
   - Options: calibrate the envelope for the planner arm pose, bound a suspension, or change the
     planner arm pose. Once settled, write `planner-stance` into CAPABILITIES (prepare.py); until then
     the runner refuses follow mode and the walk measurement on the robot.

c. **0.20 rad target-step limit vs the 1 s blend.** The reference blend moves at most
   1.875 × Δ / 49 rad per frame. The largest planner-stance to clip-0 joint distance across 70 sim
   exits is Δ = 0.72 rad (shoulder yaw; the median exit's largest distance is 0.68 rad). That gives
   0.028 rad per frame, about 7x under 0.20. Measured joints moved at most 0.046 rad per 20 ms
   during the blend. Caveat: SONIC's own targets were not logged, because the sim guard allowed
   1.5 rad steps. To measure them, log the per-motor max |Δq_target| in `Guard::validate` during a
   sim planner exit.

d. **Deadman unchanged**: still 150 ms, and `Deadman::refresh` still only accepts 0-3.

e. **Nothing else in the guard changes**: `native/guard.hpp` is untouched and every limit, gain
   check, latch and stop path stays the same.
