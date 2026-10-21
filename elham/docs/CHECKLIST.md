# Hardware day checklist (print this)

How to run each step is in [the README](../README.md). Tick every box; a box you cannot tick means no robot run.

## A. Desk, the day before
- [ ] Robot: 29 DoF, waist fastener off and unlocked, pelvis **grey** (old hip-pitch motor), Machine Type in the app **5** = old hip pitch, new hip roll (1–6, 13, 14 = old hip pitch; 1–3 also old hip roll; 10–12, 15, 16, 18 = new), firmware ______, hands Dex3-1 (EDU U4)
- [ ] Laptop installed (step 2): `sonic_run.sh --dry-run` works; TensorRT **10.13** (x86)
- [ ] Every policy folder has `manifest.json` and a passing `parity.json` (`sonic_run.sh --real` refuses otherwise)
- [ ] Every (policy, motion) for tomorrow passed the preview with `--delay-ms 20` (step 5a); the policy passed the deploy plumbing check (5b); TensorRT engines cached
- [ ] Preview videos of tomorrow's motions on the laptop; the whole team has watched them
- [ ] Motion list written on the trial sheet with each motion's **root travel**: fixed hoist = travel ≤ 0.5 m only; longer needs a trolley
- [ ] Robot battery, spare battery and remote charged
- [ ] `tmux` installed; launch commands written down with the explicit NIC (`enp58s0` on this laptop)

## B. Room, on the day
- [ ] Gantry rated ≥ 150 kg, low-stretch rope through both shoulder buckles, dead knot
- [ ] 3 m clear zone + motion travel marked; floor flat, dry, non-slip
- [ ] Ethernet cable direct laptop ↔ robot (no router), strain-relieved along the gantry; static IP profile `g1` up
- [ ] Simulator closed on the laptop (no `run_sim_loop.py`), and no other GPU job running (transcription, training, sims): `nvidia-smi` lists only the desktop. A GPU fault in another program can hang the GPU the deploy program runs on (2026-10-21: a Whisper job froze this laptop)
- [ ] Roles: operator (`O`) ______  remote (`Start`/`A`/`Select`) ______  rope ______  observer/cameras ______
- [ ] Everyone knows: **stop = the letter `O` (not zero, typed in the program's pane) or `Select`, then hoist**; a stop prints `EMERGENCY STOP triggered`; remote `B` is NOT a stop (back to frame 0, the policy keeps running); anyone may call stop; nobody uses Ctrl+C; laptop sound on (warnings are spoken)
- [ ] Everyone knows the never-press list (README "Stops"); the remote's `L2+B` / `L1+A` are NOT stops while the program runs
- [ ] Two cameras recording; trial sheet open; a clap at each trial start

## C. Bring-up, robot hanging, feet clear
- [ ] Power on (short press + hold > 2 s), wait ~1 min (zero torque)
- [ ] Remote paired (DL light)
- [ ] Battery read in the Unitree app: ______ % (≥ 80 %); then disconnect the app
- [ ] Damping (V1.0.4: `L2+B`, V1.0.2: `L1+A`), then **`L2+R2` debug mode**; a leg joint moves by hand with resistance
- [ ] `ping -c 50 192.168.123.161` → 0 % loss
- [ ] `tmux new -s sonic`; right pane: `python scripts/lowstate_monitor.py --iface enp58s0 --check 5` → **PRE-FLIGHT: PASS**
- [ ] Right pane: live monitor running (`python scripts/lowstate_monitor.py --iface enp58s0`); move one arm joint by hand, the right joint changes

## D. Phase A tests (once per day, robot hanging)
- [ ] Launch, `Init Done`, symmetric pose; `G1 type:` ______ written down; hands off the robot's hands during INIT
- [ ] `O` → joints go soft
- [ ] Relaunch, `@`, `Select` → joints go soft
- [ ] Relaunch, `pkill -9 -x g1_deploy_onnx_ref` from the right pane → stiff for ___ s before the launcher's damping
- [ ] Power-off rehearsal once (short press + hold > 2 s, rope under tension)

## E. Per trial
1. Robot hanging by the shoulders, **torso upright** (not folded onto the waist), feet clear → `sonic_run.sh --real ...` → type `hanging` → **hands off the keyboard**
2. `Init Done` → press `@` (terminal confirms the gamepad)
3. Lower until feet flat, rope near-taut but carrying no weight → remote **`Start`** (quick arm/knee adjustment is expected)
4. Slack ~5 cm over 5–10 s; stand 10–20 s: lean within ~5° of the motion's value in MOTIONS.md (stand test ≈ −1°), no buzzing, no rocking; temperatures OK
5. Slack 5–10 cm → **`A`** → watch against the preview
6. `completed` (robot returns to frame 0) → re-tension rope → **`Select`**
7. Write on the sheet: trial folder, outcome, max temperature (monitor pane), `lean held` (`analyze_logs.py`)

**ABORT = `O`/`Select` FIRST, then hoist** (hoist at once only if already falling), on:
motion unlike the preview · shaking/chatter · leaning clearly more than the preview · hanging on the rope ·
foot slip or unplanned step · `ErrorMonitor` / CRC / `HighTemp` / `Lost LowState` · a spoken warning ·
`LowState age` > ~10 ms more than once ·
monitor rings 85 °C · anyone in the zone.

Cool down (hanging, damped) when the monitor rings 70 °C. Standing idle under ~2 min per launch.
Robot still stiff after a stop: `python scripts/damping_hold.py --iface enp58s0 --seconds 5`, else hoist + power off.

## F. After a fall
1. `O`/`Select`; rope handler takes the load; hands off until the joints are soft (pinch points)
2. Read `[ErrorMonitor]` / motor error codes and temperatures (monitor pane)
3. Inspect shells, feet, cables, the Ethernet port; photograph damage
4. If the robot fell more than once: Unitree IMU calibration before the next session
5. Redo the phase-A launch in the air before the next ground run
6. Write the trial folder name and what happened on the sheet

## G. Shutdown
- [ ] Robot hanging, rope under tension → damping combo → power off (short press + hold > 2 s)
- [ ] Copy `logs/<session>/` and the videos to storage
- [ ] `python scripts/analyze_logs.py logs/<session>/* --json logs/<session>_summary.json` and `python scripts/build_dataset.py logs/<session>/*`
