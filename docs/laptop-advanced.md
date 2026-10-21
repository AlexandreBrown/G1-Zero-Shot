# Laptop details

## MuJoCo

The app and simulator use the bundled Kimodo idle by default; `--idle` overrides it.
The current bundled file fails validation and must be corrected or replaced before simulation/control.
Test the same idle reference, library and acceptance profile before hardware.
Both terminals run on the **LOCAL GPU laptop**.

```bash
# LOCAL — terminal 1; install simulator once if .venv_sim is missing
cd ~/Documents/g1_zero/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
python ../G1-Zero-Shot/scripts/laptop/sim.py
```

```bash
# LOCAL — terminal 2
cd ~/Documents/g1_zero/G1-Zero-Shot
bash scripts/laptop/run.sh --sim --interface lo --motions ~/motions
```

In the panel, use **Suspended setup**, wait for checks, release simulator suspension
with key **9**, then use **Feet down / start**. Test playback completion, keyboard
release, focus loss, heartbeat loss and stop. The simulator wrapper holds your
initial pose until it receives the first body command.

## Rebuild

If upgrading from the old `artifacts/laptop/` layout, run `bash scripts/laptop/setup.sh`
once to create the new installation. Do not copy an old `env.sh` between PCs.
Keep the old directory until the new build works; it is not deleted automatically.
To reuse an existing compatible TensorRT download, set `TensorRT_ROOT` to its
extracted folder before setup.


Do this with the controller stopped and the real robot supported. The current
controller requires `sonic-laptop-v3`; older versions are rejected.

```bash
# LOCAL
cd ~/Documents/g1_zero/G1-Zero-Shot
source .runtime/sonic/env.sh
mv .runtime/sonic/deploy ".runtime/sonic/deploy-backup-$(date +%Y%m%d-%H%M%S)"
python -m sonic_laptop.prepare --upstream ../GR00T-WholeBodyControl --output .runtime/sonic/deploy
bash scripts/laptop/build.sh
```

Repeat MuJoCo checks after controller changes. Changing motion acceptance settings
does not require a C++ rebuild.

## Acceptance limits

Open **Acceptance limits** before setup. Applying changes saves the profile and
locks setup until you reload and validate the library. Import checks, library
validation and the audit CLI all use this profile. Once setup starts, editing is
locked. Each validated session saves a separate `acceptance-limits.json` snapshot.

| Setting / JSON key | Default | Allowed range |
| --- | --- | --- |
| Reference joint speed / `reference_speed` | 4 rad/s | 0.1–8 rad/s |
| Position/velocity mismatch / `velocity_mismatch` | 1 rad/s | 0.1–4 rad/s |
| Root travel from first frame / `root_displacement` | 0.15 m | 0.01–1 m |
| Root tilt / `root_tilt` | 0.20 rad | 0.01–0.50 rad |
| Minimum hand envelope clearance / `hand_clearance` | 0.025 m | 0.025–0.20 m |

The default profile is `.runtime/sonic/acceptance-limits.json`. If absent, built-in
defaults apply. You can edit this JSON or use a separate `--limits` file:

```json
{
  "reference_speed": 4.0,
  "velocity_mismatch": 1.0,
  "root_displacement": 0.15,
  "root_tilt": 0.20,
  "hand_clearance": 0.025
}
```

A larger acceptance threshold permits more reference motion; it does not modify,
slow or clip a trajectory. It does not certify stability or safe stopping.
Hand clearance may be tightened but cannot be reduced below the fixed runtime
minimum of 25 mm. Joint angle bounds, endpoint dwell/idle matching, runtime motor
checks, heartbeat deadlines and damping behavior are not editable here. Raising a
reference-speed limit does not raise the controller's measured/commanded speed guards.

**LOCAL — audit using the saved app profile or an explicit profile:**

```bash
python -m sonic_laptop.audit_kimodo ~/Downloads/motion.csv --source-fps 30 \
  --limits /path/to/limits.json --output /path/to/audit.json
```

The supplied reach motion remains rejected under defaults. Increasing its speed
limit alone does not resolve its hand-envelope overlap.

## Setup notes

- The supported upstream commit is `45c03facbf848e0c743091b7d9c17952096ff908`.
  Setup fetches this revision when upstream is absent. An existing checkout is
  checked without overwriting it. Preparation checks source hashes.
- GPU setup requires the CUDA **development toolkit**, not just `nvidia-smi`.
  It installs TensorRT 10.13.3.9; set `TensorRT_ROOT` to reuse an existing compatible
  installation. Allow space for the approximately 7 GB archive and extraction.
- For apt signature errors, fix the offending package source before rerunning;
  the setup does not bypass signature checks.
- The hardware interface is the laptop's Ethernet name, not the robot IP.
  Ensure the robot's controller handover procedure is appropriate and no competing
  body/hand command publisher is running. No Orin SSH is required.
- Session logs, PID, staged references and acceptance limits are under
  `.runtime/sonic/sessions/`. **Live activity** shows controller output in the UI.
- Offline previews use Ubuntu Tk for readable fonts. They cannot arm hardware;
  the demo's example motions are placeholders, not deployable clips.

[Validation](validation-2026-10-21.md) · [Runtime safety details](sonic-laptop-safety.md) ·
[Mila workflow](sonic-mila-legacy.md) · [Orin workflow](sonic-orin.md)
