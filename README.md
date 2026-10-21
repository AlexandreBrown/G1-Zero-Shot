# Run SONIC on your G1

Use a **GPU laptop** to run pretrained SONIC and control your **G1 EDU U4 over Ethernet**.
Nothing runs on the robot's onboard PC. All commands below run on your **LOCAL laptop**.

**No NVIDIA GPU?** Use [Offline preview](#offline-preview-cpu-or-gpu). It supports the
UI and motion importing, but does not run SONIC or control the robot.

## Getting started

You need Ubuntu 22.04/24.04, an NVIDIA GPU with a working driver, the **CUDA 12.x
development toolkit**, and your G1 connected by wired Ethernet. Keep the gantry
and physical stop available. This controller still needs GPU/hardware validation;
start with [MuJoCo](docs/laptop-advanced.md#mujoco).

### 1. Install dependencies — once

Run setup **on each GPU laptop after cloning this repository**. It downloads the
supported upstream source, TensorRT, ONNX Runtime and pretrained SONIC models,
creates the Python environment and builds the controller. No artifact archive
needs to be copied or pushed to Git.

```bash
# LOCAL
cd ~/Documents/g1_zero/G1-Zero-Shot
export CUDA_HOME=/usr/local/cuda
bash scripts/laptop/setup.sh
```

Setup generates `.runtime/sonic/env.sh` locally. This is a **persistent installation**,
not a temporary artifact; keep `.runtime/` on that laptop. It is excluded from Git
because its paths and compiled files are machine-specific. `run.sh` uses it only
after setup has finished.

Already installed an older version? [Rebuild instructions](docs/laptop-advanced.md#rebuild).

### 2. Open the app

Find your **laptop's Ethernet interface**, then use it below instead of `enp3s0`:

```bash
# LOCAL
ip -br addr
bash scripts/laptop/run.sh --interface enp3s0
```

Opening the app does not actuate the robot.

**In the UI:**

1. Optional: open **Acceptance limits**, edit the motion checks, then apply.
2. Use **Import Kimodo** for generated G1 CSV files. Enter their source FPS
   (**30** for your six-second, 180-frame export) and choose a new output folder.
3. The app automatically loads `motions/kimodo/idle.csv` as the idle
   reference (30 FPS). Click **Load library** to add your SONIC motion folder.
   Use **Idle: default → Choose custom SONIC idle** only to override it.

Setup requires the idle to pass validation; a motion folder is optional for keyboard
control. Each added motion must start and end near the idle pose.
**The supplied idle currently fails clearance/stance checks, so setup stays disabled**
until it is corrected or replaced. [Motion format](docs/kimodo-import.md).

### 3. Run: setup → motion or keyboard

1. **Setup:** with the robot supported in the gantry and limbs clear, click
   **Suspended setup**. This starts body actuation. Watch **Live activity** for
   loading progress and joint checks.
2. **Lower:** wait for **CHECKS PASSED**, lower both feet onto the floor, then click
   **Feet down / start** and confirm. SONIC starts holding the reference.
3. **Motion:** select a motion and click **Play once**, or **Play all** for the folder.
   At completion, SONIC continues holding the final reference. Leave the app running.
4. **Keyboard:** click **Enable keyboard control**, release any held keys, then hold
   **Space + W/S/A/D** for forward/back/left/right movement. Releasing a key or
   leaving the window requests idle. Returning to motions requires a supported restart.

**Emergency stop:** click the large red button or press **Esc** to request body
and both-hand damping. Damping can let the robot collapse; the gantry must catch it.
Support the robot before closing the app. Normal playback sends no finger-motion commands.

## CLI shortcuts

These commands prepare motions or open the same UI; they do not bypass setup.

```bash
# LOCAL — import a Kimodo clip
bash scripts/laptop/import_kimodo.sh ~/Downloads/motion.csv \
  --source-fps 30 --output ~/motions/reach

# LOCAL — open with an existing validated library
bash scripts/laptop/run.sh --interface enp3s0 \
  --motions ~/motions
```

The UI and importer use the same saved **Acceptance limits**. To use a custom
idle, add `--idle /path/to/sonic_idle`. To use a separate acceptance
profile, add `--limits /path/to/limits.json` to either command.
Changing limits requires reloading the library. Runtime guards remain fixed.
[Limits and defaults](docs/laptop-advanced.md#acceptance-limits).

## Offline preview: CPU or GPU

No robot, GPU driver or CUDA installation is needed. This is UI/import preview,
not SONIC inference or a physics simulation.

```bash
# LOCAL — install lightweight UI dependencies once
cd ~/Documents/g1_zero/G1-Zero-Shot
bash scripts/laptop/setup_ui.sh

# LOCAL — try setup, motion and keyboard screens using fake robot state
bash scripts/laptop/demo.sh

# LOCAL — import and validate your actual motion files without robot control
bash scripts/laptop/library.sh
```

The UI setup also fetches the supported upstream source and robot assets for importing.

[MuJoCo / rebuild / limits](docs/laptop-advanced.md) ·
[Safety behavior](docs/sonic-laptop-safety.md) ·
[Validation results](docs/validation-2026-10-21.md)

## What belongs in Git?

- **`motions/`**: all motion files and preview videos, including `.csv` and `.mp4`.
- **`.runtime/sonic/`**: downloaded dependencies, generated environment, build,
  local settings and sessions. Persistent on this laptop, excluded from Git.
- **`.venv_laptop/`, `.venv_ui/`**: generated Python environments, excluded from Git.
- **`artifacts/`**: disposable reports, screenshots and optional offline-transfer
  archives. Not required to clone or install the project; excluded from Git.

The optional Orin source/model tar is for offline transfer, not Git or laptop setup.
