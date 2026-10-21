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

The upstream repository must be beside this one. If you do not have it yet:

```bash
# LOCAL
cd ~/Documents/g1_zero
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git -C GR00T-WholeBodyControl checkout 45c03facbf848e0c743091b7d9c17952096ff908
```

Install dependencies, download the pretrained models and build SONIC:

```bash
# LOCAL
cd ~/Documents/g1_zero/G1-Zero-Shot
export CUDA_HOME=/usr/local/cuda
bash scripts/laptop/setup.sh
```

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
3. Click **Load library** and select your SONIC motion folder, then its stationary
   **idle** clip. Setup becomes available only after validation passes.

You need a stationary idle reference and at least one motion. Each motion must
start and end near that idle pose. Importing does not automatically create an idle
pose or make a generated motion deployable. [Motion format](docs/kimodo-import.md).

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
  --idle ~/motions/idle --motions ~/motions
```

The UI and importer use the same saved **Acceptance limits**. To use a separate
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

Actual motion importing also needs the upstream checkout from step 1.

[MuJoCo / rebuild / limits](docs/laptop-advanced.md) ·
[Safety behavior](docs/sonic-laptop-safety.md) ·
[Validation results](docs/validation-2026-10-21.md)
