# G1 Zero Shot — SONIC GPU-laptop controller

Pretrained SONIC on a **LOCAL x86_64 NVIDIA GPU laptop**, controlling a G1 EDU U4
through wired Ethernet. Includes a desktop panel, motion buttons, play-once/folder
playback, hold-to-move keyboard control, joint guards, hand-clearance envelopes and
a latched software stop. Dex3 hand commands are disabled.

**Commissioning code: not yet validated on hardware.** Clip completion keeps SONIC
active on the final standing reference. Software stop instead commands damping,
which can let the robot fall: use your gantry and physical stop. Read the
[exact behavior and limitations](docs/sonic-laptop-safety.md) before arming.

**LOCAL — open the library UI and import Kimodo motions (no GPU or robot needed):**

```bash
cd ~/Documents/g1_zero/G1-Zero-Shot
bash scripts/laptop/library.sh
```

Click **Import Kimodo**, choose files or a folder, enter the source FPS, and choose
an unused output folder. The result list shows **Checks passed**, **Needs review**,
or **Import failed**; click a row for details. **Load output library** then asks
for your validated idle clip and checks the whole library. Offline library mode
cannot arm hardware. Import and library changes in the control panel are available
only before arming. See [Kimodo import instructions](docs/kimodo-import.md).

**LOCAL — preview the UI without a GPU or robot:**

```bash
cd ~/Documents/g1_zero/G1-Zero-Shot
bash scripts/laptop/demo.sh
```

The offline demo has simulated clip progress and keyboard controls. Its example
motions are UI placeholders, not deployable motions. It never starts a controller.
Importing through the demo still writes converted files to the output folder you choose.

1. **LOCAL — get the supported upstream source**, beside this repository. Use a
   separate checkout if your existing one has changes. The audited revision is:
   `45c03facbf848e0c743091b7d9c17952096ff908`.

   ```bash
   cd ~/Documents/g1_zero
   git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
   cd GR00T-WholeBodyControl
   git checkout 45c03facbf848e0c743091b7d9c17952096ff908
   git lfs pull --include='gear_sonic_deploy/g1/**' --exclude=''
   ```

2. **LOCAL — install/build.** Ubuntu 22.04/24.04, working NVIDIA driver and CUDA
   **12.x development toolkit** are prerequisites (`nvidia-smi` alone does not
   prove the toolkit exists). The script installs apt/Python dependencies,
   downloads TensorRT **10.13.3.9** and pretrained models, then builds an isolated
   patched controller. Allow substantial disk space for TensorRT. It never starts
   robot control. Existing TensorRT can be supplied via `TensorRT_ROOT`.

   ```bash
   cd ~/Documents/g1_zero/G1-Zero-Shot
   export CUDA_HOME=/usr/local/cuda
   bash scripts/laptop/setup.sh
   ```

   If apt reports unrelated broken repositories, fix those sources first; do not
   bypass signatures. For a failed build after preparation, rerun
   `bash scripts/laptop/build.sh`. Upstream documents the required desktop
   TensorRT version in its [installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html).

3. **LOCAL — supply your motion library.** Example structure:

   ```text
   my_motions/
     idle/       # stationary standing reference, hands clear of body
     small_wave/ # each directory contains SONIC's CSV + metadata files
     arm_raise/
   ```

   Every clip must start/end near idle with a stationary 0.5-second dwell. See the
   [motion format and checks](docs/sonic-laptop-safety.md#motion-library-contract).
   No unchecked demonstration motions are supplied.
   **Kimodo-G1 imports are supported:** see [the importer instructions](docs/kimodo-import.md).

   ```bash
   # LOCAL — example for a 30 Hz default Kimodo-G1 CSV export
   bash scripts/laptop/import_kimodo.sh /absolute/path/wave.csv \
     --source-fps 30 --output /absolute/path/my_motions/wave
   ```

4. **LOCAL — test in MuJoCo first**, in two terminals. Use the upstream simulator
   installation if `.venv_sim` is absent (its installer recreates that environment).

   ```bash
   # LOCAL — terminal 1, upstream simulator
   cd ~/Documents/g1_zero/GR00T-WholeBodyControl
   # First time only: bash install_scripts/install_mujoco_sim.sh
   source .venv_sim/bin/activate
   python ../G1-Zero-Shot/scripts/laptop/sim.py --idle /absolute/path/my_motions/idle
   ```

   ```bash
   # LOCAL — terminal 2, guarded panel
   cd ~/Documents/g1_zero/G1-Zero-Shot
   bash scripts/laptop/run.sh --sim --interface lo \
     --idle /absolute/path/my_motions/idle --motions /absolute/path/my_motions
   ```

   Use **Suspended setup**, wait for **CHECKS PASSED**, then **Feet down / start**. Use upstream
   simulator key `9` to release suspension when ready. Test completion, queues,
   keyboard release/focus loss and stop while supported. The preliminary hand
   envelope rejects the stock default arm configuration; the simulation wrapper
   holds your validated idle pose until the first controller command arrives.

5. **LOCAL — hardware, after successful simulation.** Connect wired Ethernet to
   the robot network; inspect `ip -br addr` and select the **laptop's Ethernet
   interface** (e.g. `enp3s0`, not the robot IP). Verify network configuration and
   manufacturer controller handover procedure for your G1. No Orin SSH is needed.
   With the robot supported and workspace/hand clearance verified:

   ```bash
   cd ~/Documents/g1_zero/G1-Zero-Shot
   bash scripts/laptop/run.sh --interface enp3s0 \
     --idle /absolute/path/my_motions/idle --motions /absolute/path/my_motions
   ```

   Opening the panel sends nothing. **Suspended setup starts body actuation after
   model loading**, ramps to the validated idle pose and holds it with joint PD;
   hands remain uncommanded. Keep the robot suspended with clear limbs. Wait for
   **CHECKS PASSED** (joint error ≤0.12 rad and speed ≤0.3 rad/s for two seconds),
   then lower until both feet are planted. Click **Feet down / start** and confirm
   ground support. This starts SONIC; it does not start a motion or walking.
   Separately select **Play once**, **Play all**, or **Enable keyboard control**.
   Keep the panel running after completion. Checks do not measure foot contact or balance.

6. **LOCAL — controls and shutdown.** Clip buttons use your validated library.
   Keyboard tab: enable planner, hold **Space + W/S/A/D** for slow movement;
   release to request idle. Fresh movement commands expire after 150 ms; enabling
   keyboard clears previously held keys. This bounds command intent, not physical stopping time. **Esc / STOP** latches damping. Support the robot
   before **Close**. Restart a fresh session to rearm or switch from planner back
   to clips. Logs/status/PID live under `artifacts/laptop/sessions/`.

7. **LOCAL — tests** (no GPU or robot commands):

   ```bash
   cd ~/Documents/g1_zero/G1-Zero-Shot
   source artifacts/laptop/env.sh
   python -m unittest discover -s tests -v
   ```

Existing [Mila simulation instructions](docs/sonic-mila-legacy.md) and
[Orin preflight/bundling](docs/sonic-orin.md) are separate workflows. The scripts
in `scripts/laptop/` are the new laptop deployment path.

8. **LOCAL — rebuilding an older guarded controller:** these controls require
   `sonic-laptop-v2`; the launcher rejects v1. With the controller stopped and the
   robot fully supported, preserve the old build and regenerate:

   ```bash
   cd ~/Documents/g1_zero/G1-Zero-Shot
   source artifacts/laptop/env.sh
   mv artifacts/laptop/deploy "artifacts/laptop/deploy-backup-$(date +%Y%m%d-%H%M%S)"
   python -m sonic_laptop.prepare --upstream ../GR00T-WholeBodyControl --output artifacts/laptop/deploy
   bash scripts/laptop/build.sh
   ```

   Repeat the MuJoCo validation before hardware deployment. Full GPU compilation
   and real-robot stopping behavior have not been validated by the CPU/UI tests.
