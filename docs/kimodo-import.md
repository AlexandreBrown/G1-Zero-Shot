# Import Kimodo-G1 motions

The offline importer accepts **default Kimodo-G1 MuJoCo CSV** exports, or
**Kimodo-G1 NPZ** using an installed Kimodo environment. It produces a SONIC
motion directory readable by our control panel. No GPU inference, DDS, or robot
commands are involved in importing.

## Import through the UI

**LOCAL:** run `bash scripts/laptop/library.sh` from the repository. This opens a
library workspace with hardware arming disabled. Click **Import Kimodo**:

1. Choose CSV/NPZ files or a folder.
2. Enter the actual source FPS and a **new** output folder.
3. For NPZ, optionally select the **local** Python executable with Kimodo installed.
4. Click **Import motions**. Conversion runs in a worker so the window stays responsive.
5. Select a result row for the reason if it needs review or failed. Every converted
   clip includes the same JSON report as the CLI importer.
6. Click **Load output library**, choose your validated idle clip, and let the
   existing library/transition checks run. Rejected libraries do not replace the
   currently loaded library.

The same import button is available in the controller UI **before arming**.
Import, library reload and arm are mutually excluded during import/validation;
active controller sessions cannot change their motion snapshots.

**CLUSTER (Kimodo UI):** generate with the **G1** model, export the default G1 CSV,
and note the export FPS. Download/copy those files to the local laptop, then use
the file picker. CSV import does not need Kimodo installed on the laptop. A cluster
Python path cannot be used in the local NPZ interpreter field. Export NPZ as well
if you want to retain the original generated motion for later inspection.

## Command-line import

1. **LOCAL — import a CSV.** Supply the actual export frame rate; CSV contains no
   timestamp. The example assumes **30 Hz**, not a universal default.

   ```bash
   cd ~/Documents/g1_zero/G1-Zero-Shot
   bash scripts/laptop/import_kimodo.sh /absolute/path/wave.csv \
     --source-fps 30 --output /absolute/path/sonic_motions/wave
   ```

2. **LOCAL — or import a whole folder.** Each CSV/NPZ becomes one motion
   subdirectory. Import is nonrecursive and never overwrites existing output.
   If a directory contains both `wave.csv` and `wave.npz`, select one format
   first; duplicate names are rejected rather than silently choosing one.

   ```bash
   bash scripts/laptop/import_kimodo.sh /absolute/path/kimodo_exports \
     --source-fps 30 --output /absolute/path/new_sonic_library
   ```

3. **LOCAL — NPZ option.** Point to Python in your **existing Kimodo environment**.
   CSV import only needs the laptop environment's NumPy/SciPy; it does not install
   Kimodo. The NPZ helper loads numeric arrays with pickle disabled, verifies the
   G1 skeleton shape, and calls NVIDIA's `MujocoQposConverter` on CPU.

   ```bash
   bash scripts/laptop/import_kimodo.sh /absolute/path/wave.npz \
     --kimodo-python /absolute/path/kimodo/.venv/bin/python \
     --source-fps 30 --output /absolute/path/sonic_motions/wave
   ```

4. **LOCAL — review `import-report.json` in each output.** Exit code `0` means
   conversion and per-clip commissioning checks passed. Exit code `2` means an
   import or commissioning check failed. Validly converted but rejected clips
   remain available for offline review, with the reason in the report; the control
   panel will still reject them. The report records the source hash and FPS.
   Successful clips are retained if another clip in a batch fails.

5. **LOCAL — use the converted library with the existing panel.** Supply a
   separately validated stationary idle clip. The panel still checks idle matching
   and transitions across the library, then you test in MuJoCo before hardware.

   ```bash
   bash scripts/laptop/run.sh --sim --interface lo \
     --idle /absolute/path/sonic_motions/idle \
     --motions /absolute/path/sonic_motions
   ```

## Supported conventions

- CSV is **headerless**, with exactly 36 values per row: world xyz in meters,
  root quaternion **wxyz**, then 29 joint angles in radians in G1 MuJoCo order.
  Coordinates are **z-up / +x forward**. Use Kimodo's default raw joint-DOF export;
  custom `mujoco_rest_zero=True` exports are not supported. The importer cannot
  distinguish alternate conventions from numeric CSV alone.
- NPZ requires a single clip with `local_rot_mats` shaped `(T,34,3,3)` and
  `root_positions` shaped `(T,3)`. Human SOMA/SMPL-X exports need retargeting and
  are rejected. NPZ is passed through the official exporter with
  `root_quat_w_first=True`, `mujoco_rest_zero=False`, and `device='cpu'`.
- Resampling uses linear interpolation for root translation and bounded joint
  angles, quaternion SLERP for root orientation, then numerical derivatives at
  **50 Hz**. The final pose is preserved; an off-grid endpoint adds less than
  20 ms of terminal hold. It does not add the 0.5-second dwell our guards require.
- Output joint arrays use **IsaacLab order**. Body data is **root-only**, with
  metadata `[0]`, suitable for the standard SONIC **joint-based encoder mode 0**.
  Root linear/angular velocity CSVs are included. This is not a full-body FK
  export for SMPL or VR encoder modes, and it does not contain hand commands.
- The importer does not retarget, clip joint angles, slow the motion, generate an
  idle pose, stitch poses, or alter unsafe endpoints to make checks pass.
  Conversion does not establish physical safety or balance.

## Verification

Tests exercise CSV conversion end-to-end through `read_clip` and `stage_library`,
known joint permutations and velocity units, quaternion sign flips/shortest-path
interpolation, endpoint timing, malformed data, overwrite refusal, batch import,
and rejection of human NPZ files. The NPZ subprocess handoff is tested with a
stub exporter. A real Kimodo NPZ export and physical tracking have **not** been
validated in this workspace because Kimodo is not installed here.

References: NVIDIA's [Kimodo output formats](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/output_formats.html),
[official G1 exporter](https://github.com/nv-tlabs/kimodo/blob/main/kimodo/exports/mujoco.py),
and the upstream `GR00T-WholeBodyControl/docs/source/references/motion_reference.md`.
