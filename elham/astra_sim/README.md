# astra_sim: GPT Astra → SONIC in MuJoCo, and the robot GUI idle

What made `gpt_astra --backend sim` work on Elham's laptop (2026-10-28) and the idle used for the first
hardware bring-up of the panel. The `gpt_astra/` and `sonic_laptop/native/guard.hpp` changes are separate
commits on this branch.

| File | What |
|---|---|
| `drive_astra_sim.sh` | Drives `gpt_astra/run.py --backend sim` through its prompts inside a private user+net namespace (own loopback DDS/ZMQ ports and lock file, so it can run beside another G1 sim), sim on CPU 3, controller on CPU 2, rest on E-cores 12-19. `NET=1` gives the namespace internet (slirp4netns) for OpenAI; the key is read from `~/.config/gpt_astra/openai_key`. `ORDER=readme` tests README's release-then-start order. |
| `deploy-sim-guard.patch` | Sim-only controller changes on top of the stock `prepare.py` output (main's guard): G1 session's calibrated guard v2 (refuses to start unless `SONIC_GUARD_SIM` is set), setup-only waist gain x8, 1 s envelope grace after CONTROL start, `SONIC_CONTROL_CPU`. |
| `idle_sim_demo/idle_apose.csv` | Kimodo CSV (30 FPS) of SONIC's stance with the arms opened (from the G1-Zero-Shot session's sim demo). |
| `idle_sim_demo_sonic/` | The same idle converted with `scripts/laptop/import_kimodo.sh --source-fps 30`: 200 frames at 50 Hz, commissioning checks passed. Pass it with `--idle`; not yet validated on hardware. |

## Sim (GPT Astra)

```bash
# deploy-sim-guard.patch targets the STOCK guard (main's sonic_laptop/native/guard.hpp, i.e. without
# commit cd09769 of this branch, which removes lines the patch edits). Build the sim controller from a
# stock checkout: prepare.py output -> .runtime/sonic/deploy-sim, then from the repo root:
(cd .runtime/sonic/deploy-sim && patch -p1 < ../../../elham/astra_sim/deploy-sim-guard.patch)
# rebuild deploy-sim the same way scripts/laptop/build.sh builds deploy, then:
NET=1 bash elham/astra_sim/drive_astra_sim.sh run_sim --max-decisions 10 --max-decision-age 6
```

The driver exports `GPT_ASTRA_BALLOON=hover`: the balloon floats at reach height because a free fall lasts
~0.5 s and a GPT decision 2-6 s. Order that works: lower the band, start SONIC, release the band immediately;
README's release-then-start collapses the robot in sim (pelvis 0.79 -> 0.53 m, ankle past its limit).

## Robot GUI

```bash
bash scripts/laptop/run.sh --interface enp58s0 --idle elham/astra_sim/idle_sim_demo_sonic
```
