# groot_vla: GR00T N1.7 VLA + SONIC on the G1

Runs a GR00T N1.7 vision-language-action policy on top of NVIDIA's SONIC whole-body controller, in MuJoCo
or on our G1, following the [VLA inference tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html).

Only code lives here. The venv, caches, the two pinned upstream clones and the models (~17 GB) go to
`$GROOT_VLA_HOME` (default: this folder, git-ignored). Nothing outside these two folders is written, and
Alex's controller (`sonic_laptop/`, `scripts/laptop/`) is untouched. Never run this and Alex's
`demo.sh`/`run.sh --sim` at once: both use DDS domain 0 on `lo` and ZMQ 5557, so they would command each
other's robot (`run.sh` refuses to start while another G1 sim/deploy is running).

## Install

```bash
export GROOT_VLA_HOME=/path/with/17GB/groot_vla   # optional; put it in ~/.zshrc so run.sh finds it too
bash elham/groot_vla/setup.sh   # clones pinned upstreams, applies patches/, one venv, builds SONIC deploy, downloads the VLA checkpoint
```

The Cosmos-Reason2-2B backbone repo is gated: accept its license on HuggingFace, then
`source elham/groot_vla/env.sh && $GROOT_VLA/.venv/bin/hf auth login` and rerun `setup.sh` (it only fetches config/tokenizer files).

## Run

```bash
bash elham/groot_vla/run.sh sim     # MuJoCo, prompt "grab the bottle"
bash elham/groot_vla/run.sh real    # real G1 (see below)
bash elham/groot_vla/run.sh stop
```

In the `sonic_inference` tmux session: pane 0 (C++ deploy) press Enter to confirm, then in the
keyboard pane: `k` start control loop, `i` blend to initial pose, `p` unpause VLA, `p` pause, `k` stop.
In sim, the MuJoCo window keys `7`/`9` lower/release the support band. The panes live on a private tmux
socket: `TMUX_TMPDIR=$GROOT_VLA_HOME/.tmux tmux attach -t sonic_inference`.

Sim result (2026-10-28): the VLA raises the right arm, reaches forward and closes the hand while SONIC keeps the
robot standing; 30 chunks in 45 s, mean latency 0.87 s.

## What is where

| Path | What |
|---|---|
| **in this folder** | |
| `env.sh` | caches, HF_HOME, TensorRT/onnxruntime paths; sourced by `setup.sh` and `run.sh` |
| `setup.sh` | idempotent install (every step skips if already done) |
| `patches/` | local Isaac-GR00T change (backbone from config + local tokenizer dir) |
| `tools/record_sim_video.py` | `capture` during a sim run (CPU only), `render` afterwards: third-person + head-camera mp4 |
| **in `$GROOT_VLA_HOME`** | |
| `.venv` | one Python 3.10 env for policy server, VLA client and sim (`.venv_inference`, `.venv_sim` in the WBC clone are symlinks to it) |
| `GR00T-WholeBodyControl/` | upstream main `1fa5f7f`; deploy built in `gear_sonic_deploy/target/release` |
| `Isaac-GR00T/` | pinned to `e574928` (last Python 3.10 commit; `main` needs 3.12, which `install_inference.sh` does not support) |
| `checkpoints/grab-bottle-v10/checkpoint-30000` | `cloudwalk-research/gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune`, UNITREE_G1_SONIC, 12/12 on a real G1, prompt "grab the bottle" |
| `robot/pc2_camera_server.py` | head RealSense -> ZMQ :5555 server for PC2, same wire format as gear_sonic |
| `robot/probe_pc2.sh` | read-only check of PC2 Python/RealSense |

Local deviations from upstream:
- SONIC release ONNX files and the planner are symlinks into `~/.cache/huggingface` (GEAR-SONIC, same sha as HF main).
- TensorRT 10.13 from the system apt install (`deps/tensorrt` holds symlinks); onnxruntime 1.16.3 from `/opt/onnxruntime`.
- `Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py`: with `GR00T_BACKBONE_FROM_CONFIG=1` the backbone is
  built from the Cosmos-Reason2-2B config instead of downloading its 4.5 GB of weights; the finetuned
  checkpoint holds every backbone weight. Only config/tokenizer files are fetched (gated repo: accept
  the license on HF and log in once).
- Same files, `GR00T_BACKBONE_LOCAL_DIR` (set in `env.sh`): processor/config load from the local Cosmos-Reason2
  snapshot, because transformers 4.57.3 calls the HF API (mistral-regex check) for hub ids even with
  `HF_HUB_OFFLINE=1`. The server therefore runs without internet (needed at the robot).
- `GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py` (`patches/gr00t-wbc-*`): `--hand-action-mode
  inspire_grip` and `--zero-hand-state`, on by default in `run.sh`. The cloudwalk checkpoints were trained on
  Inspire FTP hands: their hand actions are grip levels (channels 0-4, 0 = open, 1 = closed, checked on an
  episode) and the hand state was always zero (checkpoint `statistics.json`: min = max = 0). Passing them
  through would send grip levels as Dex3 radians (right thumb_1 +1.0 where the Dex3 grasp needs -0.7). The
  client now drives our Dex3 hands with `grip x` the upstream Dex3 closed preset and sends zero hand state.
- Isaac-GR00T installed with `--no-deps` plus only the inference imports (no TensorRT wheels, deepspeed, wandb).

## Speed

Measured 2026-10-28, GPU capped at 45 W by the laptop BIOS (default 125 W; nvidia-powerd reports the
BIOS disables Dynamic Boost). One chunk covers 0.8 s (40 actions at 50 Hz); the tutorial targets 2.5 Hz.

| Setup | VLA latency |
|---|---|
| Sim run (GPU also renders MuJoCo) | mean 0.87 s, max 1.0 s |
| Policy server alone (= real robot: camera on PC2, no sim) | mean 0.51 s, p50 0.44 s, p90 0.67 s, max 1.03 s |

`tools/bench_vla_latency.py` reproduces the second row from a `record_sim_video.py capture` file.
Under GPU saturation the sim loop can stall > 500 ms and trip SONIC's LowState watchdog; don't add
GPU work (rendering, recording) during a sim run.

## Other public UNITREE_G1_SONIC checkpoints (survey 2026-10-28)

41 public GR00T checkpoints carry a `unitree_g1_sonic` head (checked in their statistics); NVIDIA releases
none (NVlabs/GR00T-WholeBodyControl#171: collect and fine-tune your own). Almost all use the `release` SONIC.

| Checkpoint | Data / result | Fit for our G1 |
|---|---|---|
| cloudwalk-research `...-371ep-v10-finetune` ckpt-30000 (in use) | real G1, grab bottle, 12/12 | yes with the Inspire->Dex3 grip mapping |
| cloudwalk-research v2 (10/12), v5 (9/12), v6/v9 (8/12), v11 (untested) | same task and hands | same mapping; v10 is the best scored |
| ETHRC-humanoid `groot-n1_7-red-cube-box-locomanip-10k` | real G1, walk + carry cube/box, Dex3 units, no closed-loop result | only Dex3-native candidate; gantry first |
| wsagi `GR00T-N1.7-G1-SONIC-BonesSeed(-V2)`, `-LAFAN` | sim, prompt -> kick/walk/jump | sim demos only, not for the robot |
| ETHRC-humanoid `gr00t-g1-sonic-3cam` | needs two wrist cameras | not usable |

Real-robot SONIC datasets for fine-tuning our own: cloudwalk grab-bottle sets (Inspire, up to 502 ep),
ETHRC-humanoid `red_cube_cardbox_all` (304 ep, Dex3 layout), NONHUMAN-RESEARCH children's table (533 ep, Dex1).

## Real robot

1. Laptop Ethernet on `192.168.123.x` (deploy.sh `real` auto-detects that interface).
2. On PC2 (`192.168.123.164`): `python3 pc2_camera_server.py` (needs pyrealsense2, cv2, numpy, pyzmq, msgpack;
   check with `ssh unitree@192.168.123.164 'bash -s' < robot/probe_pc2.sh`).
3. Robot in the gantry, then `bash run.sh real`.

The checkpoint was trained on another lab's G1 (Inspire FTP hands, head RealSense D435i, their table).
It performs the grasp motion even without a bottle in view.
