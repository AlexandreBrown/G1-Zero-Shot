#!/usr/bin/env bash
# Rebuild the groot_vla install from scratch, or finish a partial one (every step skips if done).
# Needs: Ubuntu 22.04 x86_64, NVIDIA driver + CUDA 12.x toolkit, apt TensorRT 10.13 (libnvinfer10),
# uv, git-lfs, cmake, clang, just. Disk: ~17 GB in $GROOT_VLA_HOME (default: this folder).
#
#   GROOT_VLA_HOME=/big/disk/groot_vla bash setup.sh
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
mkdir -p "$GROOT_VLA" && cd "$GROOT_VLA"

WBC_COMMIT=1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b
ISAAC_COMMIT=e574928785
CKPT_REPO=cloudwalk-research/gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune
PY=$GROOT_VLA/.venv/bin/python
step() { echo; echo "== $*"; }

step "GR00T-WholeBodyControl @ ${WBC_COMMIT:0:7} (gear_sonic LFS assets only)"
if [ ! -d "$WBC/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$WBC"
  git -C "$WBC" checkout -q "$WBC_COMMIT"
  git -C "$WBC" lfs pull --include="gear_sonic/**,gear_sonic_deploy/**"
  rm -rf "$WBC/.git/lfs/objects"
fi

step "Isaac-GR00T @ $ISAAC_COMMIT (last Python 3.10 commit) + local patch"
if [ ! -d "$ISAAC_GROOT/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GROOT"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$ISAAC_GROOT" checkout -q "$ISAAC_COMMIT"
fi
PATCH=$GROOT_VLA_CODE/patches/isaac-gr00t-e574928-backbone-local.patch
if git -C "$ISAAC_GROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$ISAAC_GROOT" apply "$PATCH"
else
  git -C "$ISAAC_GROOT" apply --check --reverse "$PATCH" && echo "patch already applied"
fi

step "Python 3.10 venv (one env for policy server, VLA client and sim)"
if [ ! -x "$PY" ]; then
  uv python install 3.10 || true
  uv venv .venv --python "$(uv python find --no-project 3.10)" --prompt groot_vla
fi
export VIRTUAL_ENV=$GROOT_VLA/.venv
if ! "$PY" -c "import torch, flash_attn" 2>/dev/null; then
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
    "torch==2.7.1" "torchvision==0.22.1" "numpy==1.26.4"
  uv pip install --python "$PY" "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
fi
if ! "$PY" -c "import gr00t.eval.run_gr00t_server, gear_sonic, unitree_sdk2py, mujoco, pinocchio" 2>/dev/null; then
  # Isaac-GR00T and gear_sonic go in with --no-deps: only their inference/sim imports are installed
  # (no tensorrt-cu12 wheels, deepspeed, wandb). opencv < 4.12 keeps numpy 1.x for pinocchio.
  uv pip install --python "$PY" \
    "transformers==4.57.3" "diffusers==0.35.1" "scipy==1.15.3" "pandas==2.2.3" "tyro==0.9.17" \
    "albumentations==1.4.18" "pyzmq==27.0.1" "msgpack==1.1.0" "msgpack-numpy==0.4.8" "dm-tree" \
    "opencv-python>=4.5,<4.12" "pyyaml" "pillow" "tqdm" "termcolor" "huggingface-hub[cli]" \
    "mujoco" "pin" "glfw" "pygame" "loguru" "easydict" "joblib" \
    "numpy==1.26.4" "torch==2.7.1" "torchvision==0.22.1"
  uv pip install --python "$PY" --no-deps -e Isaac-GR00T -e GR00T-WholeBodyControl/gear_sonic
  uv pip install --python "$PY" -e GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python \
    "numpy==1.26.4" "torch==2.7.1" "opencv-python>=4.5,<4.12"
fi
for venv in .venv_inference .venv_sim; do
  [ -e "$WBC/$venv" ] || ln -s ../.venv "$WBC/$venv"
done

step "TensorRT 10.13 (system apt) and onnxruntime 1.16.3 for the SONIC deploy"
if ! ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.13* >/dev/null 2>&1; then
  echo "TensorRT 10.13 not found: install libnvinfer10=10.13.* and libnvinfer-dev (SONIC needs exactly 10.13 on x86)"; exit 1
fi
mkdir -p deps/tensorrt/include
for h in /usr/include/x86_64-linux-gnu/Nv*.h; do ln -sf "$h" deps/tensorrt/include/; done
ln -sfn /usr/lib/x86_64-linux-gnu deps/tensorrt/lib
if [ ! -d /opt/onnxruntime ] && [ ! -d deps/onnxruntime-linux-x64-1.16.3 ]; then
  curl -sSL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | tar -xz -C deps
fi
source "$GROOT_VLA_CODE/env.sh"

step "SONIC release policy + planner (reuse ~/.cache/huggingface GEAR-SONIC if present)"
D=$WBC/gear_sonic_deploy
if [ ! -e "$D/policy/release/model_encoder.onnx" ]; then
  SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/*/ 2>/dev/null | head -1)
  if [ -n "$SNAP" ] && [ -e "$SNAP/model_encoder.onnx" ] && [ -e "$SNAP/planner_sonic.onnx" ]; then
    ln -s "$SNAP/model_encoder.onnx" "$D/policy/release/model_encoder.onnx"
    ln -s "$SNAP/model_decoder.onnx" "$D/policy/release/model_decoder.onnx"
    mkdir -p "$D/planner/target_vel/V2"
    ln -s "$SNAP/planner_sonic.onnx" "$D/planner/target_vel/V2/planner_sonic.onnx"
  else
    (cd "$WBC" && "$PY" download_from_hf.py)
  fi
fi

step "Build gear_sonic_deploy"
if [ ! -x "$D/target/release/g1_deploy_onnx_ref" ]; then
  (cd "$D" && set +e && source scripts/setup_env.sh >/dev/null 2>&1; set -e
   mkdir -p build && cd build
   cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   cmake --build . -j"${BUILD_JOBS:-8}")
fi

step "VLA checkpoint ($CKPT_REPO, checkpoint-30000, ~6.9 GB)"
if [ ! -f "$VLA_CKPT/model-00002-of-00002.safetensors" ]; then
  "$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$CKPT_REPO", allow_patterns=["checkpoint-30000/*"],
                  local_dir="$GROOT_VLA/checkpoints/grab-bottle-v10")
EOF
fi

step "Cosmos-Reason2-2B config + tokenizer only (gated: accept the license on HF, then log in)"
if [ -z "$GR00T_BACKBONE_LOCAL_DIR" ]; then
  if [ -s "$HF_HOME/token" ]; then
    "$PY" -c 'from huggingface_hub import snapshot_download; snapshot_download("nvidia/Cosmos-Reason2-2B", allow_patterns=["*.json", "*.txt", "*.jinja"])'
  else
    echo "Accept https://huggingface.co/nvidia/Cosmos-Reason2-2B, then run:"
    echo "  source $GROOT_VLA_CODE/env.sh && $GROOT_VLA/.venv/bin/hf auth login && bash $GROOT_VLA_CODE/setup.sh"
    exit 1
  fi
fi

step "Done. Run: bash $GROOT_VLA_CODE/run.sh sim"
