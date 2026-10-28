# Source this before any groot_vla command.
# GROOT_VLA_HOME: where the venv, caches, upstream clones and models live (~17 GB); defaults to this folder.
export GROOT_VLA_CODE=$(dirname "$(readlink -f "${BASH_SOURCE[0]:-${(%):-%x}}")")
export GROOT_VLA=$(readlink -f "${GROOT_VLA_HOME:-$GROOT_VLA_CODE}")
export UV_CACHE_DIR=$GROOT_VLA/.cache/uv
export UV_PYTHON_INSTALL_DIR=$GROOT_VLA/.python
export HF_HOME=$GROOT_VLA/.cache/huggingface
export WBC=$GROOT_VLA/GR00T-WholeBodyControl
export ISAAC_GROOT=$GROOT_VLA/Isaac-GR00T
export VLA_CKPT=$GROOT_VLA/checkpoints/grab-bottle-v10/checkpoint-30000
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TensorRT_ROOT=$GROOT_VLA/deps/tensorrt
if [ -d "$GROOT_VLA/deps/onnxruntime-linux-x64-1.16.3" ]; then
  export onnxruntime_ROOT=$GROOT_VLA/deps/onnxruntime-linux-x64-1.16.3
else
  export onnxruntime_ROOT=/opt/onnxruntime
fi
export LD_LIBRARY_PATH="$onnxruntime_ROOT/lib:/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export GR00T_BACKBONE_LOCAL_DIR=$(ls -d $HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/*/ 2>/dev/null | head -1)
