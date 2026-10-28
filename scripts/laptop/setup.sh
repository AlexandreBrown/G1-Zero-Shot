#!/usr/bin/env bash
# LOCAL GPU laptop only. Installs dependencies/builds; never starts a controller.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="${SONIC_UPSTREAM:-$(dirname "$ROOT")/GR00T-WholeBodyControl}"
DEPS="$ROOT/.runtime/sonic/deps"
[[ $(uname -m) == x86_64 ]] || { echo 'This setup is for an x86_64 GPU laptop.'; exit 1; }
nvidia-smi
mkdir -p "$DEPS"
PACKAGES=(build-essential cmake git git-lfs curl python3-venv python3-tk \
 libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev \
 nlohmann-json3-dev zlib1g-dev libgtest-dev libboost-dev libgl1 libglfw3 fonts-dejavu-core)
# Ubuntu 22.04 ships the C++ msgpack/cppzmq headers inside libmsgpack-dev/libzmq3-dev.
[[ $(. /etc/os-release && echo "$VERSION_ID") == 22.04 ]] || PACKAGES+=(libmsgpack-cxx-dev cppzmq-dev)
MISSING=()
for package in "${PACKAGES[@]}"; do
 dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed' || MISSING+=("$package")
done
if ((${#MISSING[@]})); then
 sudo apt-get update
 sudo apt-get install -y "${MISSING[@]}"
fi
source "$ROOT/scripts/laptop/source.sh"
ensure_sonic_source
# Require a working CUDA 12 toolkit; do not replace a laptop's NVIDIA driver.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -f "$CUDA_HOME/include/cuda_runtime.h" ]]; then
 echo 'Install the NVIDIA CUDA 12 toolkit (not just a driver), then rerun with CUDA_HOME set.'
 echo 'Official instructions: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/'
 exit 1
fi
TensorRT_ROOT="${TensorRT_ROOT:-$DEPS/TensorRT-10.13.3.9}"
if [[ ! -f "$TensorRT_ROOT/include/NvInfer.h" ]]; then
 [[ "$TensorRT_ROOT" == "$DEPS/TensorRT-10.13.3.9" ]] || { echo "Invalid TensorRT_ROOT: $TensorRT_ROOT"; exit 1; }
 echo 'Downloading NVIDIA TensorRT 10.13.3.9 (~7 GB); allow ample extraction space.'
 curl --fail --location --retry 3 --continue-at - \
  'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz' \
  -o "$DEPS/tensorrt.tar.gz"
 tar -xzf "$DEPS/tensorrt.tar.gz" -C "$DEPS" --exclude='*.a' --exclude='*/samples/*' --exclude='*/python/*'
fi
# TensorRT 10.13 headers define NV_TENSORRT_MAJOR via TRT_MAJOR_ENTERPRISE; resolve that indirection.
TRT_VERSION=$(awk '/^#define/ {v[$2]=$3} END {ma=v["NV_TENSORRT_MAJOR"]; if (ma in v) ma=v[ma];
 mi=v["NV_TENSORRT_MINOR"]; if (mi in v) mi=v[mi]; print ma "." mi}' "$TensorRT_ROOT/include/NvInferVersion.h")
[[ $TRT_VERSION == 10.13 ]] || { echo "TensorRT $TRT_VERSION at $TensorRT_ROOT; 10.13 is required."; exit 1; }
onnxruntime_ROOT="$DEPS/onnxruntime-linux-x64-1.16.3"
if [[ ! -f "$onnxruntime_ROOT/include/onnxruntime_cxx_api.h" ]]; then
 curl --fail --location --retry 3 'https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz' -o "$DEPS/onnxruntime.tgz"
 tar -xzf "$DEPS/onnxruntime.tgz" -C "$DEPS"
fi
/usr/bin/python3 -m venv "$ROOT/.venv_laptop"
"$ROOT/.venv_laptop/bin/pip" install -r "$ROOT/sonic_laptop/requirements.txt"
# Persist only paths; no conda/Mila/OpenSSL paths in this laptop environment.
ENV_FILE="$ROOT/.runtime/sonic/env.sh"
printf 'export SONIC_UPSTREAM=%q\nexport CUDA_HOME=%q\nexport CUDAToolkit_ROOT=%q\nexport TensorRT_ROOT=%q\nexport onnxruntime_ROOT=%q\n' \
 "$UPSTREAM" "$CUDA_HOME" "$CUDA_HOME" "$TensorRT_ROOT" "$onnxruntime_ROOT" > "$ENV_FILE"
printf 'export LD_LIBRARY_PATH="%s/lib:%s/lib:%s/lib64:${LD_LIBRARY_PATH:-}"\n' "$TensorRT_ROOT" "$onnxruntime_ROOT" "$CUDA_HOME" >> "$ENV_FILE"
printf 'export PATH="%s/bin:$PATH"\nexport HAS_ROS2=0\n' "$ROOT/.venv_laptop" >> "$ENV_FILE"
source "$ENV_FILE"
# download_from_hf.py copies read-only Hugging Face cache files; a rerun must be able to overwrite them.
for models in policy planner; do
 if [[ -d "$UPSTREAM/gear_sonic_deploy/$models" ]]; then chmod -R u+w "$UPSTREAM/gear_sonic_deploy/$models"; fi
done
(cd "$UPSTREAM" && python download_from_hf.py)
cd "$ROOT"
# Re-running setup refreshes generated source while preserving the previous build.
if [[ -e "$ROOT/.runtime/sonic/deploy" ]]; then
 mv "$ROOT/.runtime/sonic/deploy" "$ROOT/.runtime/sonic/deploy-backup-$(date +%Y%m%d-%H%M%S)-$$"
fi
python -m sonic_laptop.prepare --upstream "$UPSTREAM" --output "$ROOT/.runtime/sonic/deploy"
bash "$ROOT/scripts/laptop/build.sh"
echo 'LOCAL setup complete. Persistent installation: .runtime/sonic/'
echo 'Next: bash scripts/laptop/run.sh --interface YOUR_ETHERNET_INTERFACE' 
