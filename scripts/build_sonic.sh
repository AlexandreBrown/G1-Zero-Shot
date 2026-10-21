#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sonic_env.sh"
cd "$HOME/Documents/g1_zero/GR00T-WholeBodyControl/gear_sonic_deploy"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DTensorRT_ROOT="$TensorRT_ROOT" \
  -DTensorRT_FIND_COMPONENTS='nvinfer;nvinfer_plugin;nvonnxparser' \
  -DCMAKE_LIBRARY_PATH="$TensorRT_ROOT/lib;$SONIC_CPP_PREFIX/lib" \
  -DCMAKE_INCLUDE_PATH="$SONIC_CPP_PREFIX/include" \
  -DCMAKE_CXX_FLAGS="-isystem $SONIC_CPP_PREFIX/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$SONIC_CPP_PREFIX/lib -Wl,-rpath,$SONIC_CPP_PREFIX/lib"
cmake --build build --target g1_deploy_onnx_ref -j "${SLURM_CPUS_PER_TASK:-4}"
