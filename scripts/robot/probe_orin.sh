#!/usr/bin/env bash
# ROBOT (Orin): read-only inspection. No DDS publisher, controller, or sudo.
set -u
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset LD_LIBRARY_PATH LD_PRELOAD
echo 'SONIC ORIN READ-ONLY PREFLIGHT — NO MOTOR COMMANDS'
echo '--- Architecture / OS ---'
uname -m
cat /etc/os-release
echo '--- Jetson model ---'
if [ -r /proc/device-tree/model ]; then tr -d '\0' < /proc/device-tree/model; echo; fi
echo '--- L4T / JetPack / TensorRT packages ---'
cat /etc/nv_tegra_release 2>/dev/null || true
dpkg-query -W -f='${Package}\t${Version}\n' \
  'nvidia-l4t-core' 'nvidia-jetpack' 'libnvinfer*' 'libnvonnxparsers*' \
  'libcudla*' 'nvidia-l4t-dla-compiler' 2>/dev/null || true
echo '--- CUDA ---'
if [ -x /usr/local/cuda/bin/nvcc ]; then /usr/local/cuda/bin/nvcc --version; fi
echo '--- Compiler and build tools ---'
command -v g++ cmake git pkg-config python3 || true
g++ --version 2>/dev/null | head -1
echo '--- Network interfaces (choose robot-facing interface explicitly later) ---'
ip -br -4 addr
ip route
echo '--- Memory / storage ---'
free -h
df -h "$HOME" /tmp
echo '--- Existing SONIC/simulator processes ---'
pgrep -af '[g]1_deploy_onnx_ref|[r]un_sim_loop.py|[s]onic_headless.py' || true
echo '--- Development headers ---'
for header in /usr/include/aarch64-linux-gnu/NvInfer.h \
  /usr/include/NvInfer.h /usr/include/yaml-cpp/yaml.h \
  /usr/include/eigen3/Eigen/Core /usr/include/zmq.hpp \
  /usr/include/msgpack.hpp /usr/include/nlohmann/json.hpp; do
  if [ -f "$header" ]; then echo "FOUND $header"; fi
done
echo '--- TensorRT version header ---'
for header in /usr/include/aarch64-linux-gnu/NvInferVersion.h /usr/include/NvInferVersion.h; do
  if [ -f "$header" ]; then rg '^#define NV_TENSORRT_(MAJOR|MINOR|PATCH)' "$header" 2>/dev/null || grep '^#define NV_TENSORRT_' "$header"; fi
done
echo 'END OF READ-ONLY REPORT — controller was not started'
