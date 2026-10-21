# Source this file on a Mila compute node before building or launching SONIC.
module load cuda/12.6.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDAToolkit_ROOT="$CUDA_HOME"
export TensorRT_ROOT="$HOME/Documents/g1_zero/deps/TensorRT-10.13.3.9"
export onnxruntime_ROOT="$HOME/Documents/g1_zero/deps/onnxruntime-linux-x64-1.16.3"
export SONIC_CPP_PREFIX="$HOME/Documents/g1_zero/deps/cpp"
export CMAKE_PREFIX_PATH="$SONIC_CPP_PREFIX:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:$SONIC_CPP_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$TensorRT_ROOT/bin:$HOME/.local/bin:$PATH"
export PKG_CONFIG_PATH="$SONIC_CPP_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export HAS_ROS2=0
