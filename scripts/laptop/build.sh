#!/usr/bin/env bash
# LOCAL. Builds only the isolated guarded tree, never the stock upstream controller.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifacts/laptop/env.sh"
DEPLOY="$ROOT/artifacts/laptop/deploy"
[[ $(cat "$DEPLOY/GUARDED_BUILD") == sonic-laptop-v3 ]]
cmake -S "$DEPLOY" -B "$DEPLOY/build" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
 -DTensorRT_FIND_COMPONENTS='nvinfer;nvinfer_plugin;nvonnxparser' \
 -DCMAKE_PREFIX_PATH="$TensorRT_ROOT;$onnxruntime_ROOT;$CUDA_HOME"
cmake --build "$DEPLOY/build" --target g1_deploy_onnx_ref -j "${BUILD_JOBS:-2}"
python - "$DEPLOY" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); binary=p/'target/release/g1_deploy_onnx_ref'
(p/'binary-manifest.json').write_text(json.dumps({'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2))
PY
