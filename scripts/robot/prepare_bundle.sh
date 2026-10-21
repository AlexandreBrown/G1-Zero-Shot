#!/usr/bin/env bash
# LOCAL: create an offline source/model bundle. Never connects to the robot.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="${SONIC_UPSTREAM_ROOT:-$(dirname "$REPO_ROOT")/GR00T-WholeBodyControl}"
OUTPUT="$REPO_ROOT/artifacts/orin"
mkdir -p "$OUTPUT"
for relative in policy/release/model_encoder.onnx policy/release/model_decoder.onnx \
  policy/release/observation_config.yaml planner/target_vel/V2/planner_sonic.onnx; do
  test -s "$UPSTREAM/gear_sonic_deploy/$relative"
  if head -c 128 "$UPSTREAM/gear_sonic_deploy/$relative" | grep -q 'version https://git-lfs.github.com/spec'; then
    echo "Git LFS pointer instead of downloaded asset: $relative" >&2; exit 1
  fi
done
tar -czf "$OUTPUT/sonic-orin-source-models.tar.gz" \
  --exclude='gear_sonic_deploy/build' --exclude='gear_sonic_deploy/target' \
  --exclude='gear_sonic_deploy/logs' --exclude='*.trt' --exclude='*.engine' \
  --exclude='*.o' --exclude='__pycache__' \
  -C "$UPSTREAM" gear_sonic_deploy LICENSE legal \
  -C "$REPO_ROOT" scripts/robot docs/sonic-orin.md
git -C "$UPSTREAM" rev-parse HEAD > "$OUTPUT/upstream-commit.txt"
(cd "$OUTPUT" && sha256sum sonic-orin-source-models.tar.gz > SHA256SUMS)
echo "Prepared $OUTPUT/sonic-orin-source-models.tar.gz"
echo 'Contains source and ONNX models; excludes x86 build outputs and TensorRT engines.'
echo 'ARM64 dependencies and engines must be built for the Orin after preflight.'
