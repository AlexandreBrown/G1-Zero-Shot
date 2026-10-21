#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/sonic_env.sh"
cd "$HOME/Documents/g1_zero/GR00T-WholeBodyControl/gear_sonic_deploy"
exec target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type keyboard --output-type zmq --disable-crc-check
