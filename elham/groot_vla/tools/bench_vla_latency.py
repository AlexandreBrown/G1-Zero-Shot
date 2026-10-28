"""Measure GR00T policy-server latency the way the real robot uses it: no MuJoCo rendering on the GPU.

Replays observations recorded with `record_sim_video.py capture` (head-camera frames + SONIC state)
through the same PolicyClient call and observation layout as gear_sonic/scripts/run_vla_inference.py.
Needs the policy server running (run.sh starts it; nothing else has to run).

    source env.sh
    $GROOT_VLA/.venv/bin/python tools/bench_vla_latency.py --data run.pkl --calls 40
"""

import argparse
import pickle
import time

import cv2
import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.inference.vla_utils import prepare_observation_for_eval
from gr00t.policy.server_client import PolicyClient


def build_observation(sample: dict, robot_model, prompt: str) -> dict:
    """Same steps as run_vla_inference.prepare_observation_from_sensors (recorded base_quat_measured for gravity)."""
    left_hand = np.asarray(sample["left_hand_q"], dtype=float).copy()
    left_hand[5], left_hand[6] = left_hand[3], left_hand[4]
    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=np.asarray(sample["body_q"], dtype=float),
        left_hand_actuated_joint_values=left_hand,
        right_hand_actuated_joint_values=np.asarray(sample["right_hand_q"], dtype=float),
    )
    image = cv2.imdecode(np.frombuffer(sample["ego_jpg"], np.uint8), cv2.IMREAD_COLOR)
    observation = {
        "video": {"ego_view": image[np.newaxis, np.newaxis]},
        "state": {},
        "language": {"annotation.human.task_description": [[prompt]]},
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": sample["t"],
    }
    observation = prepare_observation_for_eval(robot_model, observation)
    gravity = compute_projected_gravity(np.asarray(sample["base_quat_measured"], dtype=np.float64))
    observation["state"]["projected_gravity"] = np.asarray(gravity, dtype=np.float32)[np.newaxis, np.newaxis]
    return observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prompt", default="grab the bottle")
    parser.add_argument("--port", type=int, default=5550)
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        samples = pickle.load(f)["samples"]
    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    policy = PolicyClient(host="127.0.0.1", port=args.port)
    assert policy.ping(), "policy server not reachable on 127.0.0.1:%d" % args.port

    stride = max(1, len(samples) // (args.calls + args.warmup))
    latencies = []
    for k in range(args.calls + args.warmup):
        observation = build_observation(samples[(k * stride) % len(samples)], robot_model, args.prompt)
        start = time.monotonic()
        action, _ = policy.get_action(observation)
        if k >= args.warmup:
            latencies.append(time.monotonic() - start)
    latencies = np.array(latencies)
    token = action.get("motion_token", action.get("action.motion_token"))
    print(f"action motion_token shape {np.shape(token)}")
    print(f"latency over {len(latencies)} calls: mean {latencies.mean():.3f}s  "
          f"p50 {np.median(latencies):.3f}s  p90 {np.percentile(latencies, 90):.3f}s  max {latencies.max():.3f}s")
    chunk_s = 40 / 50
    print(f"chunk covers {chunk_s:.2f}s; tutorial target 0.40s (2.5 Hz); "
          f"fraction of chunk already stale at mean latency: {min(1.0, latencies.mean() / chunk_s):.0%}")


if __name__ == "__main__":
    main()
