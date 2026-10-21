"""Pack trial folders (sim2sim or real G1) into dataset episodes: <trial>/episode.npz + <trial>/episode.json.

Per trial (scripts/sonic_run.sh output) the episode holds, aligned on the deploy program's control ticks (50 Hz,
CONTROL state, by the state logger's entry index):
  t_monotonic, t_realtime       seconds
  q, dq, tau_est                joint state and estimated torque, HARDWARE order (29)
  motor_temperature             (29, 2)     motor_error (29)
  action                        last_action at tick k (raw policy output of tick k-1), IsaacLab order (29)
  q_target                      motor targets active while q[k] was measured, HARDWARE order (29), before
                                --kp/--kd scaling
  action_out, q_target_out      the policy's output for the inputs of tick k (= action[k+1]); NaN on the last tick
  base_quat_wxyz, base_ang_vel, base_accel, torso_quat_wxyz, torso_ang_vel, torso_accel
  token (64), encoder_mode, motion_playing (bool), motion_name (str per tick)
  policy_input                  decoder input vector (994) per tick, if --policy-input-logfile rows match
  ref_pelvis_pos, ref_pelvis_quat_wxyz, ref_q   the reference the policy tracked (hardware order), if rows match
and, when the full-rate recorder ran (<trial>/hw), every LowState / LowCmd / torso-IMU message as hw_<stream>_<field>
(t on the same CLOCK_MONOTONIC clock when the recorder ran on the deploy computer).
episode.json: row counts, alignment notes, run_manifest.txt, policy manifest and the program's metadata.json.

    python scripts/build_dataset.py logs/<session>/*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deploy_logs as dl  # noqa: E402
import session_recorder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NUM_JOINTS = 29
STATE_FILES = {
    "q": "q",
    "dq": "dq",
    "tau_est": "motor_torque",
    "motor_temperature": "motor_temperature",
    "motor_error": "motor_error",
    "action": "action",
    "base_quat_wxyz": "base_quat",
    "base_ang_vel": "base_ang_vel",
    "base_accel": "base_accel",
    "torso_quat_wxyz": "torso_quat",
    "torso_ang_vel": "torso_ang_vel",
    "torso_accel": "torso_accel",
}
POST_FILES = {"token": "token_state", "encoder_mode": "encoder_mode", "motion_playing": "motion_playing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", type=Path)
    return parser.parse_args()


def deploy_constants() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    constants = yaml.safe_load((REPO / "configs" / "sonic_deploy_constants.yaml").read_text())
    rows = constants["joints"]
    return (
        np.array(constants["isaaclab_to_hw"]),
        np.array([r["default"] for r in rows]),
        np.array([r["action_scale"] for r in rows]),
    )


def joint_targets(action_isaaclab: np.ndarray) -> np.ndarray:
    """q_target[hw i] = default[i] + action[isaaclab_to_hw[i]] * scale[i] (g1_deploy_onnx_ref control step)."""
    isaaclab_to_hw, default, scale = deploy_constants()
    return default + action_isaaclab[:, isaaclab_to_hw] * scale


def build_episode(trial: Path) -> tuple[dict[str, np.ndarray], dict]:
    state = trial / "state"
    base = dl.read_logged(state / "q.csv")
    arrays: dict[str, np.ndarray] = {"t_monotonic": base.t_monotonic, "t_realtime": base.t_realtime}
    meta: dict = {"trial": str(trial), "ticks": int(len(base.index)), "rows": {}}
    for key, name in STATE_FILES.items():
        logged = dl.read_logged(state / f"{name}.csv")
        meta["rows"][name] = int(len(logged.index))
        arrays[key] = dl.align(base.index, logged)
    arrays["q"] = arrays["q"][:, :NUM_JOINTS]
    arrays["dq"] = arrays["dq"][:, :NUM_JOINTS]
    arrays["tau_est"] = arrays["tau_est"][:, :NUM_JOINTS]
    arrays["motor_error"] = arrays["motor_error"][:, :NUM_JOINTS]
    arrays["motor_temperature"] = arrays["motor_temperature"].reshape(len(base.index), NUM_JOINTS, -1)
    arrays["action"] = arrays["action"][:, :NUM_JOINTS]
    arrays["q_target"] = joint_targets(arrays["action"])
    # The program logs last_action BEFORE inference: action[k] was decided at tick k-1 (it is the last_action
    # observation of tick k, and q_target[k] is the target active while q[k] was measured). The policy's output for
    # the inputs of tick k (policy_input[k], token[k]) is action[k + 1]: stored as action_out / q_target_out.
    arrays["action_out"] = np.vstack((arrays["action"][1:], np.full((1, NUM_JOINTS), np.nan)))
    arrays["q_target_out"] = np.vstack((arrays["q_target"][1:], np.full((1, NUM_JOINTS), np.nan)))
    for key, name in POST_FILES.items():
        path = state / f"{name}.csv"
        if path.exists() and path.stat().st_size > 0:
            logged = dl.read_logged(path)
            meta["rows"][name] = int(len(logged.index))
            arrays[key] = dl.align(base.index, logged)
    if "motion_playing" in arrays:
        arrays["motion_playing"] = arrays["motion_playing"][:, 0] > 0.5
    if "encoder_mode" in arrays:
        arrays["encoder_mode"] = arrays["encoder_mode"][:, 0]
    names_path = state / "motion_name.csv"
    if names_path.exists() and names_path.stat().st_size > 0:
        index, names = dl.read_motion_names(names_path)
        lookup = dict(zip(index.tolist(), names))
        arrays["motion_name"] = np.array([lookup.get(int(i), "") for i in base.index])

    for key, filename, width in (("policy_input", "obs.csv", None), ("target", "target.csv", 36)):
        path = trial / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        rows = dl.read_unindexed(path)
        meta["rows"][filename] = int(len(rows))
        if len(rows) != len(base.index):
            meta.setdefault("not_aligned", []).append(
                f"{filename}: {len(rows)} rows vs {len(base.index)} ticks; stored unaligned as {key}_unaligned"
            )
            arrays[f"{key}_unaligned"] = rows
            continue
        if key == "target":
            arrays["ref_pelvis_pos"] = rows[:, 0:3]
            arrays["ref_pelvis_quat_wxyz"] = rows[:, 3:7]
            arrays["ref_q"] = rows[:, 7:width]
        else:
            arrays[key] = rows

    hw = trial / "hw"
    if hw.is_dir() and any(hw.glob("chunk_*.npz")):
        for name, values in session_recorder.merge_chunks(hw).items():
            arrays["hw_" + name.replace("/", "_")] = values
        recorder_summary = hw / "recorder.json"
        if recorder_summary.exists():
            meta["recorder"] = json.loads(recorder_summary.read_text())

    for name in ("run_manifest.txt",):
        if (trial / name).exists():
            meta[name] = (trial / name).read_text()
    for name, path in (
        ("policy_manifest", trial / "policy_manifest.json"),
        ("deploy_metadata", state / "metadata.json"),
    ):
        if path.exists():
            meta[name] = json.loads(path.read_text())
    meta["arrays"] = {k: list(v.shape) for k, v in arrays.items()}
    return arrays, meta


def main() -> None:
    args = parse_args()
    for trial in args.trials:
        if not dl.is_trial(trial):
            print(f"{trial}: skipped (no logged control ticks: not a trial folder, or stopped before ])")
            continue
        arrays, meta = build_episode(trial)
        np.savez_compressed(trial / "episode.npz", **arrays)
        (trial / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
        notes = "; ".join(meta.get("not_aligned", [])) or "all streams aligned"
        print(f"{trial}: {meta['ticks']} ticks, {len(arrays)} arrays ({notes})")


if __name__ == "__main__":
    main()
