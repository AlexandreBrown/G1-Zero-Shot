"""Preview a trial in MuJoCo with the exact ONNX files and deploy-format motion, as the robot will run it.

The sequence copies the deploy program's state machine (g1_deploy_onnx_ref):
  1. the robot stands at the default pose on the floor (the moment you press ] after lowering it);
  2. ] : the policy starts; the reference is paused on frame 0 (all 10 future frames = frame 0, zero velocities)
         and its heading is re-anchored to the robot's yaw; held for --hold-s seconds. As in the deploy program,
         the 10-step state history starts zero-padded (gravity reads +z in the padding) and fills over 9 ticks;
  3. T : the motion plays, one reference frame per 20 ms control tick;
  4. at the end the reference snaps back to frame 0, paused, with the heading re-anchored to the robot (the deploy
     program does both), held for --tail-s.
Physics is SONIC's own sim2sim scene (scene_43dof.xml, 200 Hz, PD every 5 ms with the deploy gains), the torque
limits are the deploy simulator's (88 Nm hip pitch/roll) unless --torque-limits training. Inference is ONNX
Runtime on CPU; there is no DDS latency and no TensorRT, so this is a preview and a quick check, not the gate
(the robot runs the real deploy binary).

Writes <out>.mp4 (50 fps, real time, camera following the pelvis), <out>.json (fell, max tilt, lowest pelvis,
peak torque and speed against configs/g1_joint_limits.yaml, tracking error while playing) and <out>.npz (per
control step: phase, reference frame, action, joint positions and peak torques in hardware order, tilt, pelvis).

    MUJOCO_GL=egl python scripts/preview_mujoco.py --sonic-root $SONIC_ROOT --policy artifacts/policies/base \
        --motion artifacts/motions/single/squat_lv1 --out previews/base_squat_lv1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g1_limits  # noqa: E402
from rotations import quat_apply, quat_conj, quat_mul, rotation_6d  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCENE = Path("gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
SIM_DT, DECIMATION, CONTROL_DT = 0.005, 4, 0.02
HISTORY, NUM_FUTURE, FRAME_STEP = 10, 10, 5
HAND_KP, HAND_KD = 1.5, 0.1
ARMATURE_7520_14 = 0.010177520
FALL_HEIGHT = 0.45
# The deploy program's state history starts empty at ]: GetLatest pads with zero entries, and a zero quaternion
# becomes gravity (0, 0, +1) in GatherHisGravityDir. So for the first 9 control ticks the policy sees this entry
# as the older history (state_logger.cpp GetLatest / makeZeroEntry_, g1_deploy_onnx_ref.cpp GatherHisGravityDir).
STARTUP_HISTORY_ENTRY = [np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29), np.array([0.0, 0.0, 1.0])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, required=True, help="folder with model_encoder.onnx / model_decoder.onnx"
    )
    parser.add_argument("--motion", type=Path, required=True, help="a single-motion folder from export_motions.py")
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    parser.add_argument("--hold-s", type=float, default=2.0, help="seconds paused on frame 0 before playing")
    parser.add_argument("--tail-s", type=float, default=2.0, help="seconds after the motion ends")
    parser.add_argument("--torque-limits", choices=("deploy", "training"), default="deploy")
    parser.add_argument(
        "--revision",
        choices=("scene", "old", "new"),
        default="scene",
        help="scene: SONIC's MuJoCo model as is (armature 0.01 everywhere). old / new: per-joint rotor inertia of the "
        "real motors (from the deploy gains) with the old 7520-14 (88 Nm) or new 7520-22 (139 Nm) hip-pitch motor; "
        "overrides --torque-limits",
    )
    parser.add_argument(
        "--delay-ms", type=int, default=0, help="action latency, multiple of 5 ms (the real robot: about 2-5 ms)"
    )
    parser.add_argument("--kp-scale", default="", help="per-motor Kp factors, deploy syntax: '0,6=0.75' or '0-28=0.8'")
    parser.add_argument("--kd-scale", default="", help="per-motor Kd factors, same syntax")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def gain_factors(spec: str) -> np.ndarray:
    """Per-motor factors from the deploy program's --motor-kp-scale syntax (hardware indices: '0,6=0.75;12-14=1.1')."""
    factors = np.ones(29)
    for part in filter(None, spec.split(";")):
        indices, value = part.split("=")
        for item in indices.split(","):
            lo, _, hi = item.partition("-")
            factors[int(lo) : int(hi or lo) + 1] = float(value)
    return factors


def find_motion_folder(path: Path) -> Path:
    if (path / "joint_pos.csv").exists():
        return path
    (folder,) = [p.parent for p in path.glob("*/joint_pos.csv")]
    return folder


def load_reference(folder: Path) -> dict:
    def load(name: str) -> np.ndarray:
        return np.loadtxt(folder / name, delimiter=",", skiprows=1, ndmin=2)

    return {
        "joint_pos": load("joint_pos.csv"),
        "joint_vel": load("joint_vel.csv"),
        "pelvis_pos": load("body_pos.csv")[:, 0:3],
        "pelvis_quat": load("body_quat.csv")[:, 0:4],
    }


def yaw_of(quat: np.ndarray) -> float:
    """Heading as the deploy program computes it: atan2 of the rotated forward axis."""
    forward = quat_apply(quat, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(forward[1], forward[0]))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Robot:
    """SONIC's MuJoCo G1 with the deploy PD law; joints indexed in hardware order."""

    def __init__(self, sonic_root: Path, torque_limits: str, delay_ms: int = 0, revision: str = "scene") -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(sonic_root / SCENE))
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        constants = yaml.safe_load((REPO / "configs" / "sonic_deploy_constants.yaml").read_text())
        rows = constants["joints"]
        self.isaaclab_to_hw = np.array(constants["isaaclab_to_hw"])
        self.hw_to_isaaclab = np.argsort(self.isaaclab_to_hw)
        self.kp = np.array([r["kp"] for r in rows])
        self.kd = np.array([r["kd"] for r in rows])
        self.scale = np.array([r["action_scale"] for r in rows])
        self.default = np.array([r["default"] for r in rows])
        self.names = [r["name"] for r in rows]
        joint_ids = [self.model.joint(n).id for n in self.names]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        actuator_of_joint = {int(self.model.actuator_trnid[a, 0]): a for a in range(self.model.nu)}
        self.act = np.array([actuator_of_joint[j] for j in joint_ids])
        hands = [j for j in range(self.model.njnt) if "_hand_" in self.model.joint(j).name]
        self.hand_qadr = np.array([self.model.jnt_qposadr[j] for j in hands], dtype=int)
        self.hand_dadr = np.array([self.model.jnt_dofadr[j] for j in hands], dtype=int)
        self.hand_act = np.array([actuator_of_joint[j] for j in hands], dtype=int)
        table = g1_limits.load_joint_table()
        self.effort = g1_limits.deploy_sim_effort(table) if torque_limits == "deploy" else table.tau_train.copy()
        if revision != "scene":
            # Kp = armature * (2 pi 10 Hz)^2 in the deploy program, so the gains give each motor's rotor inertia.
            armature = self.kp / (2 * np.pi * 10.0) ** 2
            self.effort = table.tau_train.copy()
            for i, name in enumerate(self.names):
                if "hip_pitch" in name and revision == "old":
                    armature[i] = ARMATURE_7520_14
                    self.effort[i] = 88.0
            self.model.dof_armature[self.dadr] = armature
        self.pelvis = self.model.body("pelvis").id
        if delay_ms % round(SIM_DT * 1000):
            raise ValueError(f"--delay-ms must be a multiple of {round(SIM_DT * 1000)} ms")
        self.pending = [self.default.copy()] * (delay_ms // round(SIM_DT * 1000))

    def reset_standing(self) -> None:
        d = self.data
        self.mujoco.mj_resetData(self.model, d)
        d.qpos[0:3] = (0.0, 0.0, 1.0)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = self.default
        self.mujoco.mj_forward(self.model, d)
        foot_geoms = [
            g for g in range(self.model.ngeom) if "ankle_roll" in self.model.body(self.model.geom_bodyid[g]).name
        ]
        lowest = min(d.geom_xpos[g, 2] - self.model.geom_rbound[g] for g in foot_geoms if self.model.geom_contype[g])
        d.qpos[2] -= lowest - 0.001
        self.mujoco.mj_forward(self.model, d)

    def step(self, action_isaaclab: np.ndarray) -> np.ndarray:
        """Apply one 20 ms control step; returns the peak |torque| per joint over the substeps."""
        d = self.data
        new_target = self.default + action_isaaclab[self.isaaclab_to_hw] * self.scale
        peak = np.zeros(len(self.names))
        for _ in range(DECIMATION):
            # with a latency of n physics steps, the target computed now reaches the motors n steps later
            self.pending.append(new_target)
            target = self.pending.pop(0)
            tau = np.clip(
                self.kp * (target - d.qpos[self.qadr]) - self.kd * d.qvel[self.dadr], -self.effort, self.effort
            )
            d.ctrl[self.act] = tau
            d.ctrl[self.hand_act] = -HAND_KP * d.qpos[self.hand_qadr] - HAND_KD * d.qvel[self.hand_dadr]
            self.mujoco.mj_step(self.model, d)
            peak = np.maximum(peak, np.abs(tau))
        return peak

    def proprio(self, last_action: np.ndarray) -> list[np.ndarray]:
        d = self.data
        quat = d.qpos[3:7].copy()
        q = d.qpos[self.qadr][self.hw_to_isaaclab]
        dq = d.qvel[self.dadr][self.hw_to_isaaclab]
        gravity = quat_apply(quat_conj(quat), np.array([0.0, 0.0, -1.0]))
        return [d.qvel[3:6].copy(), q - self.default[self.hw_to_isaaclab], dq, last_action.copy(), gravity]


def encoder_input(
    reference: dict, frame: int, playing: bool, robot_quat: np.ndarray, yaw_offset: np.ndarray, encoder_dim: int
) -> np.ndarray:
    length = len(reference["joint_pos"])
    if playing:
        frames = np.minimum(frame + FRAME_STEP * np.arange(NUM_FUTURE), length - 1)
        velocities = reference["joint_vel"][frames]
    else:
        frames = np.full(NUM_FUTURE, frame)
        velocities = np.zeros((NUM_FUTURE, reference["joint_vel"].shape[1]))
    command = np.concatenate((reference["joint_pos"][frames].reshape(-1), velocities.reshape(-1)))
    ref_quat = quat_mul(yaw_offset[None], reference["pelvis_quat"][frames])
    anchor = rotation_6d(quat_mul(quat_conj(robot_quat)[None], ref_quat)).reshape(-1)
    vector = np.zeros(encoder_dim, dtype=np.float32)
    head = np.concatenate(([0.0, 0.0, 0.0, 0.0], command, anchor))
    vector[: len(head)] = head
    return vector[None]


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    sonic_root = args.sonic_root.resolve()
    folder = find_motion_folder(args.motion.resolve())
    reference = load_reference(folder)
    robot = Robot(sonic_root, args.torque_limits, args.delay_ms, args.revision)
    robot.kp = robot.kp * gain_factors(args.kp_scale)
    robot.kd = robot.kd * gain_factors(args.kd_scale)
    table = g1_limits.load_joint_table()
    thresholds = g1_limits.Thresholds()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    encoder = ort.InferenceSession(str(args.policy / "model_encoder.onnx"), options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(args.policy / "model_decoder.onnx"), options, providers=["CPUExecutionProvider"])
    encoder_dim = encoder.get_inputs()[0].shape[-1]

    robot.reset_standing()
    yaw_offset = yaw_quat(yaw_of(robot.data.qpos[3:7]) - yaw_of(reference["pelvis_quat"][0]))
    hold_steps, tail_steps = round(args.hold_s / CONTROL_DT), round(args.tail_s / CONTROL_DT)
    length = len(reference["joint_pos"])
    schedule = [(0, False)] * hold_steps + [(t, True) for t in range(length)] + [(0, False)] * tail_steps

    renderer = writer = None
    if not args.no_video:
        renderer = robot.mujoco.Renderer(robot.model, args.height, args.width)
        camera = robot.mujoco.MjvCamera()
        camera.type = robot.mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = robot.pelvis
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -15.0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                str(round(1 / CONTROL_DT)),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(args.out.with_suffix(".mp4")),
            ],
            stdin=subprocess.PIPE,
        )

    last_action = np.zeros(29)
    history = [STARTUP_HISTORY_ENTRY] * (HISTORY - 1) + [robot.proprio(last_action)]
    peak_torque = np.zeros(29)
    peak_speed = np.zeros(29)
    min_margin = np.full(29, np.inf)
    max_tilt, min_pelvis, fell_at = 0.0, np.inf, None
    tracking = []
    trace = {
        key: []
        for key in ("t", "playing", "frame", "action", "q", "peak_torque", "tilt_deg", "pelvis_pos", "lean_deg",
                    "ref_lean_deg")
    }
    was_playing = False
    for step, (frame, playing) in enumerate(schedule):
        d = robot.data
        if was_playing and not playing:
            # end of motion: the deploy program re-anchors the heading to the robot's yaw (reinitialize_heading_)
            yaw_offset = yaw_quat(yaw_of(d.qpos[3:7]) - yaw_of(reference["pelvis_quat"][0]))
        was_playing = playing
        enc_in = encoder_input(reference, frame, playing, d.qpos[3:7].copy(), yaw_offset, encoder_dim)
        token = encoder.run(None, {"obs_dict": enc_in})[0]
        actor_obs = np.concatenate([np.concatenate([h[term] for h in history]) for term in range(5)])
        dec_in = np.concatenate((token[0], actor_obs)).astype(np.float32)[None]
        action = decoder.run(None, {"obs_dict": dec_in})[0][0].astype(np.float64)
        step_torque = robot.step(action)
        peak_torque = np.maximum(peak_torque, step_torque)
        last_action = action
        history = history[1:] + [robot.proprio(last_action)]
        q_hw = d.qpos[robot.qadr]
        peak_speed = np.maximum(peak_speed, np.abs(d.qvel[robot.dadr]))
        min_margin = np.minimum(min_margin, g1_limits.position_margin(q_hw, table))
        max_tilt = max(max_tilt, float(g1_limits.tilt_deg(d.qpos[3:7])))
        min_pelvis = min(min_pelvis, float(d.qpos[2]))
        if playing:
            target = reference["joint_pos"][min(frame + 1, length - 1)]
            tracking.append(float(np.mean(np.abs(q_hw[robot.hw_to_isaaclab] - target))))
        for key, value in (
            ("t", step * CONTROL_DT),
            ("playing", playing),
            ("frame", frame),
            ("action", action),
            ("q", q_hw.copy()),
            ("peak_torque", step_torque),
            ("tilt_deg", g1_limits.tilt_deg(d.qpos[3:7])),
            ("pelvis_pos", d.qpos[0:3].copy()),
            ("lean_deg", float(g1_limits.forward_lean_deg(d.qpos[3:7]))),
            ("ref_lean_deg", float(g1_limits.forward_lean_deg(reference["pelvis_quat"][frame if playing else 0]))),
        ):
            trace[key].append(value)
        if fell_at is None and d.qpos[2] < FALL_HEIGHT:
            fell_at = step * CONTROL_DT
        if writer:
            renderer.update_scene(d, camera=camera)
            writer.stdin.write(renderer.render().tobytes())
    if writer:
        writer.stdin.close()
        writer.wait()
        renderer.close()

    torque_ratio = peak_torque / table.tau_gate
    speed_ratio = peak_speed / g1_limits.velocity_limit(table, thresholds)
    saturated = g1_limits.saturation(np.asarray(trace["peak_torque"]), robot.effort, thresholds)
    summary = {
        "policy": str(args.policy),
        "motion": folder.name,
        "torque_limits": args.torque_limits,
        "delay_ms": args.delay_ms,
        "revision": args.revision,
        "kp_scale": args.kp_scale,
        "kd_scale": args.kd_scale,
        "seconds": round(len(schedule) * CONTROL_DT, 2),
        "fell_at_s": fell_at,
        "max_tilt_deg": round(max_tilt, 1),
        "min_pelvis_z_m": round(min_pelvis, 3),
        "peak_torque_over_gate": {
            "value": round(float(torque_ratio.max()), 3),
            "joint": table.names[int(torque_ratio.argmax())],
        },
        "torque_saturation": {**saturated, "joint": table.names[saturated["joint"]] if saturated["joint"] >= 0 else ""},
        "peak_speed_over_gate": {
            "value": round(float(speed_ratio.max()), 3),
            "joint": table.names[int(speed_ratio.argmax())],
        },
        "min_joint_margin_rad": {
            "value": round(float(min_margin.min()), 3),
            "joint": table.names[int(min_margin.argmin())],
        },
        "tracking_mean_abs_joint_err_rad": round(float(np.mean(tracking)), 4) if tracking else None,
    }
    ref_max_tilt = float(g1_limits.tilt_deg(reference["pelvis_quat"]).max())
    summary["ref_max_tilt_deg"] = round(ref_max_tilt, 1)
    summary["fail_reasons"] = g1_limits.gate_verdict(
        fell_at is not None,
        max_tilt,
        ref_max_tilt,
        float(speed_ratio.max()),
        saturated,
        float(min_margin.min()),
        thresholds,
    )
    summary["quick_check_pass"] = not summary["fail_reasons"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **{k: np.asarray(v) for k, v in trace.items()})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
