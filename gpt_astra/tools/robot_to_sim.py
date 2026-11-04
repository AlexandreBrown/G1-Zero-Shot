"""Turn a robot log from gpt_astra/tools/record_lowstate.py into a folder gpt_astra/replay.py plays in MuJoCo.

The G1 model takes the logged body joint angles and the IMU orientation (heading zeroed at the first sample); the hands
stay open. The log has no base position, so every frame stands the lower foot on the floor: a robot hanging in the
gantry replays as if standing, and walking replays in place.

Run with an interpreter that has MuJoCo (the upstream sim venv):
  ../GR00T-WholeBodyControl/.venv_sim/bin/python -m gpt_astra.tools.robot_to_sim <log.csv> [--out <folder>]
  ../GR00T-WholeBodyControl/.venv_sim/bin/python gpt_astra/replay.py <folder> [--speed 0.5]
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCENE = "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
JOINTS = 29


def euler_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def heading(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_log(path: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader(path.open()))
    return rows[0], np.array([[float(v) for v in row] for row in rows[1:]])


class Stander:
    """The G1 model with the base placed so the lower foot touches the floor."""
    def __init__(self, upstream: Path, names: list[str]):
        import mujoco
        self.mujoco = mujoco
        self.scene = (upstream / SCENE).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.base = next(j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE))
        self.base_q, self.base_v = int(m.jnt_qposadr[self.base]), int(m.jnt_dofadr[self.base])
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        if any(j < 0 for j in joints):
            raise ValueError(f"Joints missing from {self.scene}: {[n for n, j in zip(names, joints) if j < 0]}")
        self.joint_q = [int(m.jnt_qposadr[j]) for j in joints]
        self.joint_v = [int(m.jnt_dofadr[j]) for j in joints]
        self.floor = next(g for g in range(m.ngeom) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE))
        body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        self.feet = [g for g in range(m.ngeom) if "ankle_roll" in body(g)]

    def pose(self, joints: np.ndarray, quaternion: np.ndarray, speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, d, mujoco = self.model, self.data, self.mujoco
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        d.qpos[self.base_q:self.base_q + 3] = (0, 0, 1)
        d.qpos[self.base_q + 3:self.base_q + 7] = quaternion / np.linalg.norm(quaternion)
        d.qpos[self.joint_q] = joints
        d.qvel[self.joint_v] = speeds
        mujoco.mj_kinematics(m, d)
        gap = min(mujoco.mj_geomDistance(m, d, self.floor, g, 2., None) for g in self.feet)
        d.qpos[self.base_q + 2] -= gap
        return d.qpos.copy(), d.qvel.copy()

    def joints_meta(self) -> list[dict]:
        m, mujoco = self.model, self.mujoco
        return [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "",
                 "qpos_address": int(m.jnt_qposadr[j]), "qvel_address": int(m.jnt_dofadr[j])} for j in range(m.njnt)]


def convert(log: Path, out: Path, upstream: Path) -> int:
    """Write out/trajectory.csv and out/trajectory_meta.json; returns the number of frames."""
    from sonic_laptop.model import load_model
    header, rows = read_log(log)
    if not len(rows):
        raise ValueError(f"{log} has no samples")
    column = {name: i for i, name in enumerate(header)}
    stander = Stander(upstream, load_model(upstream)["names"][:JOINTS])
    joints = rows[:, column["q0"]:column["q0"] + JOINTS]
    speeds = rows[:, column["dq0"]:column["dq0"] + JOINTS] if "dq0" in column else np.zeros_like(joints)
    if "qw" in column:
        quaternions = rows[:, [column["qw"], column["qx"], column["qy"], column["qz"]]]
    else:  # logs from before the quaternion columns
        quaternions = np.array([euler_quaternion(*r) for r in rows[:, [column["roll"], column["pitch"], column["yaw"]]]])
    turn = -heading(quaternions[0])
    face_forward = np.array([math.cos(turn / 2), 0, 0, math.sin(turn / 2)])
    m = stander.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_s", "sim_time_s", *[f"qpos_{i}" for i in range(m.nq)],
                         *[f"qvel_{i}" for i in range(m.nv)], *[f"ctrl_{i}" for i in range(m.nu)]])
        for row, q, dq, quaternion in zip(rows, joints, speeds, quaternions):
            turned = np.empty(4)
            stander.mujoco.mju_mulQuat(turned, face_forward, quaternion)
            qpos, qvel = stander.pose(q, turned, dq)
            elapsed = row[0] - rows[0, 0]
            writer.writerow([f"{elapsed:.4f}", f"{elapsed:.4f}", *(f"{v:.6f}" for v in qpos),
                             *(f"{v:.6f}" for v in qvel), *(["0"] * m.nu)])
    (out / "trajectory_meta.json").write_text(json.dumps({
        "scene_path": str(stander.scene), "started_at_unix": float(rows[0, 0]), "nq": m.nq, "nv": m.nv, "nu": m.nu,
        "sample_hz": round(1 / np.median(np.diff(rows[:, 0]))) if len(rows) > 1 else 50,
        "joints": stander.joints_meta(), "video_fps": 0, "source": str(log.resolve())}, indent=2))
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help="record_lowstate.py CSV (not the _cmd.csv)")
    p.add_argument("--out", type=Path, help="default: <log name>_replay next to the log")
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "GR00T-WholeBodyControl")
    a = p.parse_args(argv)
    out = a.out or a.log.with_name(a.log.stem + "_replay")
    frames = convert(a.log, out, a.upstream)
    print(f"{frames} frames -> {out}\nPlay: ../GR00T-WholeBodyControl/.venv_sim/bin/python gpt_astra/replay.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
