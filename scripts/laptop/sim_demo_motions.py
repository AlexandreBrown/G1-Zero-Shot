#!/usr/bin/env python3
"""LOCAL: generate the Kimodo-G1 CSVs (30 FPS) in motions/sim_demo/ used by demo.sh in MuJoCo.

The idle is SONIC's nominal stance with the arms opened to an A-pose (shoulder roll 0.7 rad),
with margin over the panel's 25 mm hand-envelope check. Every clip starts and ends
on that idle. Root heights come from the simulator's own foot boxes so both soles touch the floor.
These clips are simulation demo material, not validated for hardware.
Run with any Python that has numpy + mujoco: python scripts/laptop/sim_demo_motions.py
"""
import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sonic_laptop.import_kimodo import G1_JOINTS  # noqa: E402
from sonic_laptop.model import load_model  # noqa: E402

FPS = 30
SHOULDER_ROLL = 0.7  # rad; wide enough to keep the 25 mm hand-envelope check clear of setup sag and jolts
J = {name: i for i, name in enumerate(G1_JOINTS)}


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class FootModel:
    """Forward kinematics of the sim scene, used to keep both soles on the floor."""

    def __init__(self, upstream: Path):
        self.model = mujoco.MjModel.from_xml_path(
            str(upstream / 'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'))
        self.data = mujoco.MjData(self.model)
        self.address = [self.model.jnt_qposadr[self.model.joint(name).id] for name in G1_JOINTS]
        self.feet = [g for g in range(self.model.ngeom)
                     if 'ankle_roll' in self.model.body(self.model.geom_bodyid[g]).name
                     and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]

    def sole(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Left-foot box centre and lowest sole corner with the pelvis at the origin."""
        self.data.qpos[:] = 0
        self.data.qpos[3] = 1
        self.data.qpos[self.address] = q
        mujoco.mj_kinematics(self.model, self.data)
        lowest = np.inf
        for g in self.feet:
            half = self.model.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
            world = self.data.geom_xpos[g] + corners @ self.data.geom_xmat[g].reshape(3, 3).T
            lowest = min(lowest, world[:, 2].min())
        return self.data.geom_xpos[self.feet[0]].copy(), lowest


def idle_pose(model: dict) -> np.ndarray:
    q = np.array(model['default'], dtype=float)
    q[J['left_shoulder_roll_joint']] = SHOULDER_ROLL
    q[J['right_shoulder_roll_joint']] = -SHOULDER_ROLL
    return q


def segments(idle: np.ndarray, keyframes: list[tuple[float, dict]], dwell: float = 1.0) -> np.ndarray:
    """Min-jerk moves between keyframes {joint: angle}; idle dwell before and after."""
    poses, durations = [idle], [dwell]
    for duration, targets in keyframes:
        pose = poses[-1].copy()
        for name, value in targets.items():
            pose[J[name]] = value
        poses.append(pose)
        durations.append(duration)
    frames = [np.repeat(idle[None], int(dwell * FPS), axis=0)]
    for start, end, duration in zip(poses[:-1], poses[1:], durations[1:]):
        s = min_jerk(np.arange(1, int(round(duration * FPS)) + 1) / (duration * FPS))
        frames.append(start + s[:, None] * (end - start))
    frames.append(np.repeat(poses[-1][None], int(dwell * FPS), axis=0))
    return np.concatenate(frames)


def to_kimodo(joints: np.ndarray, feet: FootModel) -> np.ndarray:
    centre0, lowest0 = feet.sole(joints[0])
    rows = []
    for q in joints:
        centre, _ = feet.sole(q)
        root = np.array([0.0, 0.0, -lowest0]) + (centre0 - centre)
        rows.append(np.r_[root, 1, 0, 0, 0, q])
    return np.array(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=ROOT.parent / 'GR00T-WholeBodyControl')
    parser.add_argument('--output', type=Path, default=ROOT / 'motions/sim_demo')
    args = parser.parse_args()
    model = load_model(args.upstream)
    assert tuple(model['names']) == G1_JOINTS
    feet = FootModel(args.upstream)
    idle = idle_pose(model)
    arm_up = {'left_shoulder_pitch_joint': -1.2, 'right_shoulder_pitch_joint': -1.2,
              'left_elbow_joint': 0.3, 'right_elbow_joint': 0.3}
    arm_down = {'left_shoulder_pitch_joint': idle[J['left_shoulder_pitch_joint']],
                'right_shoulder_pitch_joint': idle[J['right_shoulder_pitch_joint']],
                'left_elbow_joint': idle[J['left_elbow_joint']], 'right_elbow_joint': idle[J['right_elbow_joint']]}
    wave_up = {'right_shoulder_roll_joint': -1.3, 'right_shoulder_pitch_joint': -0.3, 'right_elbow_joint': 1.3}
    wave = [(0.5, {'right_shoulder_yaw_joint': 0.35}), (0.5, {'right_shoulder_yaw_joint': -0.35})] * 3
    wave_down = {name: idle[J[name]] for name in
                 ('right_shoulder_roll_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_shoulder_yaw_joint')}
    depth = 0.35
    squat_down = {f'{side}_{joint}': idle[J[f'{side}_{joint}']] + delta for side in ('left', 'right')
                  for joint, delta in (('hip_pitch_joint', -depth), ('knee_joint', 2 * depth), ('ankle_pitch_joint', -depth))}
    squat_down.update(arm_up)
    squat_up = {name: idle[J[name]] for name in squat_down}
    clips = {
        'idle_apose': (args.output / 'idle', segments(idle, [], dwell=2.0)),
        'arms_forward': (args.output / 'clips', segments(idle, [(1.6, arm_up), (0.6, {}), (1.6, arm_down)])),
        'wave_right': (args.output / 'clips', segments(idle, [(1.2, wave_up), *wave, (1.2, wave_down)])),
        'squat': (args.output / 'clips', segments(idle, [(1.6, squat_down), (0.6, {}), (1.6, squat_up)])),
    }
    for name, (folder, joints) in clips.items():
        folder.mkdir(parents=True, exist_ok=True)
        rows = to_kimodo(joints, feet)
        np.savetxt(folder / f'{name}.csv', rows, delimiter=',', fmt='%.9f')
        print(f'{name}: {len(rows)} frames ({(len(rows) - 1) / FPS:.2f} s), '
              f'pelvis z {rows[:, 2].min():.3f}-{rows[:, 2].max():.3f} m, x {rows[:, 0].min():+.3f}-{rows[:, 0].max():+.3f} m')


if __name__ == '__main__':
    main()
