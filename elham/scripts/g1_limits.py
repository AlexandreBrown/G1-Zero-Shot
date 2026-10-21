"""The G1 joint table (configs/g1_joint_limits.yaml) and the thresholds the gate, monitor and analyzer share."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
NUM_JOINTS = 29


@dataclass(frozen=True)
class JointTable:
    names: tuple[str, ...]
    pos_lo: np.ndarray
    pos_hi: np.ndarray
    default: np.ndarray
    tau_gate: np.ndarray
    tau_train: np.ndarray
    vel_gate: np.ndarray


@dataclass(frozen=True)
class Thresholds:
    """Abort / gate thresholds (README 'Stops', docs/CHECKLIST.md aborts, and the preview gate).

    The saturation limits are screening values: in MuJoCo, NVIDIA's released base reaches its torque clip for at
    most 0.08 s and 1.1 % of the control steps on the 18 retained motions; brief saturation is normal, long or
    frequent saturation means the policy needs torque the motor cannot give.
    """

    torque_fraction: float = 0.9
    saturation_fraction: float = 0.98
    saturation_longest_s: float = 0.1
    saturation_share: float = 0.02
    velocity_fraction: float = 0.8
    velocity_abs: float = 28.0
    position_margin: float = 0.05
    position_overshoot: float = 0.02
    tilt_excess_deg: float = 15.0
    tilt_abs_deg: float = 55.0
    temp_warn: float = 70.0
    temp_stop: float = 85.0
    lowstate_min_hz: float = 400.0
    lowstate_max_gap_s: float = 0.05


def load_joint_table(path: Path = REPO / "configs" / "g1_joint_limits.yaml") -> JointTable:
    rows = yaml.safe_load(path.read_text())["joints"]
    if [r["index"] for r in rows] != list(range(NUM_JOINTS)):
        raise ValueError(f"{path}: expected joints 0..{NUM_JOINTS - 1} in hardware order")

    def column(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows])

    return JointTable(
        names=tuple(r["name"] for r in rows),
        pos_lo=column("pos_lo"),
        pos_hi=column("pos_hi"),
        default=column("default"),
        tau_gate=column("tau_gate"),
        tau_train=column("tau_train"),
        vel_gate=column("vel_gate"),
    )


def tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and world z, from (..., 4) quaternions (w, x, y, z)."""
    x, y = quat_wxyz[..., 1], quat_wxyz[..., 2]
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


def forward_lean_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Pitch of the body x axis below the horizon in degrees, from (..., 4) quaternions (w, x, y, z); positive =
    leaning forward (nose down). For the pelvis quaternion this is the pitch the pelvis IMU reads."""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    forward_z = 2.0 * (x * z - w * y)
    return np.degrees(np.arcsin(np.clip(-forward_z, -1.0, 1.0)))


def deploy_sim_effort(table: JointTable) -> np.ndarray:
    """Torque limits of SONIC's deploy MuJoCo simulator: training limits with hip pitch and roll at 88 Nm
    (gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml)."""
    effort = table.tau_train.copy()
    for i, name in enumerate(table.names):
        if "hip_pitch" in name or "hip_roll" in name:
            effort[i] = 88.0
    return effort


def position_margin(q: np.ndarray, table: JointTable) -> np.ndarray:
    return np.minimum(q - table.pos_lo, table.pos_hi - q)


def velocity_limit(table: JointTable, thresholds: Thresholds) -> np.ndarray:
    return np.minimum(thresholds.velocity_fraction * table.vel_gate, thresholds.velocity_abs)


def saturation(torque_abs: np.ndarray, limit: np.ndarray, thresholds: Thresholds, dt: float = 0.02) -> dict:
    """Longest run (s) and share of control steps with any joint at >= saturation_fraction x its limit."""
    saturated = torque_abs >= thresholds.saturation_fraction * limit
    longest, joint = 0, -1
    for j in range(saturated.shape[1]):
        run = best = 0
        for value in saturated[:, j]:
            run = run + 1 if value else 0
            best = max(best, run)
        if best > longest:
            longest, joint = best, j
    return {"longest_s": round(longest * dt, 3), "share": round(float(saturated.any(axis=1).mean()), 4), "joint": joint}


def gate_verdict(
    fell: bool,
    max_tilt: float,
    ref_max_tilt: float | None,
    speed_over_gate: float,
    saturation_stats: dict,
    min_margin: float,
    thresholds: Thresholds,
) -> list[str]:
    """Reasons a trial fails the gate (empty list = pass).

    Tilt is judged against the motion's own pelvis tilt (kicks lean ~40 deg by design): the robot may not tilt more
    than tilt_excess_deg beyond the reference's maximum, and never more than tilt_abs_deg.
    """
    reasons = []
    if fell:
        reasons.append("fell")
    if max_tilt >= thresholds.tilt_abs_deg:
        reasons.append(f"tilt {max_tilt:.0f} deg")
    elif ref_max_tilt is not None and max_tilt > ref_max_tilt + thresholds.tilt_excess_deg:
        reasons.append(f"tilt {max_tilt:.0f} deg vs the reference's {ref_max_tilt:.0f} deg")
    if speed_over_gate > 1.0:
        reasons.append(f"joint speed {speed_over_gate:.2f} x gate")
    if saturation_stats["longest_s"] > thresholds.saturation_longest_s:
        reasons.append(f"torque saturated {saturation_stats['longest_s']:.2f} s in a row")
    if saturation_stats["share"] > thresholds.saturation_share:
        reasons.append(f"torque saturated on {saturation_stats['share']:.1%} of steps")
    if min_margin < -thresholds.position_overshoot:
        reasons.append(f"joint {-min_margin:.3f} rad past its limit")
    return reasons
