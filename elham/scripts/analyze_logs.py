"""Summarise deploy-program trial logs (sim2sim or real G1) against the G1 joint limits, and compare trials.

A trial folder is what scripts/sonic_run.sh writes: <trial>/state/*.csv (the program's --enable-csv-logs output,
hardware joint order, 50 Hz, CONTROL state only) and <trial>/target.csv (--target-motion-logfile: pelvis
position, pelvis quaternion wxyz, 29 reference joint positions in hardware order, one row per control tick).

Per trial: peak |torque| per joint over the conservative limit (configs/g1_joint_limits.yaml tau_gate), time near
the limit, peak |joint speed|, smallest distance to a joint limit, largest base tilt, hottest motor, and the joint
tracking error against the reference while the motion plays, and the pelvis forward lean while the policy holds
frame 0 before the first play (after SETTLE_S; compare with the simulated value in docs/MOTIONS.md). The gate passes
when tilt <= the reference's own maximum + 15 deg (and < 55 deg), |dq| <= min(0.8 x limit, 28 rad/s), no joint is at
>= 98 % of its torque limit for more than 0.1 s in a row or on more than 2 % of the steps, and no joint goes more
than 0.02 rad past its range (scripts/g1_limits.py gate_verdict). A fall shows up as tilt; the log has no base
height.

    python scripts/analyze_logs.py logs/2026-10-22_sim/base_squat_1 [more trials ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deploy_logs as dl  # noqa: E402
import g1_limits  # noqa: E402

SETTLE_S = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write all summaries here")
    return parser.parse_args()


def summarise(trial: Path, table: g1_limits.JointTable, thresholds: g1_limits.Thresholds) -> dict:
    state = trial / "state"
    base = dl.read_logged(state / "q.csv")
    raw = {
        name: dl.align(base.index, dl.read_logged(state / f"{name}.csv"))
        for name in ("dq", "motor_torque", "motor_temperature", "base_quat", "motion_playing")
    }
    n_rows = len(base.index)
    q = base.data[:, : g1_limits.NUM_JOINTS]
    dq = raw["dq"][:, : g1_limits.NUM_JOINTS]
    tau = raw["motor_torque"][:, : g1_limits.NUM_JOINTS]
    temperature = raw["motor_temperature"].reshape(n_rows, g1_limits.NUM_JOINTS, -1).max(axis=2)
    base_quat = raw["base_quat"]
    playing = raw["motion_playing"][:, 0] > 0.5

    torque_ratio = np.abs(tau) / table.tau_gate
    speed_ratio = np.abs(dq) / g1_limits.velocity_limit(table, thresholds)
    margin = g1_limits.position_margin(q, table)
    tilt = g1_limits.tilt_deg(base_quat)
    first_play = int(np.argmax(playing)) if playing.any() else n_rows
    held_lean = g1_limits.forward_lean_deg(base_quat[round(SETTLE_S / 0.02) : first_play])

    def worst(values: np.ndarray, per_joint=np.nanmax, pick=np.nanargmax) -> dict:
        """Per-joint reduction over time (NaN = a stream missed that tick), then the worst joint."""
        reduced = per_joint(values, axis=0)
        index = int(pick(reduced))
        return {"value": round(float(reduced[index]), 3), "joint": table.names[index]}

    summary = {
        "trial": str(trial),
        "rows": int(n_rows),
        "duration_s": round(n_rows * 0.02, 2),
        "playing_s": round(float(playing.sum()) * 0.02, 2),
        "peak_torque_over_limit": worst(torque_ratio),
        "peak_torque_over_training_limit": worst(np.abs(tau) / table.tau_train),
        "share_steps_above_0.9_torque_limit": round(
            float((torque_ratio > thresholds.torque_fraction).any(axis=1).mean()), 4
        ),
        "peak_speed_over_gate": worst(speed_ratio),
        "min_joint_margin_rad": worst(margin, per_joint=np.nanmin, pick=np.nanargmin),
        "max_tilt_deg": round(float(np.nanmax(tilt)), 1),
        "max_motor_temp_c": worst(temperature),
    }
    if len(held_lean):
        summary["held_lean_deg"] = round(float(np.nanmean(held_lean)), 1)
    target_path = trial / "target.csv"
    ref_max_tilt = None
    if target_path.exists() and target_path.stat().st_size > 0:
        target_rows = dl.read_unindexed(target_path).reshape(-1, 36)
        ref_max_tilt = float(g1_limits.tilt_deg(target_rows[:, 3:7]).max())
        summary["ref_max_tilt_deg"] = round(ref_max_tilt, 1)
        summary["ref_frame0_lean_deg"] = round(float(g1_limits.forward_lean_deg(target_rows[0, 3:7])), 1)
        target = target_rows[:, 7:]
        n = min(len(target), n_rows)
        summary["target_rows_minus_state_rows"] = int(len(target) - n_rows)
        mask = playing[:n]
        if mask.any():
            error = np.abs(q[:n][mask] - target[:n][mask])
            summary["tracking_mean_abs_joint_err_rad"] = round(float(error.mean()), 4)
            summary["tracking_worst_joint"] = worst(error, per_joint=np.nanmean)
    # In the deploy simulator the motors clip at the simulator's own limits (hip pitch AND roll 88 Nm), so a sim
    # trial is checked for saturation against those; a real trial against tau_gate.
    manifest = trial / "run_manifest.txt"
    is_sim = manifest.exists() and "mode: sim" in manifest.read_text()
    limit = g1_limits.deploy_sim_effort(table) if is_sim else table.tau_gate
    saturated = g1_limits.saturation(np.nan_to_num(np.abs(tau)), limit, thresholds)
    summary["torque_saturation"] = {
        **saturated,
        "joint": table.names[saturated["joint"]] if saturated["joint"] >= 0 else "",
    }
    summary["fail_reasons"] = g1_limits.gate_verdict(
        False,
        summary["max_tilt_deg"],
        ref_max_tilt,
        summary["peak_speed_over_gate"]["value"],
        saturated,
        summary["min_joint_margin_rad"]["value"],
        thresholds,
    )
    summary["gate_pass"] = not summary["fail_reasons"]
    return summary


def print_table(summaries: list[dict]) -> None:
    keys = [
        ("duration_s", "dur s"),
        ("playing_s", "play s"),
        ("peak_torque_over_limit", "tau/lim"),
        ("share_steps_above_0.9_torque_limit", ">0.9 lim"),
        ("peak_speed_over_gate", "dq/gate"),
        ("min_joint_margin_rad", "margin"),
        ("max_tilt_deg", "tilt deg"),
        ("held_lean_deg", "lean held"),
        ("ref_frame0_lean_deg", "lean ref0"),
        ("max_motor_temp_c", "temp C"),
        ("tracking_mean_abs_joint_err_rad", "track err"),
        ("torque_saturation", "saturated"),
        ("fail_reasons", "fails"),
        ("gate_pass", "gate"),
    ]
    for summary in summaries:
        print(f"\n{summary['trial']}")
        for key, label in keys:
            value = summary.get(key, "-")
            if isinstance(value, dict) and "longest_s" in value:
                value = f"{value['longest_s']} s longest, {value['share']:.1%} of steps ({value['joint']})"
            elif isinstance(value, dict):
                value = f"{value['value']} ({value['joint']})"
            print(f"  {label:10s} {value}")


def main() -> None:
    args = parse_args()
    table = g1_limits.load_joint_table()
    thresholds = g1_limits.Thresholds()
    summaries = []
    for trial in args.trials:
        if dl.is_trial(trial):
            summaries.append(summarise(trial, table, thresholds))
        else:
            print(f"{trial}: skipped (no logged control ticks: not a trial folder, or stopped before ])")
    if not summaries:
        sys.exit("no trial folders with logged control ticks")
    print_table(summaries)
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")
    sys.exit(0 if all(s["gate_pass"] for s in summaries) else 1)


if __name__ == "__main__":
    main()
