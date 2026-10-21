"""Build the deploy reference-motion folders for each stage in configs/motion_stages.yaml.

For every motion it writes  <out>/single/<motion>/<motion_key>/{joint_pos,joint_vel,body_*}.csv  (one motion per
folder: the deploy program loads every subfolder of --motion-data in an unspecified order and does not blend
between motions, so single-motion folders are the safe default), plus  <out>/<stage>/  with one symlink per
motion of that stage, and  <out>/motion_report.csv  with the numbers to check before a hardware session.

ASAP motions go through SONIC's own exporter (gear_sonic/data_process/export_motion_to_deploy.py: MotionLibRobot
FK, 30 -> 50 Hz, MuJoCo -> IsaacLab joint order, xyzw -> wxyz). Its Isaac import is blocked so the exporter uses
its source-parsing fallback; with Isaac Sim installed that import otherwise hangs on the Omniverse licence prompt.

    python scripts/export_motions.py --sonic-root ~/GR00T-WholeBodyControl \
        --asap-dir /path/asap_to_sonic/robot --out artifacts/motions
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g1_limits  # noqa: E402
import sonic_tree  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONTROL_HZ = 50.0
CSV_FILES = ("joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv", "body_lin_vel.csv", "body_ang_vel.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sonic-root", type=Path, required=True, help="GR00T-WholeBodyControl checkout")
    parser.add_argument("--asap-dir", type=Path, required=True, help="folder with asap_<name>.pkl (SONIC format)")
    parser.add_argument("--out", type=Path, required=True, help="output folder (e.g. artifacts/motions)")
    parser.add_argument("--stages", type=Path, default=REPO / "configs" / "motion_stages.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="limit to these stage names")
    return parser.parse_args()


def load_joint_table(path: Path = REPO / "configs" / "g1_joint_limits.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text())["joints"]


def read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def isaaclab_to_hardware_index() -> list[int]:
    """hw[i] = isaaclab[ISAACLAB_TO_MUJOCO[i]] (the deploy's isaaclab_to_mujoco table, policy_parameters.hpp)."""
    return [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]


def motion_stats(folder: Path, joints: list[dict]) -> dict:
    """Numbers a hardware operator needs for one exported motion folder."""
    arrays = {name: read_csv(folder / name) for name in CSV_FILES}
    frames = {name: len(a) for name, a in arrays.items()}
    to_hw = isaaclab_to_hardware_index()
    q = arrays["joint_pos.csv"][:, to_hw]
    dq = arrays["joint_vel.csv"][:, to_hw]
    default = np.array([j["default"] for j in joints])
    lo = np.array([j["pos_lo"] for j in joints])
    hi = np.array([j["pos_hi"] for j in joints])
    vel_lim = np.array([j["vel_gate"] for j in joints])
    pelvis = arrays["body_pos.csv"][:, 0:3]
    pelvis_tilt = g1_limits.tilt_deg(arrays["body_quat.csv"][:, 0:4])
    travel = np.linalg.norm(pelvis[:, :2] - pelvis[0, :2], axis=1)
    margin = np.minimum(q - lo, hi - q)
    first_dev = np.abs(q[0] - default)
    last_dev = np.abs(q[-1] - default)
    return {
        "frames": frames["joint_pos.csv"],
        "frames_consistent": len(set(frames.values())) == 1,
        "duration_s": round(frames["joint_pos.csv"] / CONTROL_HZ, 2),
        "max_travel_m": round(float(travel.max()), 2),
        "pelvis_z_min_m": round(float(pelvis[:, 2].min()), 2),
        "max_ref_pelvis_tilt_deg": round(float(pelvis_tilt.max()), 1),
        "frame0_max_dev_from_stand_rad": round(float(first_dev.max()), 2),
        "frame0_worst_joint": joints[int(first_dev.argmax())]["name"],
        "last_max_dev_from_stand_rad": round(float(last_dev.max()), 2),
        "max_joint_speed_rad_s": round(float(np.abs(dq).max()), 2),
        "max_speed_over_limit": round(float((np.abs(dq) / vel_lim).max()), 2),
        "min_pos_margin_rad": round(float(margin.min()), 3),
        "min_margin_joint": joints[int(np.unravel_index(margin.argmin(), margin.shape)[1])]["name"],
    }


def export_asap_motion(pkl: Path, out_dir: Path, sonic_root: Path) -> Path:
    """Export one motion_lib PKL with SONIC's exporter; returns the folder it created."""
    sys.modules["gear_sonic.envs.manager_env.robots"] = None
    from gear_sonic.data_process import export_motion_to_deploy as exporter

    sys.path.insert(0, str(sonic_root / "gear_sonic_deploy" / "reference"))
    import convert_motions

    exported = exporter.export_motions(
        motion_file=str(pkl),
        target_fps=int(CONTROL_HZ),
        body_indexes=None,
        device="cpu",
        zero_root_xy=False,
        asset_root=exporter.DEFAULT_ASSET_ROOT,
        asset_file=exporter.DEFAULT_ASSET_FILE,
    )
    if len(exported) != 1:
        raise RuntimeError(f"{pkl} holds {len(exported)} motions; expected exactly one")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "_deploy_export.pkl"
    import joblib

    joblib.dump(exported, pkl_path)
    convert_motions.convert_motion_data(str(pkl_path), str(out_dir))
    pkl_path.unlink()
    (key,) = exported.keys()
    return out_dir / key


def clear_stage(stage_dir: Path) -> None:
    """Remove the stage's motion links, so a motion dropped from the stage in the YAML is not loaded any more."""
    if stage_dir.is_dir():
        for link in stage_dir.iterdir():
            if link.is_symlink():
                link.unlink()


def link_stage(stage_dir: Path, motion_folder: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    link = stage_dir / motion_folder.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(motion_folder, stage_dir))


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    asap_dir = args.asap_dir.resolve()
    sonic_root = args.sonic_root.resolve()
    stages = yaml.safe_load(args.stages.read_text())["stages"]
    joints = load_joint_table()
    sonic_tree.enter_sonic_tree(sonic_root)

    rows = []
    for stage_name, stage in stages.items():
        if args.only and stage_name not in args.only:
            continue
        clear_stage(out / stage_name)
        for name in stage.get("upstream", []):
            source = sonic_root / "gear_sonic_deploy" / "reference" / "example" / name
            single = out / "single" / name
            if single.exists():
                shutil.rmtree(single)
            shutil.copytree(source, single / name)
            folder = single / name
            link_stage(out / stage_name, folder)
            rows.append({"stage": stage_name, "motion": name, "source": "upstream", **motion_stats(folder, joints)})
        for name in stage.get("asap", []):
            pkl = asap_dir / f"asap_{name}.pkl"
            single = out / "single" / name
            if single.exists():
                shutil.rmtree(single)
            folder = export_asap_motion(pkl, single, sonic_root)
            link_stage(out / stage_name, folder)
            rows.append({"stage": stage_name, "motion": name, "source": "asap", **motion_stats(folder, joints)})
            print(f"[{stage_name}] {name}: {rows[-1]['frames']} frames, travel {rows[-1]['max_travel_m']} m")

    report = out / "motion_report.csv"
    with open(report, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bad = [r["motion"] for r in rows if not r["frames_consistent"] or r["min_pos_margin_rad"] < 0]
    print(f"wrote {report} ({len(rows)} motions)")
    if bad:
        sys.exit(f"ERROR: inconsistent frame counts or joint limits exceeded in: {bad}")


if __name__ == "__main__":
    main()
