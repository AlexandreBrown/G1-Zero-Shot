"""Where the grab poses' hands really meet, from a run.py --measure grab run. For every logged hold window it takes
the held joint angles (the simulation's trajectory.csv, or on the robot the CSV of record_lowstate.py), puts the robot
model's pelvis at the origin, and finds the balloon centres (at the hands' height) where both hands are within 3 cm
of the balloon's surface: the test the simulation uses to call a balloon held. From those cradle regions it fits each
grab reach's meeting distance and the waist-yaw tracking; --write stores and dates the grab section.

Run with an interpreter that has MuJoCo (the upstream sim venv):
  ../GR00T-WholeBodyControl/.venv_sim/bin/python -m gpt_astra.tools.grab_report <run folder> [--joints joints.csv]
      [--calibration gpt_astra/calibration/robot.json] [--write]
"""
import argparse
import csv
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpt_astra import calibration
from gpt_astra.follow import coverage_gaps, grab_variants

CRADLE = .03
SCENE = "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
GRID_X, GRID_Y = np.arange(.25, .66, .01), np.arange(-.3, .31, .01)


def hold_windows(run: Path) -> list[dict]:
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    return [e for e in events if e["event"] == "grab_hold"]


def joints_from_trajectory(run: Path, start: float, end: float) -> dict[str, float]:
    """Median joint angles (by joint name) of the simulation trajectory over the wall-time window."""
    meta = json.loads((run / "trajectory_meta.json").read_text())
    rows = np.array([list(map(float, r)) for r in list(csv.reader((run / "trajectory.csv").open()))[1:]])
    t0 = meta["started_at_unix"]
    window = rows[(rows[:, 0] >= start - t0) & (rows[:, 0] <= end - t0)]
    q = np.median(window[:, 2:2 + meta["nq"]], axis=0)
    return {j["name"]: float(q[j["qpos_address"]]) for j in meta["joints"] if j["name"]}


def joints_from_lowstate(path: Path, names: list[str], start: float, end: float) -> dict[str, float]:
    """Median body joint angles of record_lowstate.py's CSV (motor order) over the wall-time window."""
    rows = np.array([list(map(float, r)) for r in list(csv.reader(path.open()))[1:]])
    window = rows[(rows[:, 0] >= start) & (rows[:, 0] <= end)]
    if not len(window):
        raise ValueError(f"No joint samples between {start:.1f} and {end:.1f} in {path}")
    return dict(zip(names, np.median(window[:, 1:30], axis=0)))


class Cradle:
    """The robot model with a balloon, pelvis at the origin: which balloon centres both hands hold."""
    def __init__(self, upstream: Path, radius: float):
        import mujoco
        from gpt_astra.tasks.catch_balloon.sim import prepare_scene
        self.mujoco = mujoco
        with tempfile.TemporaryDirectory() as tmp:
            self.model = mujoco.MjModel.from_xml_path(str(prepare_scene(upstream / SCENE, Path(tmp))))
        self.data = mujoco.MjData(self.model)
        self.ball = self.model.geom("target_balloon_geom").id
        self.model.geom_size[self.ball][0] = radius
        joint = self.model.joint("target_balloon_joint").id
        self.ball_q = int(self.model.jnt_qposadr[joint])
        self.base_q = next(int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)
                           if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and j != joint)
        body_name = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g]))
        self.hands = {side: [g for g in range(self.model.ngeom) if body_name(g).startswith((side + "_hand",
                                                                                          side + "_wrist"))]
                      for side in ("left", "right")}

    def region(self, joints: dict[str, float]) -> tuple[np.ndarray, float]:
        """Balloon centres (x, y) both hands cradle, and the height they were searched at (the hands' height)."""
        m, d, mujoco = self.model, self.data, self.mujoco
        d.qpos[:] = m.qpos0
        d.qpos[self.base_q:self.base_q + 7] = [0, 0, 0, 1, 0, 0, 0]
        for name, value in joints.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j >= 0 and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                d.qpos[m.jnt_qposadr[j]] = value
        d.qpos[self.ball_q:self.ball_q + 7] = [5, 5, 5, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        z = float(np.mean([d.geom_xpos[g][2] for geoms in self.hands.values() for g in geoms]))
        found = []
        for x in GRID_X:
            for y in GRID_Y:
                d.qpos[self.ball_q:self.ball_q + 3] = [x, y, z]
                mujoco.mj_kinematics(m, d)
                if all(min(mujoco.mj_geomDistance(m, d, self.ball, g, CRADLE, None) for g in geoms) < CRADLE
                       for geoms in self.hands.values()):
                    found.append((x, y))
        return np.array(found).reshape(-1, 2), z


def fit(regions: list[tuple[str, np.ndarray]], cal: calibration.Calibration) -> dict:
    """Meeting distance per reach (the centres' distance from the pelvis axis) and the yaw tracking (centre bearing
    over waist yaw, twisted poses only)."""
    variants = {v[0]: v for v in grab_variants(cal)}
    meeting, tracking = dict(), []
    for suffix, region in regions:
        if not len(region):
            continue
        cx, cy = region.mean(axis=0)
        meeting.setdefault(int(suffix.split("_")[0]), []).append(float(np.hypot(cx, cy)))
        yaw = variants[suffix][1]
        if abs(yaw) >= .15:
            tracking.append(float(np.arctan2(cy, cx) / yaw))
    meeting_x = list(cal.meeting_x)
    for reach, values in meeting.items():
        meeting_x[reach] = round(float(np.median(values)), 3)
    return {"meeting_x_m": meeting_x,
            "yaw_tracking": round(float(np.median(tracking)), 3) if tracking else cal.yaw_tracking}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--joints", type=Path, help="record_lowstate.py CSV (robot); default: the sim trajectory")
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "GR00T-WholeBodyControl")
    p.add_argument("--calibration", type=Path, default=calibration.for_backend("robot"))
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    from sonic_laptop.model import load_model
    cal = calibration.load(a.calibration)
    if a.write and (problem := calibration.platform_problem(a.run, cal)):
        print("FAIL: " + problem)
        return 1
    names = load_model(a.upstream)["names"]
    cradle = Cradle(a.upstream, cal.balloon_radius)
    variants = {v[0]: v for v in grab_variants(cal)}
    regions, failed = [], []
    for window in hold_windows(a.run):
        start, end = window["start"] + .5, window["end"]
        joints = (joints_from_lowstate(a.joints, names, start, end) if a.joints
                  else joints_from_trajectory(a.run, start, end))
        region, z = cradle.region(joints)
        suffix = window["grab"]
        regions.append((suffix, region))
        model = variants[suffix]
        if not len(region):
            failed.append(suffix)
            print(f"grab_{suffix}: NO cradle region (hands too far apart?)  model meet ({model[4]:.3f}, {model[5]:+.3f})")
            continue
        width = np.ptp(region[np.abs(region[:, 0] - region[:, 0].mean()) < .006][:, 1]) + .01
        print(f"grab_{suffix}: centre ({region[:, 0].mean():.3f}, {region[:, 1].mean():+.3f}) z {z:.3f}  "
              f"x {region[:, 0].min():.2f}..{region[:, 0].max():.2f}  sideways width {width * 100:.0f} cm  "
              f"model meet ({model[4]:.3f}, {model[5]:+.3f})")
    if not regions:
        print("No grab_hold windows in this run")
        return 1
    result = fit(regions, cal)
    print(json.dumps(result))
    if failed:
        print(f"FAIL: {len(failed)} pose(s) cannot hold the balloon at all: {failed}")
    gaps = coverage_gaps(replace(cal, meeting_x=tuple(result["meeting_x_m"]), yaw_tracking=result["yaw_tracking"]))
    if gaps:
        failed.append("coverage")
        print(f"FAIL: with these meeting points no grab pose reaches {len(gaps)} spots of the approach zone, e.g. "
              f"{gaps[:3]}; narrow grab.zone_tolerance_m and rerun")
    if a.write and not failed:
        data = json.loads(a.calibration.read_text())
        data["grab"].update(result, measured=f"{date.today().isoformat()} grab_report: {len(regions)} held poses "
                                             f"({'robot joints' if a.joints else 'sim trajectory'})")
        a.calibration.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote the grab section of {a.calibration}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
