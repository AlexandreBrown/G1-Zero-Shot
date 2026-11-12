import csv
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpt_astra import calibration
from gpt_astra.calibration import SIM
from gpt_astra.tools import check_camera, record_lowstate, walk_report


def jpeg(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def render(point, cal, width, height):
    """A red sphere at point (pelvis frame) seen by the calibration's camera, in a width x height image."""
    u, v, r = check_camera.project(np.array(point), cal.camera_position, cal.camera_rotation, cal.fovy_deg,
                                   cal.balloon_radius, width, height)
    rows, cols = np.mgrid[:height, :width]
    image = np.full((height, width, 3), (40, 60, 120), np.uint8)
    image[(cols - u) ** 2 + (rows - v) ** 2 <= r ** 2] = (200, 30, 25)
    return image


class CheckCameraTests(unittest.TestCase):
    def test_fit_recovers_a_real_camera_that_differs_from_the_model(self):
        # The real head camera is tilted 2 deg down, turned 1 deg, 1.5 cm off, has a 52 deg field of view and a
        # 16:9 image (640x360 after ZmqCamera's thumbnail); the balloon is 20 cm across.
        truth = replace(SIM, camera_position=SIM.camera_position + [.015, -.01, .01],
                        camera_rotation=SIM.camera_rotation @ check_camera.rotation(np.radians([2., 1., 0.])),
                        fovy_deg=52., balloon_radius=.1)
        frames = [jpeg(render(spot, truth, 640, 360)) for spot in check_camera.SPOTS]
        results = [check_camera.summarize([frame] * 2, SIM) for frame in frames]
        empty = np.full((360, 640, 3), (40, 60, 120), np.uint8)
        shirt = empty.copy()
        shirt[100:300, 250:340] = (200, 30, 25)
        result = check_camera.analyze(check_camera.SPOTS, results, SIM, .1, [jpeg(empty), jpeg(shirt)], frames)
        self.assertGreater(max(result["error_before_m"]), .05)       # the sim calibration is clearly off
        self.assertLess(max(result["error_after_m"]), .01)
        self.assertAlmostEqual(result["camera"]["fovy_deg"], 52., delta=1.)   # trades off with the pose
        self.assertTrue(np.allclose(result["camera"]["position_m"], truth.camera_position, atol=.01))
        self.assertEqual(result["empty_detections"], 0)
        self.assertLessEqual(result["balloon"]["red_min"], 80)
        self.assertEqual(result["spot_detection_rate"], 1.)
        self.assertGreaterEqual(result["balloon"]["red_ratio"], check_camera.RED_RATIO_FLOOR)
        self.assertLessEqual(result["balloon"]["max_fit_residual"], check_camera.RESIDUAL_CAP)



class WalkReportTests(unittest.TestCase):
    def test_coast_stalls_and_settle_back_from_the_probe_log(self):
        def probe(move, before, at_stop, after):
            return {"event": "walk_probe", "move": move, "seconds": 1.0, "before": before, "at_stop": at_stop,
                    "after": after, "aborted": None}
        events = []
        for _ in range(3):
            events += [probe("forward", [.7, 0, .16], [.55, 0, .16], [.45, 0, .16]),        # 0.25 m, coast 0.10
                       probe("back", [.45, 0, .16], [.6, 0, .16], [.62, 0, .16]),           # coast 0.02
                       probe("forward_fine", [.62, 0, .16], [.6, 0, .16], [.59, 0, .16]),   # 0.03 m
                       probe("back_fine", [.59, 0, .16], [.59, 0, .16], [.6, 0, .16]),      # stalled
                       probe("left", [.6, .0, .16], [.6, -.1, .16], [.6, -.15, .16]),       # coast 0.05
                       probe("right", [.6, -.15, .16], [.6, -.05, .16], [.6, 0., .16])]
        events.append({"event": "settle_probe", "before": [.5, 0, .16], "after": [.55, 0, .16]})
        events.insert(0, {"event": "prepared", "backend": "robot"})
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            cal = run / "robot.json"
            cal.write_text(calibration.for_backend("robot").read_text())
            self.assertEqual(walk_report.main([str(run), "--calibration", str(cal), "--write"]), 1)  # back_fine stalled
            moves = walk_report.summarize(walk_report.probes(run)[0])
            self.assertEqual(moves["forward"]["coast_m"], .1)
            self.assertEqual(moves["back"]["coast_m"], .02)
            self.assertEqual(moves["left"]["coast_m"], .05)
            self.assertEqual(moves["back_fine"]["stalled"], 3)
            self.assertIsNone(calibration.load(cal).measured["walk"])     # nothing written while a move stalls


if __name__ == "__main__":
    unittest.main()

UPSTREAM = ROOT.parent / "GR00T-WholeBodyControl"
LEGS = {0: -.3, 3: .6, 4: -.3, 6: -.3, 9: .6, 10: -.3}


class RobotLogTests(unittest.TestCase):
    def test_log_rows_keep_the_columns_older_readers_use(self):
        motor = SimpleNamespace(q=.1, dq=.2, tau_est=.3)
        state = SimpleNamespace(motor_state=[motor] * 35, imu_state=SimpleNamespace(rpy=[0, .1, 0],
                                                                                   quaternion=[1, 0, 0, 0]))
        row = record_lowstate.state_row(state)
        self.assertEqual(len(row), len(record_lowstate.STATE_HEADER) - 1)
        self.assertEqual(record_lowstate.STATE_HEADER[1:33], [f"q{i}" for i in range(29)] + ["roll", "pitch", "yaw"])
        self.assertEqual(row[:29], [.1] * 29)             # grab_report reads columns 1-29 as the joint angles
        command = SimpleNamespace(motor_cmd=[SimpleNamespace(q=.5, kp=40, kd=2, tau=0)] * 35)
        self.assertEqual(len(record_lowstate.command_row(command)), len(record_lowstate.COMMAND_HEADER) - 1)
        self.assertEqual(record_lowstate.command_path(Path("a/joints.csv")), Path("a/joints_cmd.csv"))

    @unittest.skipUnless(importlib.util.find_spec("mujoco") and (UPSTREAM / "gear_sonic").is_dir(),
                         "needs MuJoCo and the upstream robot model (sim venv)")
    def test_robot_log_becomes_a_replayable_standing_trajectory(self):
        import mujoco
        from gpt_astra.tools import robot_to_sim
        pitch, yaw = .1, 1.2
        quaternion = robot_to_sim.euler_quaternion(0, pitch, yaw)
        joints = [LEGS.get(i, 0.) for i in range(29)]
        with tempfile.TemporaryDirectory() as tmp:
            for header, extra in ((record_lowstate.STATE_HEADER, [*quaternion, *[0.] * 58]),
                                  (record_lowstate.STATE_HEADER[:33], [])):    # new and old (rpy-only) logs
                log, out = Path(tmp) / "robot.csv", Path(tmp) / "replay"
                with log.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    for t in range(3):
                        writer.writerow([1000 + .02 * t, *joints, 0, pitch, yaw, *extra])
                self.assertEqual(robot_to_sim.convert(log, out, UPSTREAM), 3)
                meta = json.loads((out / "trajectory_meta.json").read_text())
                model = mujoco.MjModel.from_xml_path(meta["scene_path"])
                data = mujoco.MjData(model)
                rows = list(csv.reader((out / "trajectory.csv").open()))
                self.assertEqual(len(rows[0]), 2 + model.nq + model.nv + model.nu)   # what replay.py checks
                data.qpos[:] = [float(v) for v in rows[1][2:2 + model.nq]]
                mujoco.mj_kinematics(model, data)
                address = {j["name"]: j["qpos_address"] for j in meta["joints"]}
                self.assertAlmostEqual(data.qpos[address["left_knee_joint"]], .6)
                w, x, y, z = data.qpos[3:7]
                self.assertAlmostEqual(math.asin(2 * (w * y - z * x)), pitch, places=3)
                self.assertAlmostEqual(robot_to_sim.heading(data.qpos[3:7]), 0, places=3)   # faces forward
                stander = robot_to_sim.Stander(UPSTREAM, [])
                gap = min(mujoco.mj_geomDistance(model, data, stander.floor, g, 2., None) for g in stander.feet)
                self.assertAlmostEqual(gap, 0, delta=.001)

