"""Task snapshots, simulated standing handoff, and target geometry."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from gpt_astra.motions import ready_pose
from gpt_astra.sim import standing_command_received
from gpt_astra.task_library import snapshot_task
from gpt_astra.tasks.catch_balloon.sim import TaskSimulation, prepare_scene
from sonic_laptop.model import load_model

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT.parent / "GR00T-WholeBodyControl"
SCENE = UPSTREAM / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"


class TaskLibraryTests(unittest.TestCase):
    def test_new_task_and_override_are_snapshotted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks/reach").mkdir(parents=True)
            (root / "tasks/reach/task.txt").write_text("Reach the target\n")
            (root / "tasks/reach/sim.py").write_text("# local task simulation\n")
            with patch("gpt_astra.task_library.TASKS", root / "tasks"):
                self.assertEqual(snapshot_task("reach", None, root / "run1"), "Reach the target")
                self.assertEqual(snapshot_task("reach", "Other instruction", root / "run2"),
                                 "Other instruction")
                self.assertEqual((root / "run2/task.txt").read_text(), "Other instruction\n")
                self.assertEqual((root / "run1/sim.py").read_text(), "# local task simulation\n")
                with self.assertRaises(ValueError):
                    snapshot_task("../reach", None, root / "run3")

    def test_damping_only_does_not_release_simulated_stand(self):
        damping = [types.SimpleNamespace(kp=0, kd=8, tau=0) for _ in range(29)]
        self.assertFalse(standing_command_received(damping))
        damping[15].kp = 10
        self.assertTrue(standing_command_received(damping))


class BalloonSimulationTests(unittest.TestCase):
    def test_sphere_falls_and_resets_after_ground_contact(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("Optional test requires MuJoCo")
        if not SCENE.is_file():
            self.skipTest("Upstream MuJoCo scene not installed")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model = mujoco.MjModel.from_xml_path(str(prepare_scene(SCENE, run)))
            data = mujoco.MjData(model)
            robot = load_model(UPSTREAM)
            for name, value in zip(robot["names"], ready_pose(robot)):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            data.qpos[:7] = [0, 0, .793, 1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            task = TaskSimulation(model, data, run / "task-start")
            data.time = 20
            task.step()
            self.assertFalse(task.active)
            (run / "task-start").write_text("start\n")
            task.step()
            positions, modes = [], []
            for _ in range(2):
                data.time = task.started_at + task.next_spawn + .01
                task.step()
                mujoco.mj_forward(model, data)
                self.assertTrue(task.active)
                self.assertEqual(model.geom_type[task.geom], mujoco.mjtGeom.mjGEOM_SPHERE)
                self.assertEqual(model.geom_rgba[task.geom, 0], 1)
                position = data.xpos[task.body].copy()
                positions.append(position)
                modes.append(data.qvel[task.v])
                cam = model.camera("head_camera").id
                self.assertGreater(position[2], data.cam_xpos[cam, 2] + .15)
                if len(positions) == 1:
                    robot_q = data.qpos[:task.q].copy()
                    initial_height = float(data.qpos[task.q+2])
                    self.assertTrue(np.allclose(data.xfrc_applied[task.body], 0))
                    for _ in range(50):
                        task.step()
                        mujoco.mj_step(model, data)
                        data.qpos[:task.q] = robot_q
                        data.qvel[:task.v] = 0
                        mujoco.mj_forward(model, data)
                    self.assertLess(data.qpos[task.q+2], initial_height-.02)
                    visible_in_flight = False
                    for _ in range(600):
                        task.step()
                        if not task.active:
                            break
                        mujoco.mj_step(model, data)
                        data.qpos[:task.q] = robot_q
                        data.qvel[:task.v] = 0
                        mujoco.mj_forward(model, data)
                        relative = data.cam_xmat[cam].reshape(3, 3).T @ (
                            data.xpos[task.body] - data.cam_xpos[cam])
                        depth = -relative[2]
                        height = depth * np.tan(np.deg2rad(model.cam_fovy[cam])/2)
                        visible_in_flight |= (depth > 0 and abs(relative[0]) < height * 640/480
                                              and abs(relative[1]) < height)
                    self.assertTrue(visible_in_flight, "Target never entered the ego camera view")
                else:
                    data.qpos[task.q+2] = .05
                    mujoco.mj_forward(model, data)
                    task.step()
                self.assertFalse(task.active)
            self.assertFalse(np.allclose(positions[0], positions[1]))
            self.assertAlmostEqual(modes[0], 0)
            self.assertLess(modes[1], 0)
            self.assertIn('"event": "ground"', (run / "task_events.jsonl").read_text())

    def test_detected_two_hand_catch_resets_after_five_seconds(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("Optional test requires MuJoCo")
        if not SCENE.is_file():
            self.skipTest("Upstream MuJoCo scene not installed")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model = mujoco.MjModel.from_xml_path(str(prepare_scene(SCENE, run)))
            real = mujoco.MjData(model)
            task = TaskSimulation(model, real, run / "task-start")
            left = int(model.body_geomadr[model.body("left_wrist_yaw_link").id])
            right = int(model.body_geomadr[model.body("right_wrist_yaw_link").id])
            data = types.SimpleNamespace(time=1.0, qpos=real.qpos.copy(), qvel=real.qvel.copy(),
                xfrc_applied=real.xfrc_applied.copy(), ncon=2,
                contact=[types.SimpleNamespace(geom1=task.geom, geom2=left),
                         types.SimpleNamespace(geom1=task.geom, geom2=right)])
            data.qpos[task.q+2] = 1.0
            task.data, task.started_at, task.active, task.attempt = data, 0, True, 1
            task.step()
            data.time = 1.31
            task.step()
            self.assertEqual(task.caught_at, 1.31)
            data.time = 6.32
            task.step()
            self.assertFalse(task.active)
            events = (run / "task_events.jsonl").read_text()
            self.assertIn('"event": "caught"', events)
            self.assertIn('"event": "reset_after_catch"', events)


if __name__ == "__main__":
    unittest.main()
