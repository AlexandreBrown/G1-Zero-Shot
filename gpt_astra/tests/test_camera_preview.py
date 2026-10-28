"""Camera setup gates and an actual Tk window on an isolated virtual display."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from gpt_astra.camera import Frame, Frames, jpeg_thumbnail
from gpt_astra.preview import CameraPreview
from gpt_astra.run import Runner, parse_args


class SilentCamera(Frames):
    def __init__(self, *args):
        super().__init__()

    def poll(self, state=None):
        pass

    def close(self):
        pass


class CameraSetupTests(unittest.TestCase):
    def test_missing_stream_errors_without_policy_request_or_actuation(self):
        args = parse_args(["--policy", "fake", "--auto-confirm", "--camera-timeout", ".1"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.MockCamera", SilentCamera), \
             patch("gpt_astra.run.MockSession.arm") as arm, \
             patch("gpt_astra.run.FakePolicy.decide") as decide, redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                with self.assertRaisesRegex(TimeoutError, "No live camera view"):
                    runner.run()
                arm.assert_not_called()
                decide.assert_not_called()
            finally:
                runner.cleanup()

    def test_stream_stall_at_enter_gate_is_detected_without_confirmation(self):
        args = parse_args(["--policy", "fake", "--auto-confirm"])
        args.auto_confirm = False
        runner = Runner(args)
        runner.camera = SilentCamera()
        runner.camera.add(Frame(b"unused", time.monotonic()-.4, "ego"))
        runner.camera_ready = True
        # No ENTER is pressed. The gate must keep checking stream freshness itself.
        def no_input(*args):
            time.sleep(.01)
            return [], [], []
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.select.select", no_input), \
             redirect_stdout(io.StringIO()):
            runner.directory = Path(tmp)
            with self.assertRaisesRegex(TimeoutError, "Camera missing/stale"):
                runner.gate("Confirm the camera")

    def test_no_desktop_errors_before_policy_or_controller_start(self):
        args = parse_args(["--policy", "fake", "--auto-confirm", "--preview"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.CameraPreview", side_effect=RuntimeError("Cannot open camera preview")), \
             patch("gpt_astra.run.MockSession.arm") as arm, \
             patch("gpt_astra.run.FakePolicy.decide") as decide, redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                with self.assertRaisesRegex(RuntimeError, "Cannot open camera preview"):
                    runner.run()
                arm.assert_not_called()
                decide.assert_not_called()
            finally:
                runner.cleanup()


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("Xvfb"):
            raise unittest.SkipTest("Optional GUI tests require Xvfb")
        try:
            import tkinter
            from PIL import ImageTk
        except ImportError:
            raise unittest.SkipTest("Optional GUI tests require Tk/ImageTk")
        cls.server = subprocess.Popen(["Xvfb", "-displayfd", "1", "-screen", "0", "1024x768x24",
                                       "-nolisten", "tcp"], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
        if not select.select([cls.server.stdout], [], [], 5)[0]:
            cls.server.terminate()
            cls.server.communicate(timeout=5)
            raise RuntimeError("Xvfb did not start")
        display = cls.server.stdout.readline().strip()
        if not display.isdigit():
            cls.server.terminate()
            _, error = cls.server.communicate(timeout=5)
            raise RuntimeError("Xvfb failed: " + error)
        cls.environment = patch.dict(os.environ, {"DISPLAY": ":" + display})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.server.terminate()
        cls.server.communicate(timeout=5)

    def test_actual_window_renders_updates_reports_stale_and_detects_close(self):
        preview = CameraPreview("mock")
        try:
            self.assertFalse(preview.ready)
            for color in ("red", "blue"):
                frame = Frame(jpeg_thumbnail(Image.new("RGB", (640, 480), color)), time.monotonic(), "ego")
                preview.update(frame)
                self.assertTrue(preview.ready)
                self.assertIn("LIVE", preview.status.cget("text"))
                self.assertEqual(preview.rendered, frame.received)
                self.assertEqual(preview.canvas.bbox(preview.item), (0, 0, 640, 480))
                time.sleep(.11)
            old = Frame(frame.jpeg, time.monotonic()-1, "ego")
            preview.update(old)
            self.assertFalse(preview.ready)
            self.assertIn("STALE", preview.status.cget("text"))
            preview.update(stopped=True)
            self.assertIn("STOPPED", preview.status.cget("text"))
            preview.root.tk.call(preview.root.protocol("WM_DELETE_WINDOW"))
            with self.assertRaisesRegex(RuntimeError, "preview closed"):
                preview.update()
            preview.update(stopped=True)  # Shutdown remains possible with a closed window.
        finally:
            preview.close()

    def test_complete_mock_loop_with_live_window_and_camera_confirmation_first(self):
        args = parse_args(["--policy", "fake", "--auto-confirm", "--preview", "--max-decisions", "1"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                labels = [event["event"] for event in events]
                self.assertLess(labels.index("camera_confirmed"), labels.index("preflight_response"))
                self.assertLess(labels.index("camera_confirmed"), labels.index("suspended_checks_passed"))
                self.assertEqual(labels[-1], "finished")
                self.assertTrue(runner.preview.ready)
            finally:
                runner.cleanup()
            self.assertTrue(runner.preview.closed)

    def test_closed_preview_cannot_confirm_camera_or_start_controller(self):
        args = parse_args(["--policy", "fake", "--auto-confirm", "--preview"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.MockSession.arm") as arm, \
             patch("gpt_astra.run.FakePolicy.decide") as decide, redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                with patch.object(runner, "gate", lambda *a: runner.preview.close()):
                    with self.assertRaisesRegex(RuntimeError, "preview closed"):
                        runner.run()
                arm.assert_not_called()
                decide.assert_not_called()
            finally:
                runner.cleanup()


if __name__ == "__main__":
    unittest.main()
