import base64
from contextlib import redirect_stdout, redirect_stderr
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpt_astra.camera import Frame, Frames, MockCamera, ZmqCamera, jpeg_thumbnail
from gpt_astra.motions import prepare_session, ready_pose
from gpt_astra.tasks.catch_balloon.motions import generate_library
from gpt_astra.policy import Astra, Decision, validate_decision
from gpt_astra.run import MockSession, Runner, decision_allowed, parse_args
from sonic_laptop.model import load_model, clearance
from sonic_laptop.motions import read_clip

UPSTREAM = ROOT.parent / "GR00T-WholeBodyControl"


class ApiFixture:
    """Real HTTP transport, deterministic fake Responses service; no OpenAI calls."""
    def __init__(self):
        self.requests = []
        self.reply = None
        self.delay = 0
        self.status = 200
        self.headers = {}
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(request)
                time.sleep(outer.delay)
                user = json.loads(request["input"][0]["content"][0]["text"])
                action = ("hold" if user["controller"]["preflight"] else
                          next((name for name in user["available_actions"] if name not in ("hold", "stop")), "hold"))
                reply = outer.reply or {"status": "completed", "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": json.dumps({"action": action, "confidence": .95,
                                                               "reason": "fixture only"})}]}]}
                raw = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in outer.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/responses"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ApiFixture()
        self.actions = {"hold": "hold", "stop": "stop", "tap_left_small": "left upward tap"}
        self.client = Astra("test-key-not-a-secret", self.actions, endpoint=self.fixture.url)
        jpeg = jpeg_thumbnail(Image.new("RGB", (64, 64), "red"))
        self.frame = Frame(jpeg, time.monotonic(), "test")

    def tearDown(self):
        self.fixture.close()

    def test_real_http_payload_images_model_and_schema(self):
        d = self.client.decide("tap the balloon", [self.frame], {"preflight": False})
        self.assertEqual(d.action, "tap_left_small")
        request = self.fixture.requests[0]
        self.assertEqual(request["model"], "gpt-6-astra")
        self.assertFalse(request["store"])
        self.assertTrue(request["text"]["format"]["strict"])
        data = request["input"][0]["content"][1]["image_url"].split(",")[1]
        self.assertEqual(base64.b64decode(data), self.frame.jpeg)
        self.assertNotIn("test-key-not-a-secret", json.dumps(request))

    def test_refusal_and_incomplete_never_create_command(self):
        self.fixture.reply = {"status": "completed", "output": [{"content": [{"type": "refusal"}]}]}
        with self.assertRaisesRegex(ValueError, "refused"):
            self.client.decide("test", [self.frame], {"preflight": False})
        self.fixture.reply = {"status": "incomplete", "output": []}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.client.decide("test", [self.frame], {"preflight": False})

    def test_http_429_classification_redaction_and_no_retry(self):
        self.fixture.status = 429
        self.fixture.headers = {"Retry-After": "12"}
        cases = [
            ("credit_balance_exhausted", "insufficient_quota", "add credits"),
            ("insufficient_quota", "insufficient_quota", "spending limits"),
            ("rate_limit_exceeded", "tokens", "Retry-After: 12 seconds"),
            ("slow_down", "rate_limit_error", "reduce request rate"),
            ("test-key-not-a-secret", "unrecognized", "Cause not supplied or unrecognized"),
        ]
        for count, (code, kind, expected) in enumerate(cases, 1):
            with self.subTest(code=code):
                self.fixture.reply = {"error": {"code": code, "type": kind,
                                               "message": "Echoed key: test-key-not-a-secret"}}
                with self.assertRaises(RuntimeError) as caught:
                    self.client.decide("test", [self.frame], {"preflight": False})
                message = str(caught.exception)
                self.assertIn(expected, message)
                self.assertIn("No automatic retry", message)
                self.assertNotIn("test-key-not-a-secret", message)
                self.assertEqual(len(self.fixture.requests), count)

    def test_http_429_malformed_body_still_reports_status(self):
        self.fixture.status = 429
        for body in (b"not JSON", {"error": "unexpected"}, ["unexpected"]):
            self.fixture.reply = body
            with self.assertRaisesRegex(RuntimeError, "OpenAI HTTP 429: Cause not supplied"):
                self.client.decide("test", [self.frame], {"preflight": False})

    def test_local_validation_rejects_unknown_nonfinite_and_extra(self):
        for value in ({"action": "jump", "confidence": 1, "reason": "x"},
                      {"action": "hold", "confidence": float("nan"), "reason": "x"},
                      {"action": "hold", "confidence": True, "reason": "x"},
                      {"action": "hold", "confidence": 1, "reason": "x", "joints": [0]*29}):
            with self.assertRaises(ValueError):
                validate_decision(value, self.actions)

    def test_post_network_freshness_busy_and_confidence_gates(self):
        good = Decision("tap_left_small", .9, "x")
        state = {"state": 2, "playing": False}
        self.assertTrue(decision_allowed(good, 10, 11, state, 2, .8)[0])
        self.assertFalse(decision_allowed(good, 10, 13, state, 2, .8)[0])
        self.assertFalse(decision_allowed(good, 10, 11, {**state, "playing": True}, 2, .8)[0])
        self.assertFalse(decision_allowed(good, 10, 11, {"state": 1}, 2, .8)[0])
        self.assertFalse(decision_allowed(good, 10, 11, {**state, "fault": True}, 2, .8)[0])
        self.assertFalse(decision_allowed(Decision("tap_left_small", .1, "x"), 10, 11, state, 2, .8)[0])


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.model = load_model(UPSTREAM)
        cls.library = generate_library(cls.path / "library", cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_references_pass_existing_limits_and_endpoints(self):
        directory = self.path / "session"
        directory.mkdir()
        _, catalog, indexes, actions = prepare_session(directory, UPSTREAM, self.library)
        self.assertEqual(len(catalog), 4)
        self.assertEqual(len(indexes), 3)
        self.assertEqual(set(actions), {"hold", "stop", *indexes})
        for name in indexes:
            arrays, q = read_clip(self.library / name, self.model)
            self.assertTrue(np.allclose(q[0], q[-1]))
            self.assertTrue(np.allclose(q[:, :15], q[0, :15]))
            self.assertLess(np.max(np.abs(arrays["joint_vel"])), 4)

    def test_independent_mujoco_fk_reach_moves_hand_forward_and_upward(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("Optional MuJoCo not installed")
        model = mujoco.MjModel.from_xml_path(str(UPSTREAM / "gear_sonic_deploy/g1/g1_29dof.xml"))
        data = mujoco.MjData(model)
        def position(q, side):
            for name, value in zip(self.model["names"], q):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            mujoco.mj_forward(model, data)
            return data.body(side + "_wrist_yaw_link").xpos.copy()
        for side in ("left", "right"):
            _, q = read_clip(self.library / f"reach_{side}", self.model)
            start, peak = position(q[0], side), position(q[65], side)
            self.assertGreater(peak[2]-start[2], .10)
            self.assertGreater(peak[0]-start[0], .08)
            self.assertGreater(min(clearance(self.model, pose) for pose in q), .025)

    def test_generated_clips_through_actual_cpp_reader_and_command_gate(self):
        if not shutil.which("g++") or not Path("/usr/include/eigen3/Eigen").exists():
            self.skipTest("Optional C++/Eigen headers not installed")
        from sonic_laptop.prepare import patch as prepare_deploy, INPUT
        tree = prepare_deploy(UPSTREAM / "gear_sonic_deploy", self.path / "deploy")
        include = tree / "src/g1/g1_deploy_onnx_ref/include"
        (include / "guarded_input.inc").write_text(INPUT)
        session = self.path / "native-session"
        session.mkdir()
        prepare_session(session, UPSTREAM, self.library)
        binary = self.path / "native-motion"
        subprocess.run(["g++", "-std=c++17", "-O2", "-fno-fast-math", "-I/usr/include/eigen3",
                        "-I" + str(include), str(Path(__file__).with_name("native_motion.cpp")),
                        "-lyaml-cpp", "-lz", "-o", str(binary)], check=True, capture_output=True)
        subprocess.run([str(binary), str(session)], check=True, timeout=10, capture_output=True)


class CameraTests(unittest.TestCase):
    def test_old_frames_are_not_refreshed_by_reading(self):
        frames = Frames()
        frames.add(Frame(b"jpeg", time.monotonic()-1, "ego"))
        with self.assertRaises(TimeoutError):
            frames.latest()

    def test_zmq_upstream_raw_and_base64_jpeg_protocol(self):
        import msgpack
        import zmq
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        camera = ZmqCamera("127.0.0.1", port, "ego_view")
        image = jpeg_thumbnail(Image.new("RGB", (30, 40), "blue"))
        try:
            time.sleep(.2)
            for content in (image, base64.b64encode(image).decode()):
                message = {"timestamps": {"ego_view": time.time()}, "images": {"ego_view": content}}
                for _ in range(20):
                    publisher.send(msgpack.packb(message, use_bin_type=True))
                    time.sleep(.01)
                    camera.poll()
                    if camera.last_stamp == message["timestamps"]["ego_view"]:
                        break
                self.assertTrue(camera.latest()[0].jpeg.startswith(b"\xff\xd8"))
            before = camera.history[-1].received
            publisher.send(msgpack.packb(message, use_bin_type=True))
            time.sleep(.02)
            camera.poll()
            self.assertEqual(camera.history[-1].received, before)
        finally:
            camera.close()
            publisher.close(linger=0)
            context.term()


class ConfigurationTests(unittest.TestCase):
    def test_model_default_environment_and_cli_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse_args(["--auto-confirm"]).model, "gpt-6-astra")
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-6.1-sol"}):
            self.assertEqual(parse_args(["--auto-confirm"]).model, "gpt-6.1-sol")
            self.assertEqual(parse_args(["--auto-confirm", "--model", "gpt-6-astra"]).model,
                             "gpt-6-astra")
        with patch.dict(os.environ, {"OPENAI_MODEL": " "}), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--auto-confirm"])


class IntegrationTests(unittest.TestCase):
    def test_hardware_cannot_bypass_gates_or_use_fake_policy(self):
        for extra in (["--auto-confirm"], ["--policy", "fake"], ["--image", "x.jpg"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--backend", "robot", "--interface", "eth0", *extra])

    def test_complete_app_through_real_http_and_mock_sonic_ipc(self):
        fixture = ApiFixture()
        def local_astra(key, actions, **kwargs):
            return Astra(key, actions, endpoint=fixture.url, **kwargs)
        try:
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-6.1-sol"}):
                args = parse_args(["--backend", "mock", "--auto-confirm", "--max-decisions", "1"])
            with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
                 patch("gpt_astra.run.Astra", local_astra), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
                runner = Runner(args)
                try:
                    runner.run()
                    events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                    labels = [e["event"] for e in events]
                    self.assertEqual(events[0]["model"], "gpt-6.1-sol")
                    self.assertLess(labels.index("preflight_response"), labels.index("suspended_checks_passed"))
                    self.assertLess(labels.index("standing_reference_active"), labels.index("decision"))
                    self.assertTrue((runner.directory / "task/task.txt").is_file())
                    for event in events:
                        if event["event"] == "observation":
                            self.assertTrue(all((runner.directory / image).read_bytes().startswith(b"\xff\xd8")
                                                for image in event["images"]))
                    acks = [e for e in events if e["event"] == "command_ack"]
                    self.assertEqual([e["command"] for e in acks], ["grounded_start", "motion"])
                    self.assertEqual(labels[-1], "finished")
                finally:
                    runner.cleanup()
                self.assertIsNotNone(runner.session.process.poll())
                self.assertEqual(len(fixture.requests), 2)
                self.assertTrue(all(r["model"] == "gpt-6.1-sol" for r in fixture.requests))
        finally:
            fixture.close()

    def test_api_quota_failure_stops_before_controller_start(self):
        fixture = ApiFixture()
        fixture.status = 429
        fixture.reply = {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}
        def local_astra(key, actions, **kwargs):
            return Astra(key, actions, endpoint=fixture.url, **kwargs)
        try:
            args = parse_args(["--backend", "mock", "--auto-confirm", "--max-decisions", "1"])
            with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
                 patch("gpt_astra.run.Astra", local_astra), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
                runner = Runner(args)
                try:
                    with self.assertRaisesRegex(RuntimeError, "insufficient_quota"):
                        runner.run()
                    self.assertIsNone(runner.session.process)
                    self.assertEqual(len(fixture.requests), 1)
                finally:
                    runner.cleanup()
        finally:
            fixture.close()

    def test_mock_latches_when_lease_stops_and_never_rearms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "catalog.json").write_text('[{"index":0,"frames":50}]')
            session = MockSession(path, path, "lo")
            try:
                session.arm()
                time.sleep(.6)
                self.assertTrue(session.poll()["fault"])
                self.assertTrue(session.stopped)
                with self.assertRaises(RuntimeError):
                    session.command("grounded_start")
            finally:
                session.terminate_supported()
                session.process.wait(timeout=2)

    def test_delayed_api_response_holds_while_controller_heartbeat_continues(self):
        fixture = ApiFixture()
        fixture.delay = .4  # Longer than native lease expiry; event loop must keep servicing it.
        def local_astra(key, actions, **kwargs):
            return Astra(key, actions, endpoint=fixture.url, **kwargs)
        try:
            args = parse_args(["--backend", "mock", "--auto-confirm", "--max-decisions", "1",
                               "--max-decision-age", ".1"])
            with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
                 patch("gpt_astra.run.Astra", local_astra), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), redirect_stdout(io.StringIO()):
                runner = Runner(args)
                try:
                    runner.run()
                    events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                    self.assertEqual([e["command"] for e in events if e["event"] == "command_ack"],
                                     ["grounded_start"])
                    self.assertTrue(any(e.get("reason") == "decision based on stale image" for e in events))
                    self.assertEqual(events[-1]["event"], "finished")
                finally:
                    runner.cleanup()
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
