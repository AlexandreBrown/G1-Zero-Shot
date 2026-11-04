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
from gpt_astra.policy import Astra, ClaudeCode, Decision, FakePolicy, RelayPolicy, validate_decision
from gpt_astra.calibration import SIM
from gpt_astra.follow import approach_x
from gpt_astra.run import MockSession, Runner, decision_allowed, parse_args
from sonic_laptop.model import load_model, clearance
from sonic_laptop.motions import read_clip
from sonic_laptop.session import Session

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


class ClaudeCodeTests(unittest.TestCase):
    """Fake `claude` executable: records argv/stdin/env and replies with stream-json events."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.executable = self.directory / "claude"
        self.actions = {"reach_left": "left", "hold": "wait"}
        self.frames = [Frame(b"jpeg-old", 1.0, "ego"), Frame(b"jpeg-new", 1.5, "ego")]

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, events, code=0, auth=None):
        if auth is None:
            auth = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
        record = self.directory / "record.json"
        self.executable.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1:] == ["auth", "status"]:
    print(json.dumps({auth!r}))
    sys.exit(0)
json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
           "billing_env": sorted(k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")))}},
          open({str(record)!r}, "w"))
for event in {events!r}:
    print(json.dumps(event))
sys.exit({code})
""")
        self.executable.chmod(0o755)
        return record

    def decide(self):
        billing = {"ANTHROPIC_API_KEY": "not-a-secret", "ANTHROPIC_BASE_URL": "http://gateway.invalid",
                   "CLAUDE_CODE_USE_BEDROCK": "1"}
        with patch.dict(os.environ, billing):
            return ClaudeCode(self.actions, model="haiku", executable=str(self.executable)).decide(
                "catch", self.frames, {"preflight": False})

    def test_payload_flags_and_structured_decision(self):
        record = self.fake([{"type": "system", "subtype": "init", "apiKeySource": "none"},
                            {"type": "result", "is_error": False, "structured_output":
                             {"action": "reach_left", "confidence": .9, "reason": "visible"}}])
        self.assertEqual(self.decide(), Decision("reach_left", .9, "visible"))
        sent = json.loads(record.read_text())
        self.assertEqual(sent["billing_env"], [])
        self.assertNotEqual(Path(sent["cwd"]).resolve(), ROOT)
        argv = sent["argv"]
        for flag in ("-p", "--safe-mode", "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--model") + 1], "haiku")
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertEqual(schema["properties"]["action"]["enum"], ["reach_left", "hold"])
        self.assertEqual(schema["properties"]["reason"]["maxLength"], 500)
        content = json.loads(sent["stdin"])["message"]["content"]
        self.assertEqual(json.loads(content[0]["text"])["available_actions"], self.actions)
        self.assertEqual([base64.b64decode(c["source"]["data"]) for c in content[1:]],
                         [b"jpeg-old", b"jpeg-new"])

    def test_api_key_billing_error_and_invalid_output_never_create_command(self):
        result = {"type": "result", "is_error": False, "structured_output":
                  {"action": "reach_left", "confidence": .9, "reason": "visible"}}
        for auth in ({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"},
                     {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "bedrock"}, {}):
            record = self.fake([result], auth=auth)
            with self.assertRaisesRegex(RuntimeError, "not logged in"):
                self.decide()
            self.assertFalse(record.exists())
        self.fake([{"type": "system", "subtype": "init", "apiKeySource": "ANTHROPIC_API_KEY"}, result])
        with self.assertRaisesRegex(RuntimeError, "not the subscription"):
            self.decide()
        init = {"type": "system", "subtype": "init", "apiKeySource": "none"}
        for events, code in (([init, {**result, "is_error": True}], 0), ([init, result], 1), ([init], 0),
                             ([init, {**result, "structured_output": {"action": "walk", "confidence": 1,
                                                                     "reason": ""}}], 0)):
            self.fake(events, code)
            with self.assertRaises(ValueError):
                self.decide()


class RelayPolicyTests(unittest.TestCase):
    def test_request_file_reply_error_and_timeout(self):
        frames = [Frame(b"jpeg", 1.0, "ego")]
        with tempfile.TemporaryDirectory() as tmp:
            relay = Path(tmp)
            policy = RelayPolicy({"catch": "c", "hold": "h"}, relay, timeout=5)
            def answer(reply):
                for _ in range(250):
                    pending = list((relay / "requests").glob("[0-9]*.json"))
                    if pending:
                        break
                    time.sleep(.02)
                request = json.loads(pending[0].read_text())
                pending[0].unlink()
                self.assertEqual(base64.b64decode(request["frames"][0]["jpeg"]), b"jpeg")
                self.assertEqual(request["actions"], {"catch": "c", "hold": "h"})
                (relay / "decisions" / pending[0].name).write_text(json.dumps(reply))
            for reply in ({"decision": {"action": "catch", "confidence": .9, "reason": "r"}}, {"error": "boom"}):
                worker = threading.Thread(target=answer, args=(reply,))
                worker.start()
                if "error" in reply:
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        policy.decide("task", frames, {})
                else:
                    self.assertEqual(policy.decide("task", frames, {}), Decision("catch", .9, "r"))
                worker.join()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                RelayPolicy({"hold": "h"}, relay / "quiet", timeout=.3).decide("task", frames, {})


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
        for extra in (["--auto-confirm"], ["--policy", "fake"], ["--image", "x.jpg"], ["--policy", "claude"],
                      ["--pause-world"], ["--walk"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--backend", "robot", "--interface", "eth0", *extra])
        self.assertEqual(parse_args(["--policy", "claude", "--auto-confirm"]).api_timeout, 60)
        self.assertEqual(parse_args(["--auto-confirm"]).api_timeout, 15)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--backend", "robot", "--interface", "eth0", "--policy", "relay", "--relay-dir", "x"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--policy", "relay", "--auto-confirm"])

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

    def test_walk_primitives_through_planner_then_back_to_reach_clips(self):
        args = parse_args(["--backend", "mock", "--policy", "fake", "--auto-confirm", "--walk",
                           "--task-name", "approach_balloon", "--max-decisions", "8"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                decisions = [e["action"] for e in events if e["event"] == "decision"]
                self.assertEqual(decisions, ["reach_left", "reach_right", "reach_both", "walk_forward",
                                             "walk_backward", "step_left", "step_right", "reach_left"])
                self.assertEqual([e["action"] for e in events if e["event"] == "walk"], decisions[3:7])
                acks = [e["command"] for e in events if e["event"] == "command_ack"]
                self.assertEqual(acks, ["grounded_start", "motion", "motion", "motion", "planner", "idle",
                                        "idle", "idle", "idle", "reference", "motion"])
                self.assertEqual(events[-1]["event"], "finished")
            finally:
                runner.cleanup()

    def test_follow_loop_walks_grabs_holds_and_releases(self):
        args = parse_args(["--backend", "mock", "--policy", "fake", "--auto-confirm", "--follow",
                           "--task-name", "follow_balloon", "--follow-seconds", "14", "--max-decisions", "1000"])
        start = []
        def ball(jpeg, cal=None, held=False):
            start.append(time.monotonic())
            elapsed = start[-1] - start[0]
            if elapsed < 2:
                return np.array([.8, .2, .16])   # to the left: side-step left first
            if elapsed < 7:
                # In the approach zone and, as the mock robot does not settle back, already within a grab's reach.
                return np.array([approach_x(SIM) + SIM.settle_back / 2, 0., .16])
            return np.array([.9, -.3, .16])      # pulled away
        catch = Decision("catch", 1, "fixture: go after the balloon")
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.locate", ball), patch.object(FakePolicy, "decide", return_value=catch), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            commands = []
            try:
                with patch.object(MockSession, "command", autospec=True,
                                  side_effect=lambda session, command, argument=0: commands.append(
                                      (command, argument)) or Session.command(session, command, argument)):
                    runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                self.assertTrue(all(e["action"] == "catch" for e in events if e["event"] == "decision"))
                self.assertEqual([e["reason"] for e in events if e["event"] == "let_go"], ["pulled out of the hands"])
                moves = [argument for command, argument in commands if command == "move"]
                self.assertEqual(moves[0], 2)
                self.assertEqual(moves[-1], 3)             # follows the pulled balloon to the right
                acks = [e["command"] for e in events if e["event"] == "command_ack"]
                self.assertEqual(acks[:6], ["grounded_start", "planner", "idle", "reference", "motion", "motion"])
                self.assertEqual(acks[6], "planner")
            finally:
                runner.cleanup()

    def test_follow_pauses_standing_when_the_policy_keeps_failing(self):
        args = parse_args(["--backend", "mock", "--policy", "fake", "--auto-confirm", "--follow",
                           "--task-name", "follow_balloon", "--follow-seconds", "30", "--max-decisions", "1000"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.locate", lambda jpeg, cal=None, held=False: np.array([.9, 0., .16])), \
             patch.object(FakePolicy, "decide", side_effect=[Decision("hold", 1, "preflight"),
                                                             RuntimeError("Claude Code timed out"),
                                                             RuntimeError("Claude Code timed out")]), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                names = [e["event"] for e in events]
                self.assertEqual(names.count("policy_error"), 2)
                self.assertIn("policy_pause", names)
                self.assertEqual(names[-1], "finished")   # the normal end: supported, then stopped; no damping
            finally:
                runner.cleanup()

    def test_follow_pauses_standing_when_the_camera_goes_stale(self):
        args = parse_args(["--backend", "mock", "--policy", "fake", "--auto-confirm", "--follow",
                           "--task-name", "follow_balloon", "--follow-seconds", "30", "--max-decisions", "1000"])
        stale_after = []
        def fresh_frames(runner):
            if not stale_after:
                stale_after.append(time.monotonic() + 3)
            return None if time.monotonic() > stale_after[0] else Frames.latest(runner.camera)
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.locate", lambda jpeg, cal=None, held=False: np.array([.9, 0., .16])), \
             patch.object(Runner, "fresh_frames", autospec=True, side_effect=fresh_frames), \
             patch("gpt_astra.run.START_SETTLE", .5), redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                pauses = [e["reason"] for e in events if e["event"] == "policy_pause"]
                self.assertEqual(len(pauses), 1)
                self.assertIn("camera stale", pauses[0])
                self.assertEqual(events[-1]["event"], "finished")
            finally:
                runner.cleanup()

    def test_follow_refuses_a_task_without_the_grab_poses(self):
        args = parse_args(["--backend", "mock", "--policy", "fake", "--auto-confirm", "--follow",
                           "--task-name", "catch_balloon"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                with self.assertRaisesRegex(ValueError, "grab poses"):
                    runner.run()
                self.assertIsNone(runner.session)   # refused before any controller started
            finally:
                runner.cleanup()

    def test_measure_grab_plays_each_pose_and_logs_its_hold_window(self):
        args = parse_args(["--backend", "mock", "--policy", "none", "--auto-confirm", "--measure", "grab",
                           "--task-name", "static_balloon"])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.GRAB_CHECK", ("2_10", "1_14")), patch("gpt_astra.run.GRAB_HOLD", .2), \
             patch("gpt_astra.run.START_SETTLE", .5), \
             redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                holds = [e for e in events if e["event"] == "grab_hold"]
                self.assertEqual([h["grab"] for h in holds], ["2_10", "1_14"])
                self.assertTrue(all(h["end"] - h["start"] >= .2 for h in holds))
                played = [e["argument"] for e in events if e["event"] == "command_ack" and e["command"] == "motion"]
                self.assertEqual(played, [runner.indexes[n] for n in ("grab_2_10", "release_2_10", "grab_1_14",
                                                                      "release_1_14")])
            finally:
                runner.cleanup()

    def test_measure_walk_logs_probes_against_the_balloon_on_its_stand(self):
        args = parse_args(["--backend", "mock", "--policy", "none", "--auto-confirm", "--measure", "walk",
                           "--task-name", "static_balloon"])
        robot = {"x": 0., "y": 0., "last": None}
        def command(session, name, argument=0):
            if name == "move":       # mock physics: the robot moves at the commanded speed while moves arrive
                now = time.monotonic()
                dt = min(now - robot["last"], .1) if robot["last"] else 0.
                speed = .2 if argument >= 4 else .4
                robot["x"] += (1, -1, 0, 0)[argument % 4] * speed * dt
                robot["y"] += (0, 0, 1, -1)[argument % 4] * speed * dt
                robot["last"] = now
            else:
                robot["last"] = None
            return Session.command(session, name, argument)
        def locate(jpeg, cal=None, held=False):
            return np.array([.7 - robot["x"], -robot["y"], .16])
        with tempfile.TemporaryDirectory() as tmp, patch("gpt_astra.run.RUNTIME", Path(tmp)), \
             patch("gpt_astra.run.locate", locate), patch.object(MockSession, "command", autospec=True,
                                                                 side_effect=command), \
             patch("gpt_astra.run.WALK_PROBES", (("forward", 0, .5), ("back_fine", 5, .5))), \
             patch("gpt_astra.run.WALK_REPEATS", 1), patch("gpt_astra.run.WALK_SETTLE", .2), \
             patch("gpt_astra.run.STEADY", .2), patch("gpt_astra.run.START_SETTLE", .5), redirect_stdout(io.StringIO()):
            runner = Runner(args)
            try:
                runner.run()
                events = [json.loads(line) for line in (runner.directory / "events.jsonl").read_text().splitlines()]
                probes = {e["move"]: e for e in events if e["event"] == "walk_probe"}
                self.assertEqual(sorted(probes), ["back_fine", "forward"])
                forward, back = probes["forward"], probes["back_fine"]
                self.assertAlmostEqual(forward["before"][0] - forward["after"][0], .2, delta=.05)  # 0.5 s at 0.4
                self.assertAlmostEqual(back["after"][0] - back["before"][0], .1, delta=.05)        # 0.5 s at 0.2
                self.assertIn("settle_probe", [e["event"] for e in events])
            finally:
                runner.cleanup()

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
