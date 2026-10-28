#!/usr/bin/env python3
"""One entry point: images -> selected GPT model -> bounded motion names -> guarded SONIC."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonic_laptop.paths import DEPLOY
from sonic_laptop.session import Session
from gpt_astra.camera import MockCamera, ZmqCamera
from gpt_astra.motions import RUNTIME, generate_standing_library, prepare_session
from gpt_astra.policy import Astra, FakePolicy
from gpt_astra.preview import CameraPreview
from gpt_astra.task_library import snapshot_task, task_directory, task_generator
from sonic_laptop.session import atomic_write


class MockSession(Session):
    def arm(self):
        if self.process or self.stopped:
            raise RuntimeError("Session already started/stopped")
        self.heartbeat()
        self.log = (self.directory / "controller.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("mock_controller.py")), str(self.directory)],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.directory / "pid").write_text(str(self.process.pid))


def decision_allowed(decision, frame_received, now, state, max_age, confidence):
    """Never queue commands from old images or while the previous motion is running."""
    if state.get("fault") or state.get("state") != 2:
        return False, "controller not active"
    if state.get("playing"):
        return False, "previous motion still active"
    if now-frame_received > max_age or now < frame_received:
        return False, "decision based on stale image"
    if decision.confidence < confidence:
        return False, "low confidence"
    return True, "accepted"


class Runner:
    def __init__(self, args):
        self.args = args
        self.directory = None
        self.session = self.camera = self.lock = self.events = self.simulator = self.sim_log = None
        self.preview = None
        self.camera_ready = False
        self.observation_index = 0
        self.last_preview = 0
        self.state = {}
        self.support_confirmed = False

    def log(self, kind, **values):
        item = {"time": time.time(), "event": kind, **values}
        self.events.write(json.dumps(item) + "\n")
        self.events.flush()
        print(f"[{kind}] " + json.dumps(values), flush=True)

    def prepare(self):
        a = self.args
        RUNTIME.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.directory = RUNTIME / name
        self.directory.mkdir(mode=0o700)
        self.events = (self.directory / "events.jsonl").open("w")
        print(f"Run folder: {self.directory}", flush=True)
        a.task = snapshot_task(a.task_name, a.task, self.directory / "task")
        if a.policy == "astra" and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY, or use --backend mock --policy fake for offline testing")
        if a.backend != "mock":
            # Share the same exclusive process lock as the existing laptop panel.
            self.lock = Path(f"/tmp/sonic-laptop-{os.getuid()}.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another guarded SONIC session is running") from None
        model, self.catalog, self.indexes, actions = prepare_session(
            self.directory, a.upstream, a.library, task_generator(self.directory / "task"))
        if a.backend != "mock":
            if not (a.build / "guard-model.json").is_file() or not (a.build / "target/release/g1_deploy_onnx_ref").is_file():
                raise ValueError("Run bash gpt_astra/setup.sh --gpu on the GPU laptop first")
            installed = json.loads((a.build / "guard-model.json").read_text())
            if model != installed:
                raise ValueError("Generated references and guarded build use different robot models; rebuild")
        self.policy = (FakePolicy(actions) if a.policy == "fake" else
                       Astra(os.environ["OPENAI_API_KEY"], actions, model=a.model, timeout=a.api_timeout))
        self.camera = (MockCamera(a.image) if a.backend == "mock" else
                       ZmqCamera(a.camera_host, a.camera_port, a.camera_key))
        if a.backend != "mock" or a.preview:
            self.preview = CameraPreview(a.backend)
        if a.backend == "sim":
            python = a.upstream / ".venv_sim/bin/python"
            if not python.is_file():
                raise ValueError("Install upstream install_scripts/install_mujoco_sim.sh first")
            self.sim_log = (self.directory / "simulator.log").open("w")
            self.simulator = subprocess.Popen(
                [str(python), str(Path(__file__).with_name("sim.py")),
                 "--upstream", str(a.upstream), "--idle", str(self.directory / "motions/0000"),
                 "--task-dir", str(self.directory / "task"), "--run-dir", str(self.directory),
                 "--camera-port", str(a.camera_port)], cwd=a.upstream,
                stdout=self.sim_log, stderr=subprocess.STDOUT, start_new_session=True)
        cls = MockSession if a.backend == "mock" else Session
        self.session = cls(a.build, self.directory, a.interface or "lo", a.backend == "sim",
                           self.lock.fileno() if self.lock else None)
        self.log("prepared", backend=a.backend, policy=a.policy,
                 model=a.model if a.policy != "fake" else None,
                 task_name=a.task_name, task=a.task, max_decisions=a.max_decisions,
                 max_decision_age=a.max_decision_age)

    def pump(self, camera_required=False, tolerate_fault=False):
        if self.session and self.session.process:
            # Heartbeat is driven by this event loop, never by the blocked API worker.
            self.session.heartbeat()
            status = self.session.poll()
            if status:
                self.state = status
            if self.session.process.poll() is not None:
                raise RuntimeError(f"Controller exited; inspect {self.directory / 'controller.log'}")
            if self.state.get("fault") or self.session.stopped:
                if not tolerate_fault:
                    raise RuntimeError(self.state.get("reason") or "Controller heartbeat/status fault")
            for line in self.session.read_activity():
                if "[Setup]" in line or "LATCHED" in line:
                    print(line, flush=True)
        if self.simulator and self.simulator.poll() is not None and not tolerate_fault:
            raise RuntimeError(f"Simulator exited; inspect {self.directory / 'simulator.log'}")
        if not tolerate_fault:
            self.camera.poll(self.state)
            if self.preview:
                self.preview.update(self.camera.history[-1] if self.camera.history else None)
            if camera_required or self.camera_ready:
                self.camera.latest()
            if self.camera.history and time.monotonic()-self.last_preview > 1:
                self.last_preview = time.monotonic()
                path = self.directory / "latest.jpg"
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(self.camera.history[-1].jpeg)
                tmp.replace(path)
        elif self.preview:
            try:
                self.preview.update(stopped=True)
            except RuntimeError:
                pass  # A failed GUI must not block the supported-shutdown confirmation.

    def check_camera(self):
        a = self.args
        source = "mock camera" if a.backend == "mock" else f"{a.camera_host}:{a.camera_port} / {a.camera_key}"
        print(f"Waiting up to {a.camera_timeout:g}s for fresh images from {source}…", flush=True)
        end = time.monotonic() + a.camera_timeout
        while time.monotonic() < end:
            self.pump()
            if len(self.camera.history) >= 2:
                self.camera.latest()
                if not self.preview or self.preview.ready:
                    self.camera_ready = True
                    break
            self.abort_key()
            time.sleep(.02)
        else:
            raise TimeoutError(f"No live camera view from {source} within {a.camera_timeout:g}s; "
                               "check the camera server/simulator, camera key and visible preview window")
        if self.preview:
            self.gate("CAMERA CHECK: confirm the camera window shows the correct, updating view. "
                      "ENTER confirms the image before API preflight and body setup.")
            self.pump(camera_required=True)
            if not self.preview.ready:
                raise RuntimeError("Camera preview must be visible and fresh when confirming")
            self.log("camera_confirmed", camera=self.camera.history[-1].name)

    def gate(self, text, tolerate_fault=False):
        # Consume already queued keystrokes so one ENTER cannot pass two gates.
        while not self.args.auto_confirm and select.select([sys.stdin], [], [], 0)[0]:
            queued = sys.stdin.readline()
            if queued == "":
                raise EOFError("Input closed; controller is not automatically terminated")
            if queued.strip().lower() in ("q", "stop"):
                raise KeyboardInterrupt
        print("\n" + text + "\nPress ENTER to confirm; q + ENTER to abort.", flush=True)
        if self.args.auto_confirm:
            self.log("mock_auto_confirmation", prompt=text)
            return
        while True:
            self.pump(tolerate_fault=tolerate_fault)
            if select.select([sys.stdin], [], [], .02)[0]:
                line = sys.stdin.readline()
                if line == "":
                    raise EOFError("Input closed; controller is not automatically terminated")
                if line.strip().lower() == "q":
                    raise KeyboardInterrupt
                if not line.strip():
                    return

    def wait_for(self, predicate, timeout, camera_required=False):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(camera_required=camera_required)
            if predicate():
                return
            self.abort_key()
            time.sleep(.02)
        raise TimeoutError("Timed out waiting for controller/camera; inspect session logs")

    def abort_key(self):
        if not self.args.auto_confirm and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":
                raise EOFError("Input closed")
            if line.strip().lower() in ("q", "stop"):
                raise KeyboardInterrupt

    def ask_policy(self, preflight=False):
        frames = self.camera.latest()
        self.observation_index += 1
        frame_directory = self.directory / "frames"
        frame_directory.mkdir(exist_ok=True)
        frame_names = []
        for i, frame in enumerate(frames):
            name = f"{self.observation_index:04d}_{'preflight' if preflight else 'decision'}_{i}.jpg"
            (frame_directory / name).write_bytes(frame.jpeg)
            frame_names.append("frames/" + name)
        self.log("observation", preflight=preflight, images=frame_names)
        observed = frames[-1].received
        state = {"preflight": preflight, "backend": self.args.backend, **self.state}
        if not preflight:
            # astra_ws: setup_ready/detail describe the suspended setup check against the ready pose;
            # once SONIC balances on its own they stay false and GPT read them as "controller not ready".
            state.pop("setup_ready", None)
            state.pop("detail", None)
        state["previous_action"] = getattr(self, "previous_action", "hold")
        result = queue.Queue(maxsize=1)
        def request():
            try:
                result.put((self.policy.decide(self.args.task, frames, state), None))
            except Exception as error:
                result.put((None, error))
        # One request at a time. A hung socket cannot keep the controller lease alive.
        worker = threading.Thread(target=request, daemon=True)
        start = time.monotonic()
        worker.start()
        while result.empty():
            self.pump(camera_required=True)
            self.abort_key()
            if time.monotonic()-start > self.args.api_timeout + 1:
                raise TimeoutError("Vision policy timed out; loop paused, no queued command")
            time.sleep(.02)
        decision, error = result.get_nowait()
        if error:
            raise error
        self.log("preflight_response" if preflight else "decision", action=decision.action,
                 confidence=decision.confidence, reason=decision.reason,
                 latency_s=round(time.monotonic()-start, 3))
        return decision, observed

    def send(self, command, argument=0):
        if not self.session.command(command, argument):
            raise RuntimeError("Controller has an unacknowledged command")
        self.wait_for(lambda: not self.session.pending, 1)
        if self.state.get("result") != "ok":
            raise RuntimeError("Controller rejected command: " + self.state.get("result", "no result"))
        self.log("command_ack", command=command, argument=argument, sequence=self.session.seq)

    def run(self):
        self.prepare()
        self.check_camera()
        self.ask_policy(preflight=True)  # Vision, auth and schema check, before any body actuation.
        self.gate("SUSPENDED SETUP: G1 secured in gantry, feet OFF the ground, limbs clear. "
                  "ENTER starts body actuation and ramps to the ready reference. "
                  "Keep the physical stop available. Mock mode only imitates this stage.")
        self.pump(camera_required=True)
        self.session.arm()
        self.wait_for(lambda: self.state.get("setup_ready"), self.args.startup_timeout,
                      camera_required=True)
        self.log("suspended_checks_passed", detail=self.state.get("detail"))
        self.gate("CHECKS PASSED: inspect the supported posture. ENTER authorizes the LOWERING stage; "
                  "autonomous motion remains disabled.")
        self.gate("LOWERING: lower the gantry until BOTH FEET support the robot. "
                  "In MuJoCo, focus the viewer and press 9 to release suspension. "
                  "ENTER confirms feet down and starts SONIC + the vision policy. Contact is operator-confirmed.")
        self.pump(camera_required=True)
        if not self.state.get("setup_ready"):
            raise RuntimeError("Joint settling no longer passes after lowering; re-support and restart")
        self.send("grounded_start")
        self.wait_for(lambda: self.state.get("state") == 2, 5, camera_required=True)
        if self.state.get("index") != 0 or self.state.get("playing"):
            raise RuntimeError("Expected SONIC to hold the initial standing reference before GPT control")
        self.log("standing_reference_active")
        if self.args.backend == "sim":
            atomic_write(self.directory / "task-start", "start\n")
        print("Loop active. Ctrl+C or q + ENTER requests latched damping; gantry must catch the robot.",
              flush=True)
        for _ in range(self.args.max_decisions):
            self.wait_for(lambda: not self.state.get("playing"), 15, camera_required=True)
            decision, observed = self.ask_policy()
            # Freshness is checked again AFTER the network response, not only at capture.
            self.pump(camera_required=True)
            allowed, reason = decision_allowed(decision, observed, time.monotonic(), self.state,
                                               self.args.max_decision_age, self.args.min_confidence)
            if decision.action == "stop":
                self.log("policy_pause", reason=decision.reason)
                break
            if allowed and decision.action != "hold":
                self.send("motion", self.indexes[decision.action])
                self.previous_action = decision.action
            else:
                self.log("hold", reason=reason if not allowed else decision.reason)
            # No speculative action queue: finish the acknowledged clip before a new observation.
            if allowed and decision.action not in ("hold", "stop"):
                expected = next(c["frames"] for c in self.catalog if c["index"] == self.indexes[decision.action])
                self.wait_for(lambda: not self.state.get("playing") and
                              self.state.get("frame", 0) >= expected-1,
                              expected/50 + 5, camera_required=True)
            else:
                until = time.monotonic() + .25
                self.wait_for(lambda: time.monotonic() >= until, 2, camera_required=True)
        self.gate("Session finished/paused. SONIC continues holding ready. "
                  "Support the robot fully in the gantry, then ENTER to stop the controller.")
        self.support_confirmed = True
        self.session.stop()
        time.sleep(.1)
        self.session.terminate_supported()
        self.log("finished")

    def cleanup(self):
        if self.session and self.session.process and self.session.process.poll() is None:
            self.session.stop()  # Native writer independently latches body + hand damping.
            if self.args.backend == "robot" and not self.support_confirmed:
                print("DAMPING REQUESTED. Support the robot in the gantry. "
                      "The controller stays alive until you confirm support.", flush=True)
                try:
                    self.gate("Robot FULLY SUPPORTED? ENTER permits controller shutdown.", tolerate_fault=True)
                except (KeyboardInterrupt, Exception):
                    print(f"Controller left running in latched damping (PID {self.session.process.pid}). "
                          "Do not kill it until the robot is supported.", flush=True)
                else:
                    self.support_confirmed = True
                    self.session.terminate_supported()
            else:
                self.session.terminate_supported()
        if self.session and self.session.process:
            try:
                self.session.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # Never force-kill a controller with an unsupported robot.
        if self.simulator and self.simulator.poll() is None:
            os.killpg(self.simulator.pid, signal.SIGTERM)
            try:
                self.simulator.wait(timeout=15)  # Let the simulator finalize the MP4 and CSV.
            except subprocess.TimeoutExpired:
                print("Simulator is still finalizing its recording; see simulator.log.", flush=True)
        if self.camera:
            self.camera.close()
        if self.preview:
            self.preview.close()
        if self.events:
            self.events.close()
        if self.sim_log:
            self.sim_log.close()
        if self.lock:
            self.lock.close()  # Controller inherits the lock until it exits.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mock", "sim", "robot"), default="mock")
    p.add_argument("--policy", choices=("astra", "fake"), default="astra",
                   help="astra: OpenAI API using the selected model; fake: offline mock only")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-6-astra"),
                   help="OpenAI model ID; overrides OPENAI_MODEL (default: gpt-6-astra)")
    p.add_argument("--task-name", default="catch_balloon", help="Task directory under gpt_astra/tasks")
    p.add_argument("--task", help="Override the selected task's text prompt")
    p.add_argument("--interface", help="Robot: wired Ethernet interface. Simulation: lo.")
    p.add_argument("--camera-host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--camera-key", default="ego_view")
    p.add_argument("--camera-timeout", type=float, default=30, help="Seconds to wait for initial camera images")
    p.add_argument("--preview", action="store_true", help="Show camera window in mock mode; always enabled for sim/robot")
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "GR00T-WholeBodyControl")
    p.add_argument("--build", type=Path, default=DEPLOY)
    p.add_argument("--library", type=Path, help="Custom library with ready/ and actions.json")
    p.add_argument("--prepare-only", type=Path, metavar="NEW_FOLDER", help="Generate/validate default motion library; no API/robot")
    p.add_argument("--image", type=Path, help="Mock only: use this real image instead of synthetic frames")
    p.add_argument("--auto-confirm", action="store_true", help="Mock only: automate operator gates")
    p.add_argument("--max-decisions", type=int, default=20)
    p.add_argument("--api-timeout", type=float, default=15)
    p.add_argument("--max-decision-age", type=float, default=2, help="Maximum seconds from latest input image to command")
    p.add_argument("--min-confidence", type=float, default=.8)
    p.add_argument("--startup-timeout", type=float, default=300)
    a = p.parse_args(argv)
    a.model = a.model.strip()
    if not a.model or any(c.isspace() for c in a.model):
        p.error("--model / OPENAI_MODEL must be a nonempty model ID without whitespace")
    a.upstream, a.build = a.upstream.expanduser().resolve(), a.build.expanduser().resolve()
    if a.library:
        a.library = a.library.expanduser().resolve()
    for name in ("api_timeout", "max_decision_age", "startup_timeout", "camera_timeout"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 1 <= a.max_decisions <= 1000 or not 0 <= a.min_confidence <= 1:
        p.error("Use 1..1000 decisions and confidence in [0, 1]")
    if not 1 <= a.camera_port <= 65535:
        p.error("Invalid camera port")
    if a.backend != "mock" and (a.auto_confirm or a.policy == "fake" or a.image):
        p.error("--auto-confirm, --policy fake and --image are only permitted with --backend mock")
    if a.backend == "robot" and (not a.interface or a.interface == "lo"):
        p.error("--backend robot requires an explicit non-loopback --interface")
    if a.backend == "sim":
        if a.interface not in (None, "lo") or a.camera_host not in ("localhost", "127.0.0.1"):
            p.error("Simulation requires loopback interface and local camera")
        a.interface = "lo"
    if not a.auto_confirm and not a.prepare_only and not sys.stdin.isatty():
        p.error("Run in an interactive terminal; unattended gates only allowed with --backend mock --auto-confirm")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.prepare_only:
        from sonic_laptop.model import load_model
        generator = task_generator(task_directory(args.task_name)) or generate_standing_library
        generator(args.prepare_only, load_model(args.upstream))
        print(f"Kinematic references validated at {args.prepare_only}; physical performance is untested.")
        return 0
    runner = Runner(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner.run()
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Operator interrupted; ending decisions.", flush=True)
        return 130
    except Exception as error:
        print(f"Stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
