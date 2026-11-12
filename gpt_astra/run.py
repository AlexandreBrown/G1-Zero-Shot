#!/usr/bin/env python3
"""One entry point: images -> selected GPT model -> bounded motion names -> guarded SONIC."""
import argparse
from dataclasses import replace
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
import termios
import threading
import time
import tty
import uuid
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonic_laptop.paths import DEPLOY
from sonic_laptop.session import Session
from gpt_astra.camera import MockCamera, ZmqCamera
from gpt_astra.motions import RUNTIME, generate_standing_library, prepare_session
from gpt_astra import calibration
from gpt_astra.follow import (FINE, GRAB_REACHES, INTENTS, UNIT, Follower, Stillness, approach_x, grab_variants, grab_x,
                              locate, nearest_grab, tracker_state)
from gpt_astra.policy import Astra, ClaudeCode, FakePolicy, NullPolicy, RelayPolicy
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


# astra_ws walk primitives (--walk, simulation/mock): guarded planner direction (w, s, ',', '.') and seconds of
# slow walk (0.4 m/s request). Measured displacements are in the descriptions.
WALKS = {
    "walk_forward": (0, 1.2, "Walk straight forward about 0.3 m with the slow walk, then stand still. "
                             "Use when the target is visible ahead but beyond arm's reach."),
    "walk_backward": (1, 1.0, "Walk straight backward about 0.2 m, then stand still. Use when the target is "
                              "too close to the body for a reach."),
    "step_left": (2, 1.0, "Side-step to the ROBOT's left about 0.15 m, then stand still. Use when the target "
                          "is out of reach to the robot's left."),
    "step_right": (3, 1.0, "Side-step to the ROBOT's right about 0.15 m, then stand still. Use when the target "
                           "is out of reach to the robot's right."),
}


# --measure (task static_balloon). grab: these grab poses (both grab reaches, waist yaw -0.32..0.32 rad), each held
# GRAB_HOLD s. walk: each probe walks one move for its seconds toward/away from a balloon on a stand 0.9 m ahead,
# 0.15 m below the pelvis, three times; the order returns the robot near where it started.
GRAB_CHECK = tuple(f"{i}_{j}" for i in GRAB_REACHES for j in (2, 6, 10, 14, 18))
GRAB_HOLD = 2.0
WALK_PROBES = (("forward", 0, .6), ("back", 1, .6), ("forward_fine", 4, 1.5), ("back_fine", 5, 1.5),
               ("left", 2, 1.), ("right", 3, 1.))
WALK_REPEATS, WALK_SETTLE, STEADY = 3, 2., 1.  # repeats; s standing after a probe; s of frames per estimate
# A probe toward the balloon never ends nearer than TOO_CLOSE: it starts only if the commanded distance plus a
# generous coast (the robot's real coast is what it measures; sim: 0.25 m fast, 0.08 m fine) leaves that much, and it
# stops early when the balloon comes within TOO_CLOSE plus that coast, or is lost for PROBE_LOST s.
TOO_CLOSE, PROBE_COAST, PROBE_LOST = .3, {False: .3, True: .15}, .3
# Where each probe starts (balloon distance, m, by move axis): far for walks toward it, nearer for walks away so it
# stays in view (a balloon 0.15 m below the pelvis is in the head camera's view from 0.35 to ~1.4 m).
PROBE_START = {0: .95, 1: .7, 2: .8, 3: .8}
START_SETTLE = 4.5  # s after Start: the sim's gantry band fades out over 3 s and SONIC sways meanwhile
CAMERA_GRACE = 2.  # s of stale camera in follow mode before the loop pauses (the robot stands meanwhile)


def decision_allowed(decision, frame_received, now, state, max_age, confidence):
    """Never queue commands from old images or while the previous motion is running."""
    if state.get("fault") or state.get("state") != 2:
        return False, "controller not active"
    if state.get("playing") and not state.get("planner"):
        return False, "previous motion still active"
    if now-frame_received > max_age or now < frame_received:
        return False, "decision based on stale image"
    if decision.confidence < confidence:
        return False, "low confidence"
    return True, "accepted"


ABORT_KEYS = " qQ"
ABORT_HINT = "SPACE or q (no ENTER)"


def read_keys() -> str:
    """Every key typed since the last read ('' if none). During a run the terminal is in single-key mode, so a key
    arrives without ENTER; ENTER itself arrives as CR or LF."""
    keys = ""
    while select.select([sys.stdin], [], [], 0)[0]:
        chunk = os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
        if not chunk:
            raise EOFError("Input closed; controller is not automatically terminated")
        keys += chunk
    return keys


class Runner:
    def __init__(self, args):
        self.args = args
        self.directory = None
        self.session = self.camera = self.lock = self.events = self.simulator = self.sim_log = None
        self.preview = None
        self.camera_ready = False
        # Follow mode: a stale camera stops the walking (the robot stands) instead of raising into latched damping.
        self.camera_optional, self.camera_rejects = False, 0
        self.observation_index = 0
        self.calibration = calibration.load(args.calibration)
        self.last_preview = 0
        self.state = {}
        self.support_confirmed = False
        # Reconstruction log (trace.jsonl): every follower tick, controller mode change and, on the robot, every
        # camera frame (camera/); the robot's joints and SONIC's commands come from the recorder (robot_state*.csv).
        self.trace_stream = self.recorder = self.recorder_log = self.controller_seen = self.camera_logged = None
        self.record_camera, self.camera_count = args.backend == "robot", 0

    def trace(self, kind, **values):
        """High-rate line for reconstructing a run later: trace.jsonl, not printed."""
        if self.trace_stream:
            self.trace_stream.write(json.dumps({"time": time.time(), "event": kind, **values}) + "\n")

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
        self.trace_stream = (self.directory / "trace.jsonl").open("w", buffering=1)
        if self.record_camera:
            (self.directory / "camera").mkdir()
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
        if a.walk:
            actions.update({name: description for name, (_, _, description) in WALKS.items()})
        if (a.measure or a.follow) and any("grab_" + v[0] not in self.indexes for v in grab_variants(self.calibration)):
            raise ValueError("--follow/--measure need the follow_balloon grab poses: use --task-name follow_balloon "
                             "(follow) or static_balloon (measure)")
        if a.follow:  # the policy picks an intent; the runner picks grab poses and timing
            actions = {name: description for name, description in actions.items()
                       if not name.startswith(("grab_", "release_"))}
            actions.update(INTENTS)
        self.policy = (FakePolicy(actions) if a.policy == "fake" else NullPolicy() if a.policy == "none" else
                       RelayPolicy(actions, a.relay_dir, timeout=a.api_timeout) if a.policy == "relay" else
                       ClaudeCode(actions, model=a.claude_model, timeout=a.api_timeout) if a.policy == "claude" else
                       Astra(os.environ["OPENAI_API_KEY"], actions, model=a.model, timeout=a.api_timeout))
        self.camera = (MockCamera(a.image) if a.backend == "mock" else
                       ZmqCamera(a.camera_host, a.camera_port, a.camera_key, a.camera_swap_red_blue))
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
                 model={"astra": a.model, "claude": a.claude_model, "relay": "laptop relay (astra_ws/mila/claude_relay.py)"}.get(a.policy),
                 pause_world=a.pause_world, walk=a.walk, follow=a.follow,
                 task_name=a.task_name, task=a.task, max_decisions=a.max_decisions,
                 max_decision_age=a.max_decision_age)

    def pump(self, camera_required=False, tolerate_fault=False):
        if self.session and self.session.process:
            # Heartbeat is driven by this event loop, never by the blocked API worker.
            self.session.heartbeat()
            status = self.session.poll()
            if status:
                self.state = status
                mode = {key: status.get(key) for key in ("state", "index", "playing", "planner", "fault", "result",
                                                         "reason")}
                if mode != self.controller_seen:
                    self.controller_seen = mode
                    self.trace("controller", seq=status.get("seq"), frame=status.get("frame"), **mode)
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
            try:
                self.camera.poll(self.state)
            except ValueError as error:   # a frame with an old/missing capture time
                if not self.camera_optional:
                    raise
                self.camera_rejects += 1
                if self.camera_rejects in (1, 10, 100, 1000):
                    self.log("camera_frame_rejected", error=str(error), count=self.camera_rejects)
            if self.record_camera and self.camera.history and self.camera.history[-1].received != self.camera_logged:
                frame = self.camera.history[-1]
                self.camera_logged, self.camera_count = frame.received, self.camera_count + 1
                name = f"camera/{self.camera_count:06d}.jpg"
                (self.directory / name).write_bytes(frame.jpeg)
                self.trace("camera", image=name, received_monotonic=frame.received,
                           captured=getattr(self.camera, "last_stamp", None))
            if self.preview:
                self.preview.update(self.camera.history[-1] if self.camera.history else None)
            if (camera_required or self.camera_ready) and not self.camera_optional:
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
            self.gate("CAMERA CHECK: confirm the camera window shows the correct, updating view and a red balloon as "
                      "red (not blue). ENTER confirms the image before API preflight and body setup.")
            self.pump(camera_required=True)
            if not self.preview.ready:
                raise RuntimeError("Camera preview must be visible and fresh when confirming")
            self.log("camera_confirmed", camera=self.camera.history[-1].name)

    def gate(self, text, tolerate_fault=False):
        # Discard keys typed before the prompt so one ENTER cannot pass two gates (an abort key still aborts).
        if not self.args.auto_confirm and any(key in ABORT_KEYS for key in read_keys()):
            raise KeyboardInterrupt
        print("\n" + text + f"\nPress ENTER to confirm; {ABORT_HINT} to abort.", flush=True)
        if self.args.auto_confirm:
            self.log("mock_auto_confirmation", prompt=text)
            return
        while True:
            self.pump(tolerate_fault=tolerate_fault)
            if select.select([sys.stdin], [], [], .02)[0]:
                keys = read_keys()
                if any(key in ABORT_KEYS for key in keys):
                    raise KeyboardInterrupt
                if any(key in "\r\n" for key in keys):
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
        if not self.args.auto_confirm and any(key in ABORT_KEYS for key in read_keys()):
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
        paused = self.args.pause_world and not preflight
        if paused:
            # The sim task freezes the balloon and its clock while this file exists; the robot stands still
            # (no clip is playing), so the observed scene stays current until the decision arrives.
            atomic_write(self.directory / "world-paused", "paused\n")
            # Start the request only once the task confirms the freeze; a task without pause support fails here.
            try:
                self.wait_for(lambda: (self.directory / "world-paused.ack").exists(), 2, camera_required=True)
            except TimeoutError:
                raise RuntimeError("--pause-world: the task simulation did not confirm the pause") from None
        start = time.monotonic()
        worker.start()
        while result.empty():
            self.pump(camera_required=True)
            self.abort_key()
            if time.monotonic()-start > self.args.api_timeout + 1:
                raise TimeoutError("Vision policy timed out; loop paused, no queued command")
            time.sleep(.02)
        if paused:
            (self.directory / "world-paused").unlink()
            observed += time.monotonic() - start
        decision, error = result.get_nowait()
        if error:
            raise error
        self.log("preflight_response" if preflight else "decision", action=decision.action,
                 confidence=decision.confidence, reason=decision.reason,
                 latency_s=round(time.monotonic()-start, 3))
        return decision, observed

    def follow_loop(self):
        """follow_balloon. The policy, asked over and over on the latest images, picks the intent: catch, follow,
        release or stop (a reply takes ~4-15 s). The 20 Hz loop carries the intent out without waiting for it:
        walks after the balloon at the follow distance while it moves, closes in and grabs once it is still
        (catch), holds it, and lets go the moment it is pulled out of the hands. Walking is closed-loop on the camera
        (Follower) within the calibration's safety limits. A failed policy request is retried once; a second failure
        in a row pauses the loop with the robot standing (or holding), never in damping."""
        cal = self.calibration
        intent, mode, decisions, pending, errors = "follow", "follow", 0, None, 0
        follower, still, grab, meet = Follower(cal), Stillness(cal.still_tolerance), None, (grab_x(cal), 0.)
        approach_since = last_move = out_since = ball = walking = limit = None
        # In sim the gantry band is released up to 1 s after the Start handover and fades out over 3 s; walking
        # before it is gone tipped the robot (guard latch 0.4 s after entering the planner at 3.0 s).
        self.settle(START_SETTLE)
        self.enter_planner()
        self.camera_optional, camera_lost = True, None
        end = time.monotonic() + self.args.follow_seconds
        while time.monotonic() < end and decisions < self.args.max_decisions:
            self.pump(camera_required=True)
            self.abort_key()
            now = time.monotonic()
            frames = self.fresh_frames()
            if frames is None:
                camera_lost, ball = camera_lost or now, None
                if now - camera_lost > CAMERA_GRACE:
                    self.log("policy_pause", reason=f"camera stale for {CAMERA_GRACE:g} s")
                    self.camera_ready = False
                    break
            elif frames[-1].received != getattr(self, "located_frame", None):  # the camera runs at 10 Hz
                camera_lost = None
                self.located_frame, ball = frames[-1].received, locate(frames[-1].jpeg, cal, held=mode == "hold")
            is_still = still.update(ball, now, last_move is not None and now - last_move < .4)
            tracker = {**tracker_state(ball, mode, meet, cal), "intent": intent, "still": is_still}
            step = None
            if mode == "follow":
                # Close in only to catch a balloon that has stopped; otherwise keep a comfortable distance.
                if intent == "catch" and is_still and approach_since is None:
                    approach_since = now
                if intent != "catch" or ball is None and now - follower.last_seen > 1 or \
                        approach_since is not None and now - approach_since > 20:
                    approach_since = None
                step = follower.update(ball, now, *((approach_x(cal), cal.grab_tolerance_x)
                                                    if approach_since is not None
                                                    else (cal.follow_x, cal.follow_tolerance_x)))
                if step.direction != walking:
                    self.log("steer", direction=step.direction, tracker=tracker)
                    walking = step.direction
                if step.reason != limit:
                    if step.reason:
                        self.log("walk_limit", reason=step.reason, dead_reckoned_m=follower.position.round(2).tolist(),
                                 tracker=tracker)
                    limit = step.reason
                if step.direction is not None and not self.session.pending:
                    self.session.command("move", step.direction)  # the deadman stops the walk once these stop
                    last_move = now
                if approach_since is not None and step.in_grab_zone and is_still:
                    grab = self.catch()
                    follower.planner_switch()
                    if grab:
                        mode, meet = "hold", self.held_at(grab)
                    else:
                        follower.planner_switch()
                    still, approach_since, out_since, walking = Stillness(cal.still_tolerance), None, None, None
                    follower.reset()
            elif mode == "hold":
                # Let go as soon as the balloon has left the hands (or the policy asked to): no policy round trip.
                gone = frames is not None and (ball is None or np.hypot(ball[0] - meet[0], ball[1] - meet[1]) > .08)
                out_since = (out_since or now) if gone else None
                if intent != "catch" or out_since is not None and now - out_since >= (.6 if ball is None else .3):
                    self.log("let_go", reason="policy" if intent != "catch" else "pulled out of the hands")
                    self.play("release_" + grab)
                    self.enter_planner()
                    follower.planner_switch()
                    mode, still, meet, out_since = "follow", Stillness(cal.still_tolerance), (grab_x(cal), 0.), None
                    follower.reset()
                    if intent == "release":
                        intent = "follow"
            self.trace("follow", ball=None if ball is None else [round(float(v), 4) for v in ball], mode=mode,
                       intent=intent, still=is_still, approach=approach_since is not None, tracker=tracker,
                       step=None if step is None else {"direction": step.direction, "reason": step.reason,
                                                       "in_grab_zone": step.in_grab_zone},
                       dead_reckoned_m=follower.position.round(3).tolist())
            if pending is None and frames is not None:
                pending = self.request_decision(frames, tracker)
            if pending is not None and not pending.empty():
                decision, error = pending.get_nowait()
                pending = None
                if error:
                    errors += 1
                    self.log("policy_error", error=f"{type(error).__name__}: {error}"[:300], consecutive=errors)
                    if errors >= 2:
                        self.log("policy_pause", reason="policy unavailable: two failed requests in a row")
                        break
                    continue
                errors = 0
                decisions += 1
                self.log("decision", action=decision.action, confidence=decision.confidence,
                         reason=decision.reason, latency_s=round(time.monotonic() - self.request_started, 3),
                         tracker=tracker)
                if decision.action == "stop":
                    self.log("policy_pause", reason=decision.reason)
                    break
                if decision.confidence < self.args.min_confidence:
                    self.log("hold", reason="low confidence")
                elif decision.action in INTENTS:
                    intent = decision.action
            time.sleep(.05)
        if mode == "follow":
            self.stop_walking()

    def catch(self):
        """Stop, leave the planner and close the arms with the grab pose whose hands meet nearest the balloon."""
        self.stop_walking()
        self.leave_planner()
        ball = self.steady_ball()
        grab = nearest_grab(ball, self.calibration)
        self.log("catch", ball=None if ball is None else [round(float(v), 3) for v in ball], grab=grab)
        if not grab:
            self.log("hold", reason="grab skipped: robot settled away from the balloon")
            self.enter_planner()
            return None
        self.play("grab_" + grab)
        return grab

    def fresh_frames(self):
        """The latest camera pair, or None when the stream is stale."""
        try:
            return self.camera.latest()
        except TimeoutError:
            return None

    def steady_ball(self, seconds: float = .5, held: bool = False) -> np.ndarray | None:
        """Median balloon estimate over the next camera frames: one frame's sideways error (sd ~2 cm) is as large
        as the grab's sideways tolerance."""
        seen, last = [], None
        until = time.monotonic() + seconds
        while time.monotonic() < until or not seen and time.monotonic() < until + 1:
            self.pump(camera_required=True)
            frames = self.fresh_frames()
            frame = frames[-1] if frames else None
            if frame is not None and frame.received != last:
                last, ball = frame.received, locate(frame.jpeg, self.calibration, held=held)
                if ball is not None:
                    seen.append(ball)
            time.sleep(.03)
        return np.median(seen, axis=0) if seen else None

    def held_at(self, grab):
        """Where the camera sees the held balloon. The camera turns with the twisted torso, so "still in the
        hands" is judged against this view rather than the pelvis-frame meeting point."""
        held = self.steady_ball(held=True)
        variant = next(v for v in grab_variants(self.calibration) if v[0] == grab)
        return (variant[4], variant[5]) if held is None else (held[0], held[1])

    def request_decision(self, frames, tracker):
        """Start one policy request in the background; the result arrives on the returned queue."""
        result = queue.Queue(maxsize=1)
        state = {"preflight": False, "backend": self.args.backend, "tracker": tracker,
                 "previous_action": getattr(self, "previous_action", "hold")}
        self.ask_count = getattr(self, "ask_count", 0) + 1
        names = []
        for i, frame in enumerate(frames):
            name = f"frames/{self.ask_count:04d}_follow_{i}.jpg"
            (self.directory / name).parent.mkdir(exist_ok=True)
            (self.directory / name).write_bytes(frame.jpeg)
            names.append(name)
        self.log("observation", preflight=False, images=names, tracker=tracker,
                 previous_action=state["previous_action"])
        self.request_started = time.monotonic()
        def request():
            try:
                result.put((self.policy.decide(self.args.task, frames, state), None))
            except Exception as error:
                result.put((None, error))
        threading.Thread(target=request, daemon=True).start()
        return result

    def measure_grab(self):
        """--measure grab: play each GRAB_CHECK pose, hold it, release it. The joints are recorded by the sim
        (trajectory.csv) or, on the robot, by gpt_astra/tools/record_lowstate.py; gpt_astra/tools/grab_report.py
        turns the logged hold windows into the grab section of the calibration."""
        self.settle(START_SETTLE)
        for suffix in GRAB_CHECK:
            self.play("grab_" + suffix)
            start = time.time()
            self.settle(GRAB_HOLD)
            self.log("grab_hold", grab=suffix, start=start, end=time.time())
            self.play("release_" + suffix)

    def measure_walk(self):
        """--measure walk: from standing in the planner, walk one move for a fixed time and stop the way the follower
        does (the deadman, no explicit idle), with the balloon on its stand as the yardstick: steady estimates before,
        the last estimate at the stop, and steady estimates after the robot settled. Before every probe the robot
        walks back to a start spot for that move (closed loop, like the follower), and the balloon's apparent shift
        since the first look gives the robot's position for the fence. Then measure how far leaving the planner moves
        the robot. gpt_astra/tools/walk_report.py turns the log into the walk section."""
        # The stand's balloon is not a held one: any height; and never walk on a stale sighting for long.
        cal = self.calibration = replace(self.calibration, height_range=(-.6, .6),
                                         lost_walk=min(self.calibration.lost_walk, .3))
        self.settle(START_SETTLE)
        self.enter_planner()
        home = self.steady_ball(STEADY)
        if home is None:
            self.log("walk_probe_skipped", move="all", reason="balloon not in view at the start")
            return
        for repeat in range(WALK_REPEATS):
            for name, move, seconds in WALK_PROBES:
                fine, toward = move >= FINE, move % FINE == 0
                self.recenter(PROBE_START[move % FINE])
                before = self.steady_ball(STEADY)
                reach = (cal.fine_walk_speed if fine else cal.walk_speed) * seconds + PROBE_COAST[fine]
                if before is None or toward and before[0] - reach < TOO_CLOSE or \
                        np.linalg.norm(home[:2] - before[:2] + np.array(UNIT[move % FINE]) * reach) > cal.fence:
                    self.log("walk_probe_skipped", move=name, before=None if before is None else before.round(3).tolist())
                    continue
                at_stop, aborted, start = before, None, time.monotonic()
                seen = start
                while time.monotonic() - start < seconds:
                    self.pump(camera_required=True)
                    self.abort_key()
                    frame = self.camera.latest()[-1]
                    if frame.received != getattr(self, "probe_frame", None):
                        self.probe_frame, ball = frame.received, locate(frame.jpeg, cal)
                        if ball is not None:
                            at_stop, seen = ball, time.monotonic()
                        if toward and ball is not None and ball[0] < TOO_CLOSE + PROBE_COAST[fine]:
                            aborted = "balloon too close"
                            break
                    if toward and time.monotonic() - seen > PROBE_LOST:
                        aborted = "balloon lost"
                        break
                    if not self.session.pending:
                        self.session.command("move", move)  # the deadman stops the walk once these stop
                    time.sleep(.03)
                walked = time.monotonic() - start
                self.settle(WALK_SETTLE)
                after = self.steady_ball(STEADY)
                self.log("walk_probe", move=name, argument=move, repeat=repeat, seconds=round(walked, 2),
                         before=before.round(4).tolist(), at_stop=np.round(at_stop, 4).tolist(),
                         after=None if after is None else after.round(4).tolist(), aborted=aborted)
        self.recenter(PROBE_START[1])
        before = self.steady_ball(STEADY)
        self.leave_planner()
        self.settle(WALK_SETTLE)
        after = self.steady_ball(STEADY)
        self.log("settle_probe", before=None if before is None else before.round(4).tolist(),
                 after=None if after is None else after.round(4).tolist())

    def recenter(self, target_x: float, timeout: float = 20.) -> bool:
        """Walk the follower's closed loop until the balloon stands target_x (+-5 cm) ahead and within the
        follower's sideways tolerance (a side-step moves 15-25 cm: a tighter one only swings left and right). Gives
        up when the balloon has been out of sight for 3 s."""
        cal = self.calibration
        follower = Follower(cal)
        start = seen = time.monotonic()
        while time.monotonic() - start < timeout and time.monotonic() - seen < 3:
            self.pump(camera_required=True)
            self.abort_key()
            frames = self.fresh_frames()
            ball = locate(frames[-1].jpeg, cal) if frames else None
            seen = time.monotonic() if ball is not None else seen
            step = follower.update(ball, time.monotonic(), target_x, .05)
            if step.in_grab_zone:
                return True
            if step.direction is not None and not self.session.pending:
                self.session.command("move", step.direction)
            time.sleep(.05)
        self.log("recenter_failed", target_x=target_x)
        return False

    def enter_planner(self):
        if not self.state.get("planner"):
            self.send("planner")
            self.wait_for(lambda: self.state.get("planner") and self.state.get("playing"), 6, camera_required=True)
            self.settle(.3)

    def stop_walking(self):
        self.wait_for(lambda: not self.session.pending, 1, camera_required=True)
        self.send("idle")
        self.settle(.4)

    def play(self, name):
        index = self.indexes[name]
        self.send("motion", index)
        expected = next(c["frames"] for c in self.catalog if c["index"] == index)
        self.wait_for(lambda: not self.state.get("playing") and self.state.get("frame", 0) >= expected-1,
                      expected/50 + 5, camera_required=True)
        self.previous_action = name

    def walk(self, name):
        """Slow-walk one bounded step through the guarded planner, then stand still in planner mode."""
        direction, seconds, _ = WALKS[name]
        self.enter_planner()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            # The controller's deadman stops the walk 150 ms after the last fresh move request.
            if not self.session.pending:
                self.session.command("move", direction)
            self.pump(camera_required=True)
            if self.state.get("fault"):
                raise RuntimeError("Controller latched while walking: " + self.state.get("reason", ""))
            time.sleep(.03)
        self.wait_for(lambda: not self.session.pending, 1, camera_required=True)
        self.send("idle")
        self.settle(1.5)
        self.log("walk", action=name, seconds=seconds)

    def leave_planner(self):
        """Back to the reference clips through the controller's 1 s blend into the ready pose."""
        self.send("reference")
        self.wait_for(lambda: not self.state.get("planner") and not self.state.get("playing"), 4,
                      camera_required=True)
        self.settle(.2)

    def settle(self, seconds):
        until = time.monotonic() + seconds
        self.wait_for(lambda: time.monotonic() >= until, seconds + 1, camera_required=True)

    def send(self, command, argument=0):
        if not self.session.command(command, argument):
            raise RuntimeError("Controller has an unacknowledged command")
        self.wait_for(lambda: not self.session.pending, 1)
        if self.state.get("result") != "ok":
            raise RuntimeError("Controller rejected command: " + self.state.get("result", "no result"))
        self.log("command_ack", command=command, argument=argument, sequence=self.session.seq)

    def start_recorder(self):
        """Robot runs: record the robot's joints/IMU and every command SONIC sends (rt/lowstate, rt/lowcmd at 50 Hz,
        gpt_astra/tools/record_lowstate.py) into the run folder, from before actuation until the controller has
        stopped. Passive; a recorder that cannot start stops the run before anything moves."""
        if self.args.backend != "robot":
            return
        upstream = self.args.upstream
        self.recorder_log = (self.directory / "recorder.log").open("w")
        self.recorder = subprocess.Popen(
            [str(upstream / ".venv_sim/bin/python"), "-m", "gpt_astra.tools.record_lowstate", "--interface",
             self.args.interface, "--out", str(self.directory / "robot_state.csv"), "--seconds", "86400",
             "--parent", str(os.getpid())],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self.recorder_log, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": f"{upstream / 'external_dependencies/unitree_sdk2_python'}:{ROOT}"})
        until = time.monotonic() + 2   # keep reading the camera meanwhile: the preflight needs a fresh frame
        self.wait_for(lambda: time.monotonic() >= until or self.recorder.poll() is not None, 3, camera_required=True)
        if self.recorder.poll() is not None:
            raise RuntimeError(f"Robot state recorder exited; inspect {self.directory / 'recorder.log'}")
        self.log("recorder_started", files=["robot_state.csv", "robot_state_cmd.csv"])

    def run(self):
        self.prepare()
        self.check_camera()
        self.start_recorder()
        self.ask_policy(preflight=True)  # Vision, auth and schema check, before any body actuation.
        self.gate("SUSPENDED SETUP: G1 secured in gantry, feet OFF the ground, limbs clear. "
                  "ENTER starts body actuation and ramps to the ready reference. "
                  "Keep the physical stop available. Mock mode only imitates this stage.")
        self.pump(camera_required=True)
        self.session.arm()
        # astra_ws: in simulation the controller's model loading starves the runner's camera thread for >0.5 s
        # now and then; nothing moves on images during this suspended stage, so only the robot requires them.
        self.wait_for(lambda: self.state.get("setup_ready"), self.args.startup_timeout,
                      camera_required=self.args.backend != "sim")
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
        if self.args.follow:
            self.follow_loop()
        elif self.args.measure == "grab":
            self.measure_grab()
        elif self.args.measure == "walk":
            self.measure_walk()
        print(f"Loop active. Ctrl+C or {ABORT_HINT} requests latched damping; gantry must catch the robot.",
              flush=True)
        for _ in range(0 if self.args.follow or self.args.measure else self.args.max_decisions):
            self.wait_for(lambda: not self.state.get("playing") or self.state.get("planner"), 15,
                          camera_required=True)
            decision, observed = self.ask_policy()
            # Freshness is checked again AFTER the network response, not only at capture.
            self.pump(camera_required=True)
            allowed, reason = decision_allowed(decision, observed, time.monotonic(), self.state,
                                               self.args.max_decision_age, self.args.min_confidence)
            if decision.action == "stop":
                self.log("policy_pause", reason=decision.reason)
                break
            if allowed and decision.action in WALKS:
                self.walk(decision.action)
                self.previous_action = decision.action
            elif allowed and decision.action != "hold":
                if self.state.get("planner"):
                    self.leave_planner()
                self.send("motion", self.indexes[decision.action])
                self.previous_action = decision.action
            else:
                self.log("hold", reason=reason if not allowed else decision.reason)
            # No speculative action queue: finish the acknowledged clip before a new observation.
            if allowed and decision.action not in ("hold", "stop") and decision.action not in WALKS:
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
        if self.recorder:
            if self.recorder.poll() is not None:
                self.log("recorder_exited_early", returncode=self.recorder.returncode)
            else:
                self.recorder.send_signal(signal.SIGINT)  # it closes both CSVs and prints its sample counts
                try:
                    self.recorder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.recorder.kill()
            self.recorder_log.close()
        if self.camera:
            self.camera.close()
        if self.preview:
            self.preview.close()
        if self.trace_stream:
            self.trace_stream.close()
        if self.events:
            self.events.close()
        if self.sim_log:
            self.sim_log.close()
        if self.lock:
            self.lock.close()  # Controller inherits the lock until it exits.


def capabilities(build: Path) -> dict[str, str]:
    """The controller build's CAPABILITIES file: one capability per line, optionally with a value."""
    path = build / "CAPABILITIES"
    lines = path.read_text().splitlines() if path.is_file() else []
    return {line.split()[0]: " ".join(line.split()[1:]) for line in lines if line.strip()}


def follow_problems(backend, build, calibration_path, required=calibration.MEASURED_SECTIONS):
    """Why --follow (or --measure walk) may not run on this backend. The controller build must declare the commands
    the loop sends (reference, fine-move), planner-stance (its guard has been checked against SONIC's planner stance,
    which hangs the hands against the thighs; Alex's decision) and the walking speeds the calibration dead-reckons
    with. On the robot, the required calibration sections must be measured (dated)."""
    if not calibration_path.is_file():
        return [f"no calibration file {calibration_path}"]
    cal = calibration.load(calibration_path)
    problems = []
    if backend != "mock":
        declared = capabilities(build)
        missing = [name for name in ("reference", "fine-move", "planner-stance") if name not in declared]
        if missing:
            problems.append(f"controller build {build} does not declare {', '.join(missing)} (CAPABILITIES)")
        for name, speed in (("walk-speed", cal.walk_speed), ("fine-walk-speed", cal.fine_walk_speed)):
            try:
                matches = abs(float(declared.get(name, "nan")) - speed) < 1e-6
            except ValueError:
                matches = False
            if not matches:
                problems.append(f"controller {name} {declared.get(name)!r} differs from the calibration's {speed}")
    if backend == "robot":
        if cal.platform != "robot":
            problems.append(f"{calibration_path} is the {cal.platform} calibration, not the robot's")
        unmeasured = [name for name in cal.unmeasured() if name in required]
        if unmeasured:
            problems.append(f"calibration sections not measured on the robot: {', '.join(unmeasured)}")
    return problems


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mock", "sim", "robot"), default="mock")
    p.add_argument("--policy", choices=("astra", "claude", "fake", "none", "relay"), default="astra",
                   help="astra: OpenAI API using the selected model; claude: headless Claude Code on the "
                        "logged-in subscription (mock/sim only); fake: offline mock only")
    p.add_argument("--claude-model", default="sonnet", help="Claude Code model alias or ID for --policy claude")
    p.add_argument("--relay-dir", type=Path, help="--policy relay: shared folder that astra_ws/mila/claude_relay.py "
                                                   "serves (ClaudeCode on the laptop's login)")
    p.add_argument("--follow", action="store_true",
                   help="Task follow_balloon: the policy picks the intent (catch/follow/release/stop); a local 20 Hz "
                        "loop walks after the red balloon, grabs, holds and lets go. On the robot only with a "
                        "build that declares the reference and fine-move commands and a fully measured "
                        "calibration (gpt_astra/ROBOT_FOLLOW_CHECKLIST.md)")
    p.add_argument("--follow-seconds", type=float, default=240, help="Length of the --follow loop")
    p.add_argument("--measure", choices=("grab", "walk"),
                   help="Calibration measurements with --task-name static_balloon and --policy none "
                        "(gpt_astra/ROBOT_FOLLOW_CHECKLIST.md): grab plays the grab poses to measure where the hands "
                        "meet (on the robot, record joints with gpt_astra/tools/record_lowstate.py); walk takes "
                        "short scripted walks toward/away from a balloon on a stand to measure how the walk moves")
    p.add_argument("--calibration", type=Path,
                   help="Platform calibration for --follow (default: gpt_astra/calibration/robot.json for the robot, "
                        "sim.json otherwise)")
    p.add_argument("--walk", action="store_true",
                   help="Simulation/mock: also offer bounded slow-walk primitives through the guarded planner "
                        "(needs a controller build with the 'reference' command)")
    p.add_argument("--pause-world", action="store_true",
                   help="Simulation: freeze the task world (balloon, task clock) while the policy decides; "
                        "decision age then counts only unfrozen time")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-6-astra"),
                   help="OpenAI model ID; overrides OPENAI_MODEL (default: gpt-6-astra)")
    p.add_argument("--task-name", default="catch_balloon", help="Task directory under gpt_astra/tasks")
    p.add_argument("--task", help="Override the selected task's text prompt")
    p.add_argument("--interface", help="Robot: wired Ethernet interface. Simulation: lo.")
    p.add_argument("--camera-host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--camera-key", default="ego_view")
    p.add_argument("--camera-swap-red-blue", action=argparse.BooleanOptionalAction,
                   help="Swap red and blue in camera frames. Default: on for --backend robot, whose camera "
                        "server (GR00T composed_camera) JPEG-encodes RGB images with OpenCV, which assumes BGR, so "
                        "red arrives blue; off for sim, whose publisher already converts")
    p.add_argument("--camera-timeout", type=float, default=30, help="Seconds to wait for initial camera images")
    p.add_argument("--preview", action="store_true", help="Show camera window in mock mode; always enabled for sim/robot")
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "GR00T-WholeBodyControl")
    p.add_argument("--build", type=Path, default=DEPLOY)
    p.add_argument("--library", type=Path, help="Custom library with ready/ and actions.json")
    p.add_argument("--prepare-only", type=Path, metavar="NEW_FOLDER", help="Generate/validate default motion library; no API/robot")
    p.add_argument("--image", type=Path, help="Mock only: use this real image instead of synthetic frames")
    p.add_argument("--auto-confirm", action="store_true", help="Mock only: automate operator gates")
    p.add_argument("--max-decisions", type=int, default=20)
    p.add_argument("--api-timeout", type=float, help="Seconds per policy request (default: 15; 60 for --policy claude)")
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
    a.calibration = (a.calibration or calibration.for_backend(a.backend)).expanduser().resolve()
    if a.api_timeout is None:
        a.api_timeout = {"claude": 60., "relay": 90.}.get(a.policy, 15.)
    for name in ("api_timeout", "max_decision_age", "startup_timeout", "camera_timeout", "follow_seconds"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 1 <= a.max_decisions <= 1000 or not 0 <= a.min_confidence <= 1:
        p.error("Use 1..1000 decisions and confidence in [0, 1]")
    if a.camera_swap_red_blue is None:
        a.camera_swap_red_blue = a.backend == "robot"
    if not 1 <= a.camera_port <= 65535:
        p.error("Invalid camera port")
    if a.backend != "mock" and (a.auto_confirm or a.policy == "fake" or a.image):
        p.error("--auto-confirm, --policy fake and --image are only permitted with --backend mock")
    if a.policy == "relay" and (a.backend == "robot" or a.relay_dir is None):
        p.error("--policy relay is simulation/mock only and needs --relay-dir")
    if a.backend == "robot" and a.policy == "none" and not a.measure:
        p.error("--policy none is for simulation/mock smoke tests and --measure")
    if a.backend == "robot" and a.policy == "claude" and not a.follow:
        p.error("--policy claude takes 4-15 s per decision: on the robot only with --follow, whose 20 Hz loop "
                "does not wait for it")
    if a.follow and a.pause_world:
        p.error("--follow keeps the world running (no --pause-world)")
    if a.follow and a.measure:
        p.error("--follow and --measure are separate runs")
    if a.follow or a.measure == "walk":
        problems = follow_problems(a.backend, a.build, a.calibration,
                                   ("camera", "balloon") if a.measure == "walk" else calibration.MEASURED_SECTIONS)
        if problems:
            p.error(f"--{'follow' if a.follow else 'measure walk'} cannot run yet: " + "; ".join(problems))
    if a.walk and a.backend == "robot":
        p.error("--walk is simulation/mock only; the walk primitives are untested on hardware")
    if a.pause_world and a.backend != "sim":
        p.error("--pause-world only exists in simulation; the real world cannot be paused")
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
    # Single-key terminal: an abort key acts without ENTER; Ctrl+C still interrupts. Restored after the last gate.
    terminal = None if args.auto_confirm else termios.tcgetattr(sys.stdin)
    if terminal:
        tty.setcbreak(sys.stdin)
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
        if terminal:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal)


if __name__ == "__main__":
    raise SystemExit(main())
