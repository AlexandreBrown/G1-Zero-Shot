"""A light red sphere approaches the robot; reset the target between attempts."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import xml.etree.ElementTree as ET


def prepare_scene(source_scene, run_directory):
    source_scene, run_directory = Path(source_scene), Path(run_directory)
    scene = ET.parse(source_scene)
    include = scene.getroot().find("include")
    if include is None:
        raise ValueError("Expected robot include in the upstream MuJoCo scene")
    robot_path = source_scene.parent / include.attrib["file"]
    robot = ET.parse(robot_path)
    compiler = robot.getroot().find("compiler")
    if compiler is None:
        raise ValueError("Expected compiler settings in the upstream robot XML")
    compiler.set("meshdir", str((robot_path.parent / compiler.get("meshdir", ".")).resolve()))
    staged_robot = run_directory / "sim_robot.xml"
    robot.write(staged_robot, encoding="unicode")
    include.set("file", str(staged_robot.resolve()))
    world = scene.getroot().find("worldbody")
    if world is None:
        raise ValueError("Expected worldbody in the upstream MuJoCo scene")
    # The upstream scene marks the origin with a red sphere site; hide it so it never passes for the balloon.
    for site in world.iter("site"):
        if site.get("name") == "com_marker":
            site.set("rgba", "0 0 0 0")
    body = ET.SubElement(world, "body", name="target_balloon", pos="100 100 5")
    ET.SubElement(body, "freejoint", name="target_balloon_joint")
    ET.SubElement(body, "geom", name="target_balloon_geom", type="sphere", size="0.075",
                  mass="0.02", rgba="1 0.015 0.015 1", friction="0.5 0.005 0.0001")
    output = run_directory / "sim_scene.xml"
    scene.write(output, encoding="unicode")
    return output


HELD = os.environ.get("GPT_ASTRA_BALLOON") == "held"
HOVER = os.environ.get("GPT_ASTRA_BALLOON") == "hover" or HELD
# Sink mode (astra_ws): the default drop, but with balloon-like buoyancy and quadratic air drag. A free-falling
# sphere crosses the hands in ~0.3 s while a reach clip needs ~1.3 s to raise the arms, so the default drop
# cannot be touched by any policy. Lift cancels 90% of the weight; drag makes the terminal speed SINK_SPEED.
SINK = os.environ.get("GPT_ASTRA_BALLOON") == "sink"
SINK_LIFT, SINK_SPEED = .9, .35  # weight fraction, m/s
# Hover mode (astra_ws): the balloon floats at reach height in front of the robot instead of dropping
# from above the camera's view, because a free fall lasts ~0.5 s and a GPT decision takes 2-3 s.
# Points are pelvis-frame offsets near the palms of reach_left / reach_right / reach_both.
HOVER_POINTS = ((.45, .17, .22), (.45, -.17, .22), (.45, 0., .22))
HOVER_SECONDS = 20.0
HOVER_STIFFNESS, HOVER_DAMPING = 2.0, .3  # N/m, N*s/m: a balloon on a string a hand can push away
# Held mode (astra_ws): someone holds the balloon still at one fixed height (z = 0.16 m, the palm height of
# the reach clips' end pose, hands at x .44, y +-.17) and moves it to the next point ~2 s after a touch.
# Points (pelvis frame): on, outside, beyond and between the reach_left/right end positions, and centered.
HELD_POINTS = ((.44, .17), (.44, -.17), (.44, 0.), (.40, -.22), (.40, .22),
               (.44, -.08), (.48, .12), (.48, -.12), (.44, .08))
# The holder takes the balloon away 1 s after a touch and shows the next point only once the controller
# reports no clip playing, so a new point never meets an arm that is still moving.
HELD_Z, HELD_AFTER_TOUCH, HELD_GAP = .16, 1.0, .3
# GPT_ASTRA_HELD_SEED=<int>: shuffle the point order and move every spawn by up to +-2 cm in x and y (z fixed).
HELD_SEED = os.environ.get("GPT_ASTRA_HELD_SEED")
if HELD:
    HOVER_POINTS = tuple((x, y, HELD_Z) for x, y in HELD_POINTS)
    HOVER_SECONDS = 12.0
    HOVER_STIFFNESS, HOVER_DAMPING = 40.0, 2.5  # a hand holding it: a touch nudges it a few cm


class TaskSimulation:
    def __init__(self, model, data, start_file):
        import mujoco
        self.mujoco, self.model, self.data = mujoco, model, data
        self.start_file = Path(start_file)
        self.pause_file = self.start_file.parent / "world-paused"
        self.pause_ack = self.start_file.parent / "world-paused.ack"
        self.status_file = self.start_file.parent / "status"
        self.frozen = None
        self.paused_time = 0.0
        self.events_file = self.start_file.parent / "task_events.jsonl"
        self.body = model.body("target_balloon").id
        self.geom = model.geom("target_balloon_geom").id
        self.floor = model.geom("floor").id
        joint = model.joint("target_balloon_joint").id
        self.q = int(model.jnt_qposadr[joint])
        self.v = int(model.jnt_dofadr[joint])
        self.rng = random.Random(4)
        self.points, self.jitter = HOVER_POINTS, None
        if HELD and HELD_SEED is not None:
            self.jitter = random.Random(int(HELD_SEED))
            self.points = self.jitter.sample(HOVER_POINTS, len(HOVER_POINTS))
        self.started_at = None
        self.next_spawn = .05
        self.active = False
        self.contact_since = self.caught_at = None
        self.attempt = 0
        self.anchor = None
        self.spawned_at = 0.0
        self.touched = False
        self.mass = float(model.body_mass[self.body])

    def clip_playing(self):
        """Guard status line: seq state index frame play planner fault setup_ready (absent before SONIC starts).

        In planner mode play stays 1 while the robot stands, so only a reference clip counts as playing.
        """
        if not self.status_file.exists():
            return False
        fields = self.status_file.read_text().split()
        return fields[4] == "1" and fields[5] == "0"

    def event(self, event, elapsed, **details):
        item = {"event": event, "sim_elapsed_s": round(elapsed, 3), "attempt": self.attempt,
                "wall_time": time.time(), **details}
        with self.events_file.open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(f"[task] {event}: attempt {self.attempt} at {elapsed:.1f}s", flush=True)

    def step(self):
        data, model = self.data, self.model
        if self.started_at is None and self.start_file.is_file():
            self.started_at = float(data.time)
        # run.py --pause-world: hold the balloon and the task clock while the policy decides.
        if self.pause_file.exists():
            if self.frozen is None:
                self.frozen = (data.qpos[self.q:self.q+7].copy(), data.qvel[self.v:self.v+6].copy())
                self.pause_ack.write_text("paused\n")
                print(f"[task] world paused at {float(data.time):.2f}s", flush=True)
            data.qpos[self.q:self.q+7] = self.frozen[0]
            data.qvel[self.v:self.v+6] = 0
            data.xfrc_applied[self.body] = [0, 0, -self.mass * model.opt.gravity[2], 0, 0, 0]
            self.paused_time += model.opt.timestep
            return
        if self.frozen is not None:
            data.qvel[self.v:self.v+6] = self.frozen[1]
            data.xfrc_applied[self.body] = 0
            self.frozen = None
            self.pause_ack.unlink()
            print(f"[task] world resumed at {float(data.time):.2f}s", flush=True)
        elapsed = None if self.started_at is None else max(0, float(data.time)-self.started_at-self.paused_time)
        if self.active:
            touching_floor = False
            hands = set()
            for contact in data.contact[:data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                if self.geom not in (a, b):
                    continue
                other = b if a == self.geom else a
                if other == self.floor:
                    touching_floor = True
                name = self.mujoco.mj_id2name(model, self.mujoco.mjtObj.mjOBJ_BODY,
                                             int(model.geom_bodyid[other])) or ""
                if name.startswith(("left_hand", "left_wrist")):
                    hands.add("left")
                if name.startswith(("right_hand", "right_wrist")):
                    hands.add("right")
            if HOVER and hands and not self.touched:
                self.touched, self.touched_at = True, elapsed
                self.event("touch_" + "_".join(sorted(hands)), elapsed)
            if len(hands) == 2 and float((data.qvel[self.v:self.v+3]**2).sum()) < .16:
                if self.contact_since is None:
                    self.contact_since = elapsed
                if self.caught_at is None and elapsed-self.contact_since >= .3:
                    self.caught_at = elapsed
                    self.event("caught", elapsed)
            else:
                self.contact_since = None
            if touching_floor or data.qpos[self.q+2] < .07:
                self.event("ground", elapsed)
                self.active = False
            elif self.caught_at is not None and elapsed-self.caught_at >= 5:
                self.event("reset_after_catch", elapsed)
                self.active = False
            elif HELD and self.touched and elapsed - self.touched_at >= HELD_AFTER_TOUCH:
                self.event("held_next", elapsed)
                self.active = False
            elif HOVER and elapsed - self.spawned_at >= HOVER_SECONDS:
                self.event("hover_timeout", elapsed)
                self.active = False
            if not self.active:
                self.next_spawn = elapsed + (HELD_GAP if HELD else self.rng.uniform(1.0, 2.0))
        if not self.active:
            data.qpos[self.q:self.q+7] = [100, 100, 5, 1, 0, 0, 0]
            data.qvel[self.v:self.v+6] = 0
            data.xfrc_applied[self.body] = 0
        if elapsed is not None and not self.active and elapsed >= self.next_spawn and not (HELD and self.clip_playing()):
            pelvis = data.body("pelvis")
            rotation = pelvis.xmat.reshape(3, 3)
            if HOVER:
                offset = list(self.points[self.attempt % len(self.points)])
                if self.jitter:
                    offset[:2] = [c + self.jitter.uniform(-.02, .02) for c in offset[:2]]
                if not HELD:
                    offset = [c + self.rng.uniform(-.03, .03) for c in offset]
                position = pelvis.xpos + rotation @ offset
                self.anchor = position.copy()
                velocity = [0, 0, 0]
            else:
                position = pelvis.xpos + rotation @ [self.rng.uniform(.42, .58),
                                                    self.rng.uniform(-.14, .14),
                                                    self.rng.uniform(.72, .80)]
                # Alternate between dropping and a gentle throw toward the robot.
                velocity = rotation @ ([-.12, 0, 0] if self.attempt % 2 else [0, 0, 0])
            data.qpos[self.q:self.q+7] = [*position, 1, 0, 0, 0]
            data.qvel[self.v:self.v+6] = [*velocity, 0, 0, 0]
            self.active = True
            self.contact_since = self.caught_at = None
            self.spawned_at, self.touched = elapsed, False
            self.attempt += 1
            self.event("spawn", elapsed, **({"offset": [round(c, 3) for c in offset]} if HOVER else {}))
        if HOVER and self.active:
            position, velocity = data.qpos[self.q:self.q+3], data.qvel[self.v:self.v+3]
            force = HOVER_STIFFNESS * (self.anchor - position) - HOVER_DAMPING * velocity
            force[2] -= self.mass * model.opt.gravity[2]
            data.xfrc_applied[self.body] = [*force, 0, 0, 0]
        if SINK and self.active:
            velocity = data.qvel[self.v:self.v+3]
            weight = -self.mass * model.opt.gravity[2]
            drag = (1 - SINK_LIFT) * weight / SINK_SPEED**2
            force = -drag * float(np.linalg.norm(velocity)) * velocity
            force[2] += SINK_LIFT * weight
            data.xfrc_applied[self.body] = [*force, 0, 0, 0]
        # The active sphere is a free MuJoCo body: gravity and contact determine
        # its motion. By default no artificial lift, position interpolation or drag is applied.
