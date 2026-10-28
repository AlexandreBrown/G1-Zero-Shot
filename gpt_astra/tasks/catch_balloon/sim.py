"""A light red sphere approaches the robot; reset the target between attempts."""
import json
import os
from pathlib import Path
import random
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
    body = ET.SubElement(world, "body", name="target_balloon", pos="100 100 5")
    ET.SubElement(body, "freejoint", name="target_balloon_joint")
    ET.SubElement(body, "geom", name="target_balloon_geom", type="sphere", size="0.075",
                  mass="0.02", rgba="1 0.015 0.015 1", friction="0.5 0.005 0.0001")
    output = run_directory / "sim_scene.xml"
    scene.write(output, encoding="unicode")
    return output


HOVER = os.environ.get("GPT_ASTRA_BALLOON") == "hover"
# Hover mode (astra_ws): the balloon floats at reach height in front of the robot instead of dropping
# from above the camera's view, because a free fall lasts ~0.5 s and a GPT decision takes 2-3 s.
# Points are pelvis-frame offsets near the palms of reach_left / reach_right / reach_both.
HOVER_POINTS = ((.45, .17, .22), (.45, -.17, .22), (.45, 0., .22))
HOVER_SECONDS = 20.0
HOVER_STIFFNESS, HOVER_DAMPING = 2.0, .3  # N/m, N*s/m: a balloon on a string a hand can push away


class TaskSimulation:
    def __init__(self, model, data, start_file):
        import mujoco
        self.mujoco, self.model, self.data = mujoco, model, data
        self.start_file = Path(start_file)
        self.events_file = self.start_file.parent / "task_events.jsonl"
        self.body = model.body("target_balloon").id
        self.geom = model.geom("target_balloon_geom").id
        self.floor = model.geom("floor").id
        joint = model.joint("target_balloon_joint").id
        self.q = int(model.jnt_qposadr[joint])
        self.v = int(model.jnt_dofadr[joint])
        self.rng = random.Random(4)
        self.started_at = None
        self.next_spawn = .05
        self.active = False
        self.contact_since = self.caught_at = None
        self.attempt = 0
        self.anchor = None
        self.spawned_at = 0.0
        self.touched = False
        self.mass = float(model.body_mass[self.body])

    def event(self, event, elapsed):
        item = {"event": event, "sim_elapsed_s": round(elapsed, 3), "attempt": self.attempt}
        with self.events_file.open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(f"[task] {event}: attempt {self.attempt} at {elapsed:.1f}s", flush=True)

    def step(self):
        data, model = self.data, self.model
        if self.started_at is None and self.start_file.is_file():
            self.started_at = float(data.time)
        elapsed = None if self.started_at is None else max(0, float(data.time)-self.started_at)
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
                self.touched = True
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
            elif HOVER and elapsed - self.spawned_at >= HOVER_SECONDS:
                self.event("hover_timeout", elapsed)
                self.active = False
            if not self.active:
                self.next_spawn = elapsed + self.rng.uniform(1.0, 2.0)
        if not self.active:
            data.qpos[self.q:self.q+7] = [100, 100, 5, 1, 0, 0, 0]
            data.qvel[self.v:self.v+6] = 0
            data.xfrc_applied[self.body] = 0
        if elapsed is not None and not self.active and elapsed >= self.next_spawn:
            pelvis = data.body("pelvis")
            rotation = pelvis.xmat.reshape(3, 3)
            if HOVER:
                offset = [c + self.rng.uniform(-.03, .03) for c in HOVER_POINTS[self.attempt % len(HOVER_POINTS)]]
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
            self.event("spawn", elapsed)
        if HOVER and self.active:
            position, velocity = data.qpos[self.q:self.q+3], data.qvel[self.v:self.v+3]
            force = HOVER_STIFFNESS * (self.anchor - position) - HOVER_DAMPING * velocity
            force[2] -= self.mass * model.opt.gravity[2]
            data.xfrc_applied[self.body] = [*force, 0, 0, 0]
        # The active sphere is a free MuJoCo body: gravity and contact determine
        # its motion. No artificial lift, position interpolation or drag is applied.
