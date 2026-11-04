"""static_balloon: a balloon fixed on a stand 0.9 m in front of the robot's pelvis, 0.15 m below it (the walk and grab
measurements of gpt_astra --measure; the walk probes start 0.9 m away). It never moves; a stiff spring holds it where it spawned."""
import json
from pathlib import Path

import numpy as np
from gpt_astra.tasks.catch_balloon.sim import prepare_scene  # noqa: F401

# 0.15 m below the pelvis the head camera sees the balloon from 0.35 to ~1.4 m (at hand height only to ~0.85 m).
AHEAD, HEIGHT = .9, -.15  # m in front of / above the pelvis at the start
STIFFNESS, DAMPING = 200.0, 10.0


class TaskSimulation:
    def __init__(self, model, data, start_file):
        self.model, self.data, self.start_file = model, data, Path(start_file)
        self.events_file = self.start_file.parent / "task_events.jsonl"
        self.body = model.body("target_balloon").id
        joint = model.joint("target_balloon_joint").id
        self.q, self.v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.mass = float(model.body_mass[self.body])
        self.anchor = None

    def step(self):
        data = self.data
        if self.anchor is None:
            data.qpos[self.q:self.q+7] = [100, 100, 5, 1, 0, 0, 0]
            data.qvel[self.v:self.v+6] = 0
            if not self.start_file.is_file():
                return
            pelvis = data.body("pelvis")
            heading = pelvis.xmat.reshape(3, 3)[:, 0]
            forward = np.array([heading[0], heading[1], 0.]) / np.linalg.norm(heading[:2])
            self.anchor = pelvis.xpos + AHEAD * forward
            self.anchor[2] = float(pelvis.xpos[2]) + HEIGHT
            data.qpos[self.q:self.q+7] = [*self.anchor, 1, 0, 0, 0]
            with self.events_file.open("a") as stream:
                stream.write(json.dumps({"event": "spawn", "sim_time_s": round(float(data.time), 3),
                                         "anchor": self.anchor.round(3).tolist()}) + "\n")
        position, velocity = data.qpos[self.q:self.q+3], data.qvel[self.v:self.v+3]
        force = STIFFNESS * (self.anchor - position) - DAMPING * velocity
        force[2] -= self.mass * self.model.opt.gravity[2]
        data.xfrc_applied[self.body] = [*force, 0, 0, 0]
