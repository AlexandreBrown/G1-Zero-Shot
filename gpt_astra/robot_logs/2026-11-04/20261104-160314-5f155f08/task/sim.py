"""follow_balloon: a person walks the balloon slowly to a new place in front of the robot, waits there until
the robot grabs it, lets it hold the balloon for a while, then pulls it away to the next place."""
import json
from pathlib import Path
import time

import numpy as np
from gpt_astra.tasks.catch_balloon.sim import prepare_scene

# Places in the robot's pelvis frame when the move starts (x forward, y = robot's left), all in front of the
# robot so forward/back walking and side-steps reach them without turning. The height stays fixed in the
# world at the grab pose's palm height.
# Within ~0.85 m: farther, a balloon at hand height is above the downward-looking head camera's view.
PLACES = ((.8, .2), (.85, -.2), (.7, .3), (.85, 0.), (.75, -.3), (.8, .1))
START = (.7, 0.)
HEIGHT = .13  # above the pelvis at the start; the grab pose's hands meet ~.15 above the pelvis of a walked robot
SPEED = .08                       # m/s; the robot's slow walk covers ~0.3 m/s
GRABBED_AFTER, HOLD_FOR, GIVE_UP = 1.0, 4.0, 60.0  # s: cradled by both hands / robot may hold it / wait for a grab
CRADLE = .03  # m between each hand and the balloon's surface
STIFFNESS, DAMPING = 40.0, 2.5    # a hand holding it: the robot's hands can squeeze it, a pull frees it


class TaskSimulation:
    def __init__(self, model, data, start_file):
        import mujoco
        self.mujoco, self.model, self.data = mujoco, model, data
        self.start_file = Path(start_file)
        self.events_file = self.start_file.parent / "task_events.jsonl"
        self.body = model.body("target_balloon").id
        self.geom = model.geom("target_balloon_geom").id
        joint = model.joint("target_balloon_joint").id
        self.q, self.v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.mass = float(model.body_mass[self.body])
        self.phase, self.place, self.grabbed = "waiting_for_start", 0, False
        self.anchor = self.goal = None
        self.phase_since = self.contact_since = self.lost_since = self.hand_geoms = None

    def event(self, event, now, **details):
        item = {"event": event, "sim_time_s": round(now, 3), "place": self.place, "wall_time": time.time(),
                **details}
        with self.events_file.open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(f"[task] {event}: place {self.place} at {now:.1f}s", flush=True)

    def pelvis_point(self, x, y):
        pelvis = self.data.body("pelvis")
        heading = pelvis.xmat.reshape(3, 3)[:, 0]
        forward = np.array([heading[0], heading[1], 0.]) / np.linalg.norm(heading[:2])
        left = np.array([-forward[1], forward[0], 0.])
        point = pelvis.xpos + x * forward + y * left
        point[2] = self.height
        return point

    def both_hands_touching(self):
        """Cradled: each hand within CRADLE of the balloon's surface (the hands have no fingers to close)."""
        if self.hand_geoms is None:
            self.hand_geoms = {side: [g for g in range(self.model.ngeom) if self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])).startswith(
                (side + "_hand", side + "_wrist"))] for side in ("left", "right")}
        return all(min(self.mujoco.mj_geomDistance(self.model, self.data, self.geom, g, CRADLE, None)
                       for g in geoms) < CRADLE for geoms in self.hand_geoms.values())

    def start_move(self, now):
        x, y = PLACES[self.place % len(PLACES)]
        self.goal = self.pelvis_point(x, y)
        self.phase, self.phase_since = "moving", now
        self.event("move_start", now, goal_pelvis_frame=[x, y])

    def step(self):
        data, now = self.data, float(self.data.time)
        if self.phase == "waiting_for_start":
            data.qpos[self.q:self.q+7] = [100, 100, 5, 1, 0, 0, 0]
            data.qvel[self.v:self.v+6] = 0
            if not self.start_file.is_file():
                return
            self.height = float(data.body("pelvis").xpos[2]) + HEIGHT
            self.anchor = self.pelvis_point(*START)
            data.qpos[self.q:self.q+7] = [*self.anchor, 1, 0, 0, 0]
            self.event("spawn", now)
            self.start_move(now)
        # The hand-distance check is costly; 50 Hz is plenty for the 1 s and 0.5 s timers below.
        if self.phase in ("waiting", "held") and round(now / self.model.opt.timestep) % 10 == 0:
            self.grabbed = self.both_hands_touching()
        grabbed = self.grabbed
        if self.phase == "moving":
            offset = self.goal - self.anchor
            distance = float(np.linalg.norm(offset))
            stride = SPEED * self.model.opt.timestep
            if distance <= stride:
                self.anchor = self.goal.copy()
                self.phase, self.phase_since, self.contact_since = "waiting", now, None
                self.event("stopped", now)
            else:
                self.anchor = self.anchor + offset / distance * stride
        elif self.phase == "waiting":
            self.contact_since = (self.contact_since or now) if grabbed else None
            if self.contact_since is not None and now - self.contact_since >= GRABBED_AFTER:
                self.phase, self.phase_since, self.lost_since = "held", now, None
                self.event("grabbed", now, waited_s=round(now - self.contact_since, 2))
            elif now - self.phase_since >= GIVE_UP:
                self.event("grab_timeout", now)
                self.place += 1
                self.start_move(now)
        elif self.phase == "held":
            self.lost_since = None if grabbed else (self.lost_since or now)
            if self.lost_since is not None and now - self.lost_since >= .5:
                self.event("let_go_early", now, held_s=round(now - self.phase_since, 2))
                self.phase, self.phase_since, self.contact_since = "waiting", now, None
            elif now - self.phase_since >= HOLD_FOR:
                self.event("pull_away", now)
                self.place += 1
                self.start_move(now)
        position, velocity = data.qpos[self.q:self.q+3], data.qvel[self.v:self.v+3]
        force = STIFFNESS * (self.anchor - position) - DAMPING * velocity
        force[2] -= self.mass * self.model.opt.gravity[2]
        data.xfrc_applied[self.body] = [*force, 0, 0, 0]
