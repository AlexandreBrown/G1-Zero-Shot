"""Local 20 Hz follower for the follow_balloon task: finds the red balloon in the ego image and steers
SONIC's slow walk (forward/back and side-steps, no turning) so the balloon stays at grab distance. Every measured
number comes from a platform calibration (gpt_astra.calibration); the defaults are the simulation's."""
from dataclasses import dataclass
import io

import numpy as np

from gpt_astra.calibration import SIM, Calibration

# follow_balloon grab poses. Shoulder roll 0.2 brings the hands together in front of the chest, about 15-18 cm
# apart, so they cradle a 15 cm balloon. Twisting the waist does the sideways placement: the hands' midpoint sits at
# x*sin(yaw_tracking*yaw). Forward/back, three reaches trade shoulder pitch against elbow to keep the hands at the
# same height; where each reach's hands meet is calibrated (Calibration.meeting_x). In simulation the region where
# both hands are within 3 cm of the balloon is ~12 cm deep but only ~4 cm wide, so the yaw grid is 0.04 rad (~1.6 cm
# at the hands). Leaning the torso did not work: SONIC kept it upright and only the arms dropped.
REACHES = ((-.55, -.5), (-.8, -.15), (-1.05, .2))  # shoulder pitch, elbow change
# Reaches used for grabs. Reach 0 (most bent elbows) often held the hands too far apart to cradle anything (no
# cradle region in 4 of 20 Mila/laptop grabs); reach 2's region is the widest sideways (9-14 cm).
GRAB_REACHES = (1, 2)
YAWS = tuple(round(.04 * k, 2) for k in range(-10, 11))
GRAB_ROLL = .2
# What the policy chooses between in follow mode; the 20 Hz loop carries the chosen intent out.
INTENTS = {
    "catch": "Go after the balloon and keep it: follow it while the person moves it, close both arms around it "
             "as soon as it is still and within reach, hold it while it stays in the hands, let go the moment "
             "the person pulls it out, then follow and grab it again. Choose it as soon as the balloon is in "
             "view, moving or not: the local controller handles all of the timing.",
    "follow": "Follow the balloon at a comfortable distance without grabbing it (let go first if holding).",
    "release": "Let go now (open the arms) and keep following without grabbing until catch is chosen again.",
}


def grab_x(cal: Calibration = SIM) -> float:
    return cal.meeting_x[1]


def approach_x(cal: Calibration = SIM) -> float:
    """Where the follower puts the balloon before a grab: midway between the grab reaches' meeting points, closer by
    the distance the robot settles back when it leaves the planner."""
    return (cal.meeting_x[GRAB_REACHES[0]] + cal.meeting_x[GRAB_REACHES[-1]]) / 2 - cal.settle_back


def grab_variants(cal: Calibration = SIM) -> list[tuple[str, float, float, float, float, float]]:
    """(suffix, waist yaw, shoulder pitch, elbow change, meeting x, meeting y) for every grab/release pair."""
    return [(f"{i}_{j}", yaw, shoulder, elbow, cal.meeting_x[i] * np.cos(cal.yaw_tracking * yaw),
             cal.meeting_x[i] * np.sin(cal.yaw_tracking * yaw))
            for i, (shoulder, elbow) in enumerate(REACHES) if i in GRAB_REACHES for j, yaw in enumerate(YAWS)]


def nearest_grab(ball: np.ndarray | None, cal: Calibration = SIM) -> str | None:
    """Suffix of the grab whose hands meet closest to the balloon, if both hands can squeeze it from there. The
    longest reach goes first: SONIC holds it most consistently (cradle regions 2-3x larger than reach 1's, which
    once shrank to 2 cm sideways and missed a balloon its model put dead centre)."""
    if ball is None:
        return None
    def miss(v):
        return max(abs(ball[0] - v[4]) / cal.meet_tolerance_x, abs(ball[1] - v[5]) / cal.meet_tolerance_y)
    fitting = [v for v in grab_variants(cal) if miss(v) <= 1]
    return min(fitting, key=lambda v: (-int(v[0].split("_")[0]), miss(v)))[0] if fitting else None


def coverage_gaps(cal: Calibration = SIM) -> list[tuple[float, float]]:
    """Balloon positions the follower accepts for a grab (approach zone, after settling back) that no grab pose
    reaches: must be empty, or the robot would close its arms beside the balloon."""
    zone_x = approach_x(cal) + cal.settle_back
    return [(round(float(x), 3), round(float(y), 3))
            for x in np.linspace(zone_x - cal.grab_tolerance_x, zone_x + cal.grab_tolerance_x, 9)
            for y in np.linspace(-cal.grab_tolerance_y, cal.grab_tolerance_y, 9)
            if nearest_grab(np.array([x, y, .16]), cal) is None]


@dataclass
class Blob:
    """The red blob's fitted circle in pixels; residual is the outline's RMS distance from it over its radius;
    coverage is the share of the circle's in-image area that is red."""
    u: float
    v: float
    radius: float
    residual: float
    pixels: int
    width: int
    height: int
    coverage: float


def fit_blob(jpeg: bytes, cal: Calibration = SIM) -> Blob | None:
    from PIL import Image  # imported here so the MuJoCo-only tools (grab_report) can use the grab model
    image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).astype(np.int16)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    mask = (red > cal.red_min) & (red > cal.red_ratio * green) & (red > cal.red_ratio * blue)
    pixels = int(mask.sum())
    if pixels < cal.min_pixels:
        return None
    # Close holes (a real balloon's specular highlight is not red) by filling each row and column between its first
    # and last red pixel: exact for a round blob; two separate red things merge into one shape that fits badly.
    mask = ((np.cumsum(mask, axis=1) > 0) & (np.cumsum(mask[:, ::-1], axis=1)[:, ::-1] > 0) &
            (np.cumsum(mask, axis=0) > 0) & (np.cumsum(mask[::-1], axis=0)[::-1] > 0))
    height, width = mask.shape
    # Fit a circle to the blob's outline away from the image border: a close balloon is cut by the bottom edge
    # (often less than half visible), which makes any width-based radius too small and the depth too large.
    inner = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    outline = mask.copy()
    outline[1:-1, 1:-1] &= ~inner
    outline[[0, -1], :] = outline[:, [0, -1]] = False
    rows, cols = np.nonzero(outline)
    if len(rows) < 12:
        return None
    a = np.column_stack([cols, rows, np.ones_like(cols)]).astype(float)
    (b0, b1, b2), *_ = np.linalg.lstsq(a, (cols ** 2 + rows ** 2).astype(float), rcond=None)
    u, v = b0 / 2, b1 / 2
    radius = float(np.sqrt(max(b2 + u * u + v * v, 1e-9)))
    residual = float(np.sqrt(np.mean((np.hypot(cols - u, rows - v) - radius) ** 2)) / radius)
    ys, xs = np.ogrid[:height, :width]
    area = int(((xs - u) ** 2 + (ys - v) ** 2 <= radius ** 2).sum())
    return Blob(float(u), float(v), radius, residual, pixels, width, height, pixels / max(area, 1))


def blob_position(blob: Blob, cal: Calibration = SIM) -> np.ndarray:
    """Balloon centre in the pelvis frame (m): the blob's direction, at the depth its apparent radius implies."""
    focal = blob.height / 2 / np.tan(np.radians(cal.fovy_deg) / 2)
    depth = focal * cal.balloon_radius / blob.radius
    ray = np.array([(blob.u - blob.width / 2) / focal, -(blob.v - blob.height / 2) / focal, -1.])
    return cal.camera_position + cal.camera_rotation @ (ray * depth)


# Held between the hands the balloon's outline is partly hidden: in simulation the fit residual reached 0.15 there
# (at most 0.03 in open view), so a held balloon is judged with this much looser circle test.
HELD_RESIDUAL_FACTOR = 3.
# A balloon fills its fitted circle (robot camera, 2026-11-04: a cord across it still leaves ~0.8 red); two red things
# merged into one fit do not (a balloon plus red labels on nearby equipment: 0.05). Held, the hands cover part of it.
MIN_COVERAGE, HELD_MIN_COVERAGE = .5, .25


def locate(jpeg: bytes, cal: Calibration = SIM, held: bool = False) -> np.ndarray | None:
    """Balloon centre in the pelvis frame (m) from one ego JPEG, or None when no plausible balloon is visible: a red
    blob whose outline is not a circle (clothing, a red bag; a red square fits with residual 0.11) or that would sit
    outside the height a held balloon has is not followed."""
    blob = fit_blob(jpeg, cal)
    if blob is None or blob.residual > cal.max_fit_residual * (HELD_RESIDUAL_FACTOR if held else 1) or \
            blob.coverage < (HELD_MIN_COVERAGE if held else MIN_COVERAGE):
        return None
    position = blob_position(blob, cal)
    return position if cal.height_range[0] <= position[2] <= cal.height_range[1] else None


@dataclass
class Step:
    """One follower command: a guarded move (0 fwd, 1 back, 2 left, 3 right; +FINE at the fine speed) or None to
    stand. reason says why a walk was stopped or refused by a safety limit (lost, max_walk, fence)."""
    direction: int | None
    in_grab_zone: bool
    reason: str | None = None


def in_grab_zone(ball: np.ndarray | None, cal: Calibration = SIM) -> bool:
    return bool(ball is not None and abs(ball[0] - approach_x(cal)) <= cal.grab_tolerance_x
                and abs(ball[1]) <= cal.grab_tolerance_y)


# The guarded planner's move takes directions 0-3 (forward, back, left, right) at the walking speed and 4-7 the
# same at the controller's fine speed. Timed stop-and-look pulses moved the robot anywhere from 0 to 0.35 m in the
# real-time sim (the stop replans to standing and either cancels or completes the step in flight), so the camera
# closes the loop instead: walk while the balloon is farther from the target than the walk coasts after the last
# request (Calibration.coast), slow down near it (forward/back; the grab poses' waist twist covers sideways
# errors), then stand until the view settles and look again. A fine walk that has not moved 2 cm in stall s has not
# started stepping (3 times in 25 at 0.2 m/s in sim): stand and start it again, which replans; only after two
# stalls in a row, and only when more than a full-speed coast remains, kick s at full speed starts the gait (a kick
# with 0.1-0.15 m left carried the robot 0.34 m on, into the balloon). A walk must move min_step before it may stop:
# a walk that starts within one coast of the target would otherwise stop on its first tick without a step, and 2 cm
# was only the lean of a gait that never stepped. Whatever the estimate says, no walk goes toward a balloon nearer
# than CLOSEST.
FINE = 4
UNIT = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))
CLOSEST = .3


class Follower:
    """Closed-loop walking on the camera estimate, one axis at a time (sideways first, which keeps the balloon in
    view). When the balloon leaves the view (beyond ~1 m it is above the image), it walks toward where it was last
    seen for cal.lost_walk s. Safety limits: no walk longer than cal.max_walk s, and none that takes the robot
    farther than cal.fence m from where it started. Distance is dead-reckoned at the commanded speeds (SONIC walks
    slower; short probes cannot bound its top speed, so nothing measured replaces them) plus each walk's coast after
    it stops, and every switch into or out of the planner (which shifts the robot by about cal.settle_back) adds that
    much uncertainty: together an upper bound on how far the robot can be."""
    def __init__(self, cal: Calibration = SIM):
        self.cal = cal
        self.position, self.slack, self.clock = np.zeros(2), 0., None
        self.reset()

    def planner_switch(self):
        """The robot left or re-entered the planner (grab, release): its position is that much less certain."""
        self.slack += self.cal.settle_back

    def reachable(self, ball: np.ndarray) -> bool:
        """A grab pose reaches the balloon from here (after the settle back of leaving the planner). A walk toward
        it must move min_step before it may stop, so a smaller correction would be forced past the target and coast
        on into the body: a reachable balloon just beyond the zone is grabbed instead."""
        return nearest_grab(np.array([ball[0] + self.cal.settle_back, ball[1], ball[2]]), self.cal) is not None

    def distance(self) -> float:
        """Upper bound on the distance from the start (m)."""
        return float(np.linalg.norm(self.position)) + self.slack

    def reset(self):
        """Forget the current walk and the last sighting (after a grab or release); keep the dead reckoning."""
        self.last, self.last_seen = None, -np.inf
        self.direction, self.settle_until, self.kick_until, self.stalls, self.sent = None, 0., 0., 0, False
        self.fine_since = self.fine_from = self.walk_from = self.walk_since = None

    def stop(self, now: float, reason: str | None = None) -> Step:
        if self.direction is not None:
            if self.sent:   # a walk refused before its first request never moved
                move = self.direction % FINE if now < self.kick_until else self.direction
                self.position += np.array(UNIT[self.direction % FINE]) * self.cal.coast[move]
            self.settle_until = now + (self.cal.settle_fine if self.direction >= FINE else self.cal.settle_fast)
            self.direction, self.kick_until, self.fine_since, self.sent = None, 0., None, False
        if reason:
            self.stalls = 0
        return Step(None, False, reason)

    def update(self, ball: np.ndarray | None, now: float, target_x: float | None = None,
               tolerance_x: float | None = None) -> Step:
        cal = self.cal
        target_x = approach_x(cal) if target_x is None else target_x
        tolerance_x = cal.grab_tolerance_x if tolerance_x is None else tolerance_x
        if self.direction is not None and self.sent and self.clock is not None:
            speed = cal.fine_walk_speed if self.direction >= FINE else cal.walk_speed
            self.position += np.array(UNIT[self.direction % FINE]) * speed * (now - self.clock)
        self.clock = now
        if ball is not None:
            self.last, self.last_seen = ball, now
        target = ball if ball is not None else self.last if now - self.last_seen < cal.lost_walk else None
        if target is None:
            return self.stop(now, "lost" if self.direction is not None else None)
        dx, dy = target[0] - target_x, target[1]
        if self.direction is None:
            if now < self.settle_until:
                return Step(None, False)
            if abs(dy) > cal.grab_tolerance_y:
                self.direction = 2 if dy > 0 else 3
            elif abs(dx) > tolerance_x and not (0 < dx <= cal.min_step and self.reachable(target)):
                self.direction = 0 if dx > 0 else 1
            else:
                return Step(None, ball is not None)
            if ball is None and self.direction < 2 and (dx, -dx)[self.direction] <= cal.fine_within:
                self.direction = None   # near the target, start only on a fresh sighting (it decides fine speed)
                return Step(None, False)
            self.walk_from, self.walk_since = (dx, -dx, dy, -dy)[self.direction], now
        axis = self.direction % FINE
        remaining = (dx, -dx, dy, -dy)[axis]
        kicking = now < self.kick_until
        if axis < 2 and self.direction < FINE and not kicking and remaining <= cal.fine_within and ball is not None:
            self.direction, self.fine_since, self.fine_from = axis + FINE, now, remaining
        elif self.direction >= FINE and now - self.fine_since > cal.stall and self.fine_from - remaining < .02:
            self.stalls += 1
            if self.stalls < 2 or remaining <= cal.coast[axis]:
                return self.stop(now)
            self.direction, self.kick_until, kicking, self.stalls = axis, now + cal.kick, True, 0
        if axis == 0 and target[0] < CLOSEST:
            return self.stop(now, "too_close")
        if remaining <= cal.coast[self.direction] and self.walk_from - remaining >= cal.min_step:
            self.stalls = 0
            return self.stop(now)
        if now - self.walk_since > cal.max_walk:
            return self.stop(now, "max_walk")
        ahead = self.position + (.05 + cal.coast[self.direction]) * np.array(UNIT[axis])
        if np.linalg.norm(ahead) + self.slack > cal.fence and np.linalg.norm(ahead) > np.linalg.norm(self.position):
            return self.stop(now, "fence")
        self.sent = True
        return Step(self.direction, False)


class Stillness:
    """The balloon counts as still once the robot itself stands and the estimate stays within tolerance (m;
    Calibration.still_tolerance, 2 cm in simulation) for 0.6 s."""
    def __init__(self, tolerance: float):
        self.history, self.tolerance = [], tolerance

    def update(self, ball: np.ndarray | None, now: float, walking: bool) -> bool:
        if walking or ball is None:
            self.history = []
            return False
        self.history = [(t, b) for t, b in self.history if now - t <= .8] + [(now, ball)]
        return (now - self.history[0][0] >= .6 and
                all(np.linalg.norm(b[:2] - ball[:2]) < self.tolerance for _, b in self.history))


def tracker_state(ball: np.ndarray | None, mode: str, meet: tuple[float, float] | None = None,
                  cal: Calibration = SIM) -> dict:
    """What the policy is told about the local estimate (pelvis frame, metres); meet is where the current grab's
    hands meet."""
    meet = (grab_x(cal), 0.) if meet is None else meet
    return {"mode": mode, "visible": ball is not None,
            "x_m": None if ball is None else round(float(ball[0]), 2),
            "y_m": None if ball is None else round(float(ball[1]), 2),
            "in_grab_zone": in_grab_zone(ball, cal),
            "between_hands": bool(ball is not None and np.hypot(ball[0] - meet[0], ball[1] - meet[1]) < .06)}
