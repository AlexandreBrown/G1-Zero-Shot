"""Grab closes both arms around a balloon in front of the chest and HOLDS that pose; release opens back to
ready. One grab/release pair per place the hands can meet (gpt_astra.follow.grab_variants): the arms, not the
legs, do the last few centimetres. No finger commands; the hands squeeze the balloon between the palms."""
import json
from pathlib import Path

import numpy as np
from gpt_astra.follow import GRAB_ROLL, grab_variants
from gpt_astra.motions import ready_pose, write_clip


def grab_pose(model, yaw, shoulder, elbow):
    pose = ready_pose(model).copy()
    for side, sign in (("left", 1), ("right", -1)):
        pose[model["names"].index(f"{side}_shoulder_pitch_joint")] = shoulder
        pose[model["names"].index(f"{side}_shoulder_roll_joint")] = sign * GRAB_ROLL
        pose[model["names"].index(f"{side}_elbow_joint")] += elbow
    pose[model["names"].index("waist_yaw_joint")] = yaw
    return pose


def blend(a, b, count):
    t = np.linspace(0, 1, count)[:, None]
    return a + (b - a) * (10 * t**3 - 15 * t**4 + 6 * t**5)


def generate_library(destination, model):
    destination = Path(destination)
    destination.mkdir()
    ready = ready_pose(model)
    write_clip(destination / "ready", np.tile(ready, (50, 1)), model)
    descriptions = {}
    for suffix, yaw, shoulder, elbow, x, y in grab_variants():
        grab = grab_pose(model, yaw, shoulder, elbow)
        write_clip(destination / f"grab_{suffix}", np.concatenate(
            [np.tile(ready, (26, 1)), blend(ready, grab, 41)[1:], np.tile(grab, (26, 1))]), model)
        write_clip(destination / f"release_{suffix}", np.concatenate(
            [np.tile(grab, (26, 1)), blend(grab, ready, 41)[1:], np.tile(ready, (26, 1))]), model)
        descriptions[f"grab_{suffix}"] = f"Grab with the hands meeting at x {x:.2f} m, y {y:+.2f} m (pelvis frame)."
        descriptions[f"release_{suffix}"] = f"Release from grab_{suffix} back to ready."
    (destination / "actions.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    return destination
