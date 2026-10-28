"""Conservative stationary reaches towards a balloon; no finger commands."""
import json
from pathlib import Path

import numpy as np
from gpt_astra.motions import ready_pose, write_clip


def generate_library(destination, model):
    destination = Path(destination)
    destination.mkdir()
    ready = ready_pose(model)
    write_clip(destination / "ready", np.tile(ready, (50, 1)), model)
    descriptions = {}
    for name, sides in (("reach_left", ("left",)), ("reach_right", ("right",)),
                        ("reach_both", ("left", "right"))):
        target = ready.copy()
        for side in sides:
            target[model["names"].index(f"{side}_shoulder_pitch_joint")] = -.9
            target[model["names"].index(f"{side}_shoulder_roll_joint")] = .25 if side == "left" else -.25
        def blend(a, b, count):
            t = np.linspace(0, 1, count)[:, None]
            return a + (b-a) * (10*t**3 - 15*t**4 + 6*t**5)
        trajectory = np.concatenate([np.tile(ready, (26, 1)), blend(ready, target, 41)[1:],
                                     np.tile(target, (51, 1)), blend(target, ready, 41)[1:],
                                     np.tile(ready, (26, 1))])
        write_clip(destination / name, trajectory, model)
        descriptions[name] = (f"Reach {' and '.join(sides)} arm(s) forward and upward toward a visible target, "
                              "hold briefly, then return to standing. Stationary feet; fingers do not close. "
                              f"About {len(trajectory)/50:.2f}s total including initial dwell.")
    (destination / "actions.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    return destination
