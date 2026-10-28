"""Generate a narrow, deterministic arm-tap vocabulary; validate with existing gates.

These are kinematic commissioning references, NOT a proven balloon policy. The
learned SONIC encoder/controller follows them; GPT only selects their names.
"""
import json
from pathlib import Path
import numpy as np
from sonic_laptop.model import load_model
from sonic_laptop.motions import read_clip, stage_library

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(__file__).resolve().parent / ".runtime"


def ready_pose(model):
    q = np.array(model["default"], dtype=float)
    # Same clear-arm pose used in this repository's independent FK guard tests.
    for side, roll in (("left", .4), ("right", -.4)):
        for part, value in (("shoulder_pitch", -.4), ("shoulder_roll", roll), ("elbow", 1.0)):
            q[model["names"].index(f"{side}_{part}_joint")] = value
    return q


def write_clip(path, q, model):
    path.mkdir(parents=True)
    joints = np.empty_like(q)
    joints[:, model["mapping"]] = q
    arrays = {"joint_pos": joints, "joint_vel": np.gradient(joints, .02, axis=0),
              "body_pos": np.tile([0, 0, .793], (len(q), 1)),
              "body_quat": np.tile([1, 0, 0, 0], (len(q), 1)),
              "body_lin_vel": np.zeros((len(q), 3)), "body_ang_vel": np.zeros((len(q), 3))}
    for name, values in arrays.items():
        np.savetxt(path / (name + ".csv"), values, delimiter=",", comments="", fmt="%.10g",
                   header=",".join(f"{name}_{i}" for i in range(values.shape[1])))
    # Encoder mode 0 (joint-based); this is the importer contract already used here.
    (path / "metadata.txt").write_text(f"Body part indexes:\n[0]\nTotal timesteps: {len(q)}\n")
    read_clip(path, model)  # Reject limits, nonfinite values, clearance, speed, discontinuity.


def generate_library(destination, model):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite library: {destination}")
    destination.mkdir(parents=True)
    ready = ready_pose(model)
    write_clip(destination / "ready", np.tile(ready, (50, 1)), model)
    descriptions = {}
    for side in ("left", "right"):
        for level, angle in (("small", .3), ("high", .6)):
            name = f"tap_{side}_{level}"
            peak = ready.copy()
            peak[model["names"].index(f"{side}_shoulder_pitch_joint")] -= angle
            # 0.5 s ready, 0.4 s upward stroke, 0.6 s return, 0.5 s ready.
            def blend(a, b, count):
                u = np.linspace(0, 1, count)[:, None]
                return a + (b-a) * (10*u**3 - 15*u**4 + 6*u**5)
            q = np.concatenate([np.tile(ready, (26, 1)), blend(ready, peak, 21)[1:],
                                blend(peak, ready, 31)[1:], np.tile(ready, (26, 1))])
            write_clip(destination / name, q, model)
            descriptions[name] = (f"Robot {side} arm upward {level} tap from ready; shoulder pitch "
                                  f"changes by {angle} rad, then returns. About {len(q)/50:.2f}s total; "
                                  "0.5s initial dwell. Feet stay in place; no finger motion.")
    (destination / "actions.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    return destination


def prepare_session(directory, upstream, library=None):
    model = load_model(upstream)
    if library is None:
        library = generate_library(directory / "generated", model)
    library = Path(library)
    descriptions = json.loads((library / "actions.json").read_text())
    catalog = stage_library(library / "ready", library, directory / "motions", model)
    indexes = {c["name"]: c["index"] for c in catalog if c["name"] != "ready"}
    if set(indexes) != set(descriptions) or {"hold", "stop"} & set(indexes):
        raise ValueError("actions.json must describe each non-ready clip exactly; hold/stop are reserved")
    if not all(isinstance(v, str) and 0 < len(v) <= 500 for v in descriptions.values()):
        raise ValueError("Each action needs a short description")
    actions = {"hold": "Keep the current ready reference; execute no new motion.",
               "stop": "Pause decisions and hold the ready reference; operator must intervene.", **descriptions}
    return model, catalog, indexes, actions
