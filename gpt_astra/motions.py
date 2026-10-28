"""Build and validate bounded SONIC motion references."""
import json
from pathlib import Path
import numpy as np
from sonic_laptop.model import load_model
from sonic_laptop.motions import read_clip, stage_library

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(__file__).resolve().parent / "runs"


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


def generate_standing_library(destination, model):
    """Default for a task without an action generator: hold the ready pose."""
    destination = Path(destination)
    destination.mkdir()
    write_clip(destination / "ready", np.tile(ready_pose(model), (50, 1)), model)
    (destination / "actions.json").write_text("{}\n")
    return destination


def prepare_session(directory, upstream, library=None, generator=None):
    model = load_model(upstream)
    if library is None:
        library = (generator or generate_standing_library)(directory / "generated", model)
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
