"""Replay a recorded MuJoCo state trajectory without commanding a robot."""
import argparse
import csv
import json
import math
from pathlib import Path
import threading
import time


def main():
    before = set(threading.enumerate())
    try:
        replay()
    finally:
        # MuJoCo's passive viewer closes asynchronously; let its render thread
        # finish before Python tears down GLFW, particularly on virtual displays.
        for thread in set(threading.enumerate()) - before:
            if thread.daemon:
                thread.join(timeout=5)


def replay():
    import mujoco
    import mujoco.viewer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="gpt_astra/runs/<run>")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        p.error("--speed must be finite and positive")
    run = args.run.expanduser().resolve()
    meta = json.loads((run / "trajectory_meta.json").read_text())
    scene = run / "sim_scene.xml"
    if not scene.is_file():
        scene = Path(meta["scene_path"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv, model.nu) != (meta["nq"], meta["nv"], meta["nu"]):
        raise ValueError("Scene dimensions do not match the recorded trajectory")
    data = mujoco.MjData(model)
    with (run / "trajectory.csv").open(newline="") as stream, \
         mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        rows = csv.reader(stream)
        header = next(rows)
        expected = 2 + model.nq + model.nv + model.nu
        if len(header) != expected:
            raise ValueError("Trajectory columns do not match its scene")
        wall_start, first_time = time.monotonic(), None
        for row in rows:
            if not viewer.is_running():
                break
            values = [float(value) for value in row]
            if len(values) != expected or not all(math.isfinite(value) for value in values):
                raise ValueError("Invalid trajectory row")
            if first_time is None:
                first_time = values[0]
            deadline = wall_start + (values[0]-first_time)/args.speed
            while viewer.is_running() and time.monotonic() < deadline:
                time.sleep(max(0, min(.02, deadline-time.monotonic())))
            with viewer.lock():
                data.time = values[1]
                data.qpos[:] = values[2:2+model.nq]
                data.qvel[:] = values[2+model.nq:2+model.nq+model.nv]
                data.ctrl[:] = values[2+model.nq+model.nv:]
                mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
