"""Record a side-by-side video of a running groot_vla sim session, in two steps.

capture: store SONIC's g1_debug state (ZMQ :5557) and the head-camera JPEG frames (ZMQ :5555) to a
         pickle. CPU only: any extra GPU work during a run can stall the sim loop past the deploy's
         500 ms LowState watchdog.
render:  after the sim is stopped, re-render a third-person view from the recorded state (MuJoCo
         joint order, measured base orientation, base height fitted so the lowest foot touches the
         floor) next to the head camera, and encode an mp4.

    source env.sh
    $GROOT_VLA/.venv/bin/python tools/record_sim_video.py capture --seconds 40 --data run.pkl
    MUJOCO_GL=egl $GROOT_VLA/.venv/bin/python tools/record_sim_video.py render --data run.pkl --out sim_vla.mp4
"""

import argparse
import os
import pickle
import subprocess
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

SCENE = Path(os.environ["WBC"]) / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
PANEL = (640, 480)


def joint_qpos_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int], list[int]]:
    """qpos addresses of the 29 body joints (MuJoCo order) and the 7+7 hand joints."""
    body, left_hand, right_hand = [], [], []
    for j in range(1, model.njnt):
        name = model.joint(j).name
        target = left_hand if "left_hand" in name else right_hand if "right_hand" in name else body
        target.append(model.jnt_qposadr[j])
    return body, left_hand, right_hand


def pose_robot(model, data, state, body_adr, left_adr, right_adr, foot_geoms) -> None:
    data.qpos[:] = 0
    data.qpos[3:7] = np.asarray(state["base_quat_measured"], dtype=float).ravel()[:4]
    data.qpos[body_adr] = np.asarray(state["body_q"], dtype=float)[: len(body_adr)]
    data.qpos[left_adr] = np.asarray(state["left_hand_q"], dtype=float)[: len(left_adr)]
    data.qpos[right_adr] = np.asarray(state["right_hand_q"], dtype=float)[: len(right_adr)]
    mujoco.mj_kinematics(model, data)
    lowest = min(data.geom_xpos[g][2] - model.geom_rbound[g] * 0.25 for g in foot_geoms)
    data.qpos[2] = -lowest
    mujoco.mj_kinematics(model, data)


def capture(seconds: float, fps: int, data_path: str) -> None:
    state_sub = ZMQStateSubscriber(host="localhost", port=5557)
    camera_sub = ComposedCameraClientSensor(server_ip="localhost", port=5555)
    samples, state, ego = [], None, None
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < seconds:
        msg = state_sub.get_msg()
        state = msg if msg is not None else state
        cam = camera_sub.read()
        if cam is not None:
            ego = cv2.imencode(".jpg", cam["images"]["ego_view"])[1].tobytes()
        if state is not None and ego is not None:
            samples.append({"t": elapsed, "ego_jpg": ego,
                            **{k: np.asarray(state[k]) for k in
                               ("base_quat_measured", "body_q", "left_hand_q", "right_hand_q")}})
        time.sleep(max(0.0, start + (len(samples) + 1) / fps - time.monotonic()))
    with open(data_path, "wb") as f:
        pickle.dump({"fps": fps, "samples": samples}, f)
    print(f"captured {len(samples)} samples to {data_path}")


def render(data_path: str, out: str, label: str) -> None:
    with open(data_path, "rb") as f:
        recording = pickle.load(f)
    fps, samples = recording["fps"], recording["samples"]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    body_adr, left_adr, right_adr = joint_qpos_addresses(model)
    foot_geoms = [g for g in range(model.ngeom) if "ankle_roll" in model.body(model.geom_bodyid[g]).name]
    renderer = mujoco.Renderer(model, height=PANEL[1], width=PANEL[0])
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.4, 150.0, -12.0
    # Hidden while incomplete: players retrying a moov-less mp4 have hung the desktop.
    partial = Path(out).with_name("." + Path(out).name + ".part")
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{2 * PANEL[0]}x{PANEL[1] + 40}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-f", "mp4", str(partial)],
        stdin=subprocess.PIPE,
    )
    for sample in samples:
        pose_robot(model, data, sample, body_adr, left_adr, right_adr, foot_geoms)
        camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, -0.1])
        renderer.update_scene(data, camera)
        ego = cv2.imdecode(np.frombuffer(sample["ego_jpg"], np.uint8), cv2.IMREAD_COLOR)
        frame = np.zeros((PANEL[1] + 40, 2 * PANEL[0], 3), np.uint8)
        frame[40:, : PANEL[0]] = renderer.render()
        frame[40:, PANEL[0]:] = cv2.resize(ego, PANEL)
        cv2.putText(frame, f"{label}   t={sample['t']:5.1f}s", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "third person (re-rendered from SONIC state)", (10, PANEL[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "head camera (VLA input)", (PANEL[0] + 10, PANEL[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed; incomplete video left at {partial}")
    partial.replace(out)
    print(f"wrote {out}: {len(samples)} frames, {len(samples) / fps:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "render"])
    parser.add_argument("--data", default="run.pkl")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default="sim_vla.mp4")
    parser.add_argument("--label", default='GR00T N1.7 VLA + SONIC (sim) - prompt: "grab the bottle"')
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.seconds, args.fps, args.data)
    else:
        render(args.data, args.out, args.label)


if __name__ == "__main__":
    main()
