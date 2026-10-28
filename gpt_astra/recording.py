"""Streaming simulation video and generalized-coordinate trajectory recording."""
import csv
import json
import math
from pathlib import Path
import time


class SimulationRecorder:
    def __init__(self, model, data, directory, scene_path):
        import cv2
        import mujoco
        self.cv2, self.mujoco = cv2, mujoco
        self.model, self.data = model, data
        self.directory = Path(directory)
        self.start = time.monotonic()
        self.started_at_unix = time.time()
        self.next_sample = self.next_video = self.next_flush = 0.0
        self.video_fps, self.sample_hz = 15, 50
        self.video_origin = None
        self.video_frames = 0
        self.previous_frame = None
        self.stream = (self.directory / "trajectory.csv").open("w", newline="")
        self.csv = csv.writer(self.stream)
        self.csv.writerow(["elapsed_s", "sim_time_s", *[f"qpos_{i}" for i in range(model.nq)],
                           *[f"qvel_{i}" for i in range(model.nv)], *[f"ctrl_{i}" for i in range(model.nu)]])
        self.metadata = {"scene_path": str(scene_path), "started_at_unix": self.started_at_unix,
                         "nq": model.nq, "nv": model.nv, "nu": model.nu,
                         "sample_hz": self.sample_hz, "video_fps": self.video_fps,
                         "joints": [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                                     "qpos_address": int(model.jnt_qposadr[i]),
                                     "qvel_address": int(model.jnt_dofadr[i])} for i in range(model.njnt)]}
        self.save_metadata()
        self.video = cv2.VideoWriter(str(self.directory / "simulation.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, (640, 480))
        if not self.video.isOpened():
            self.stream.close()
            raise RuntimeError("Cannot create simulation.mp4; OpenCV needs an MP4 video encoder")
        try:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self.camera)
            self.camera.azimuth, self.camera.elevation, self.camera.distance = 120, -20, 2.4
        except Exception:
            self.video.release()
            self.stream.close()
            raise

    def capture(self):
        elapsed = time.monotonic() - self.start
        if elapsed >= self.next_sample:
            values = [elapsed, self.data.time, *self.data.qpos, *self.data.qvel, *self.data.ctrl]
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError("Non-finite simulation state; trajectory recording stopped")
            self.csv.writerow([format(float(value), ".9g") for value in values])
            self.next_sample = elapsed + 1 / self.sample_hz
        if elapsed >= self.next_video:
            self.camera.lookat[:] = self.data.body("pelvis").xpos
            self.renderer.update_scene(self.data, camera=self.camera)
            frame = self.cv2.cvtColor(self.renderer.render(), self.cv2.COLOR_RGB2BGR)
            if self.video_origin is None:
                self.video_origin = elapsed
            # Keep video time aligned with elapsed time even when rendering drops frames.
            target = int((elapsed - self.video_origin) * self.video_fps)
            while self.previous_frame is not None and self.video_frames < target:
                self.video.write(self.previous_frame)
                self.video_frames += 1
            self.video.write(frame)
            self.video_frames += 1
            self.previous_frame = frame
            self.next_video = elapsed + 1 / self.video_fps
        if elapsed >= self.next_flush:
            self.stream.flush()
            self.next_flush = elapsed + 1

    def save_metadata(self):
        (self.directory / "trajectory_meta.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def close(self):
        self.video.release()
        self.renderer.close()
        self.stream.close()
        self.metadata.update(video_start_elapsed_s=self.video_origin, video_frames=self.video_frames)
        self.save_metadata()
