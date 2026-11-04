"""Measured, platform-specific numbers of the follow_balloon loop: head camera, balloon detection, where the grab
poses' hands meet, how the guarded walk moves, and the safety limits. One JSON file per platform in
gpt_astra/calibration/. sim.json holds what was measured in the real-time simulation (Mila); robot.json starts as a
copy and each measured section (camera, balloon, grab, walk) must be re-measured on the robot with the tools in
gpt_astra/tools/ and dated before follow mode may run on hardware (gpt_astra/ROBOT_FOLLOW_CHECKLIST.md)."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).with_name("calibration")
MEASURED_SECTIONS = ("camera", "balloon", "grab", "walk")
MOVES = ("forward", "back", "left", "right", "forward_fine", "back_fine", "left_fine", "right_fine")


@dataclass(frozen=True)
class Calibration:
    path: Path
    platform: str
    measured: dict
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    fovy_deg: float
    balloon_radius: float
    red_min: int
    red_ratio: float
    min_pixels: int
    height_range: tuple
    max_fit_residual: float
    meeting_x: tuple
    yaw_tracking: float
    settle_back: float
    meet_tolerance_x: float
    meet_tolerance_y: float
    grab_tolerance_x: float
    grab_tolerance_y: float
    walk_speed: float
    fine_walk_speed: float
    coast: dict
    fine_within: float
    settle_fast: float
    settle_fine: float
    stall: float
    kick: float
    min_step: float
    follow_x: float
    follow_tolerance_x: float
    lost_walk: float
    max_walk: float
    fence: float

    def unmeasured(self) -> list[str]:
        """Measured sections without a measurement date."""
        return [name for name in MEASURED_SECTIONS if not self.measured.get(name)]


def load(path: Path) -> Calibration:
    data = json.loads(Path(path).read_text())
    camera, balloon, grab, walk, safety = (data[k] for k in ("camera", "balloon", "grab", "walk", "safety"))
    rotation = np.array(camera["rotation"], dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2):
        raise ValueError(f"{path}: camera.rotation must be a 3x3 rotation matrix")
    coast = {MOVES.index(name): float(value) for name, value in walk["coast_m"].items()}
    if sorted(coast) != list(range(len(MOVES))):
        raise ValueError(f"{path}: walk.coast_m needs one entry per move {MOVES}")
    return Calibration(
        path=Path(path), platform=data["platform"],
        measured={name: data[name].get("measured") for name in MEASURED_SECTIONS},
        camera_position=np.array(camera["position_m"], dtype=float), camera_rotation=rotation,
        fovy_deg=float(camera["fovy_deg"]),
        balloon_radius=float(balloon["radius_m"]), red_min=int(balloon["red_min"]),
        red_ratio=float(balloon["red_ratio"]), min_pixels=int(balloon["min_pixels"]),
        height_range=tuple(balloon["height_range_m"]), max_fit_residual=float(balloon["max_fit_residual"]),
        meeting_x=tuple(grab["meeting_x_m"]), yaw_tracking=float(grab["yaw_tracking"]),
        settle_back=float(grab["settle_back_m"]),
        meet_tolerance_x=float(grab["meet_tolerance_m"][0]), meet_tolerance_y=float(grab["meet_tolerance_m"][1]),
        grab_tolerance_x=float(grab["zone_tolerance_m"][0]), grab_tolerance_y=float(grab["zone_tolerance_m"][1]),
        walk_speed=float(walk["walk_speed_mps"]), fine_walk_speed=float(walk["fine_walk_speed_mps"]),
        coast=coast, fine_within=float(walk["fine_within_m"]),
        settle_fast=float(walk["settle_s"][0]), settle_fine=float(walk["settle_s"][1]),
        stall=float(walk["stall_s"]), kick=float(walk["kick_s"]), min_step=float(walk["min_step_m"]),
        follow_x=float(walk["follow_x_m"]), follow_tolerance_x=float(walk["follow_tolerance_m"]),
        lost_walk=float(safety["lost_walk_s"]), max_walk=float(safety["max_walk_s"]),
        fence=float(safety["fence_m"]))


def platform_problem(run: Path, cal: Calibration) -> str | None:
    """Why measurements from this run folder must not go into this calibration: a robot calibration takes only
    robot runs and a simulation calibration only simulation runs."""
    prepared = next((e for e in map(json.loads, (Path(run) / "events.jsonl").read_text().splitlines())
                     if e["event"] == "prepared"), None)
    backend = prepared["backend"] if prepared else "unknown"
    wanted = "robot" if cal.platform == "robot" else "sim"
    return None if backend == wanted else f"run {run} is a {backend} run; {cal.path} takes only {wanted} runs"


def for_backend(backend: str) -> Path:
    return DIRECTORY / ("robot.json" if backend == "robot" else "sim.json")


SIM = load(DIRECTORY / "sim.json")
