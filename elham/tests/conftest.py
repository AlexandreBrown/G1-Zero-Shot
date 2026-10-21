"""Synthetic trial folders in the deploy program's formats (state logger CSVs, target / obs logs, recorder chunks)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_HW = np.array(
    [
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        0,
        0,
        0,
        0.2,
        0.2,
        0,
        0.6,
        0,
        0,
        0,
        0.2,
        -0.2,
        0,
        0.6,
        0,
        0,
        0,
    ]
)


def write_sink(path: Path, prefix: str, index: np.ndarray, data: np.ndarray, labels: list[str] | None = None) -> None:
    """FileSink format: index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,<prefix>_<i>..."""
    columns = labels or [f"{prefix}_{i}" for i in range(data.shape[1])]
    lines = ["index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp," + ",".join(columns)]
    for row, i in enumerate(index):
        t_ms = 20.0 * row
        meta = f"{i},{t_ms},{1.7e12 + t_ms},{5000.0 + t_ms},0"
        lines.append(meta + "," + ",".join(f"{v:.6f}" for v in data[row]))
    path.write_text("\n".join(lines) + "\n")


def make_trial(
    root: Path,
    ticks: int = 100,
    torque_scale: float = 3.0,
    tilt_quat=(1.0, 0.0, 0.0, 0.0),
    drop_post_row: int | None = None,
) -> Path:
    trial = root / "trial"
    state = trial / "state"
    state.mkdir(parents=True)
    index = np.arange(1000, 1000 + ticks)
    rng = np.random.default_rng(0)
    q = DEFAULT_HW + 0.01 * rng.normal(size=(ticks, 29))
    write_sink(state / "q.csv", "q", index, q)
    write_sink(state / "dq.csv", "dq", index, 0.5 * rng.normal(size=(ticks, 29)))
    write_sink(state / "motor_torque.csv", "tau", index, torque_scale * np.ones((ticks, 29)))
    write_sink(state / "motor_temperature.csv", "temp", index, 40.0 * np.ones((ticks, 58)))
    write_sink(state / "motor_error.csv", "err", index, np.zeros((ticks, 29)))
    write_sink(state / "action.csv", "act", index, np.zeros((ticks, 29)))
    quat = np.tile(np.array(tilt_quat), (ticks, 1))
    write_sink(state / "base_quat.csv", "base_q", index, quat, ["base_qw", "base_qx", "base_qy", "base_qz"])
    write_sink(state / "torso_quat.csv", "torso_q", index, quat, ["torso_qw", "torso_qx", "torso_qy", "torso_qz"])
    for name in ("base_ang_vel", "base_accel", "torso_ang_vel", "torso_accel"):
        write_sink(state / f"{name}.csv", name, index, np.zeros((ticks, 3)), ["x", "y", "z"])
    post_index = index if drop_post_row is None else np.delete(index, drop_post_row)
    playing = (np.arange(ticks) >= ticks // 2).astype(float)[:, None]
    if drop_post_row is not None:
        playing = np.delete(playing, drop_post_row, axis=0)
    write_sink(state / "motion_playing.csv", "playing", post_index, playing)
    write_sink(state / "encoder_mode.csv", "encoder_mode", post_index, np.zeros((len(post_index), 1)))
    write_sink(state / "token_state.csv", "token", post_index, np.zeros((len(post_index), 64)))
    names = ["index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,motion_name"]
    names += [f'{i},0,0,0,0,"asap_squat_lv1"' for i in post_index]
    (state / "motion_name.csv").write_text("\n".join(names) + "\n")
    target = np.concatenate((np.zeros((ticks, 3)), np.tile([1.0, 0, 0, 0], (ticks, 1)), q + 0.02), axis=1)
    (trial / "target.csv").write_text("\n".join(",".join(f"{v:.6f}" for v in row) + "," for row in target) + "\n")
    (trial / "obs.csv").write_text("\n".join(",".join("0.1" for _ in range(994)) + "," for _ in range(ticks)) + "\n")
    (trial / "run_manifest.txt").write_text("mode: sim\n")
    return trial


@pytest.fixture
def trial(tmp_path: Path) -> Path:
    return make_trial(tmp_path)
