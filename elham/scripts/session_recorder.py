"""Record every G1 low-level message of a trial at full rate, for datasets and sim-vs-real analysis. Read-only.

Subscribes (never publishes) to
  rt/lowstate       robot state, ~1040 Hz on the real G1 (500 Hz in SONIC's simulator): IMU, all 35 motors (q, dq,
                    ddq, tau_est, temperatures, error codes), remote buttons, tick
  rt/lowcmd         what the deploy program commands, 500 Hz: per motor q target, dq target, feed-forward
                    torque, kp, kd (the commanded action at hardware level)
  rt/secondary_imu  torso IMU
and writes them to <out>/chunk_0000.npz, chunk_0001.npz, ... every --chunk-s seconds (a crash loses at most one
chunk), plus <out>/recorder.json. scripts/sonic_run.sh starts it for every trial (sim and real) and stops it with
SIGINT when the deploy program exits; scripts/build_dataset.py merges it with the program's 50 Hz logs.

Each message keeps two timestamps: t = when the recorder took it from DDS (time.monotonic(), CLOCK_MONOTONIC, the
clock of the deploy program's time_monotonic_ms column on the same computer; readers are drained every 10 ms, so
t is batched) and t_source_ns = the publisher's DDS source timestamp (wall clock of the computer that sent it: the
robot for LowState, the deploy computer for LowCmd). Use tick (LowState) for exact ordering and gap counting.
The readers keep a deep history and are drained in batches as raw bytes (scripts/dds_reader.py), and each chunk is
decoded in bulk (scripts/g1_cdr.py): Python's per-message deserializer manages ~300 LowState/s, too slow for 1 kHz.
With --parent-pid the recorder also stops by itself when that process is gone (a launcher killed without its exit
trap left one recording for 25 min on 2026-10-21).

    python scripts/session_recorder.py --iface enp3s0 --out logs/<session>/<trial>/hw [--parent-pid PID]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g1_cdr  # noqa: E402

DRAIN_PERIOD_S = 0.01


def lowstate_arrays(messages: list[bytes]) -> dict[str, np.ndarray]:
    d = g1_cdr.decode(g1_cdr.LOW_STATE, messages)
    return {
        "tick": d["tick"],
        "mode_machine": d["mode_machine"],
        "mode_pr": d["mode_pr"],
        "imu_quat_wxyz": d["imu_state.quaternion"],
        "imu_gyro": d["imu_state.gyroscope"],
        "imu_acc": d["imu_state.accelerometer"],
        "imu_rpy": d["imu_state.rpy"],
        "q": d["motor_state.q"],
        "dq": d["motor_state.dq"],
        "ddq": d["motor_state.ddq"],
        "tau_est": d["motor_state.tau_est"],
        "temperature": d["motor_state.temperature"],
        "motorstate": d["motor_state.motorstate"],
        "vol": d["motor_state.vol"],
        "wireless_remote": d["wireless_remote"],
    }


def lowcmd_arrays(messages: list[bytes]) -> dict[str, np.ndarray]:
    d = g1_cdr.decode(g1_cdr.LOW_CMD, messages)
    return {
        "mode_machine": d["mode_machine"],
        "mode_pr": d["mode_pr"],
        **{key: d[f"motor_cmd.{key}"] for key in ("mode", "q", "dq", "tau", "kp", "kd")},
    }


def imu_arrays(messages: list[bytes]) -> dict[str, np.ndarray]:
    d = g1_cdr.decode(g1_cdr.IMU_STATE, messages)
    return {"quat_wxyz": d["quaternion"], "gyro": d["gyroscope"], "acc": d["accelerometer"], "rpy": d["rpy"]}


CONVERTERS = {"lowstate": lowstate_arrays, "lowcmd": lowcmd_arrays, "torso_imu": imu_arrays}


def compact(key: str, values: np.ndarray) -> np.ndarray:
    """float32 / int32 storage (the messages are float32 on the wire); timestamps keep 64 bits."""
    if key in ("t", "t_source_ns"):
        return values
    if values.dtype.kind == "f":
        return values.astype(np.float32)
    if values.dtype.kind in "iu":
        return values.astype(np.int64 if key == "tick" else np.int32)
    return values


class ChunkWriter:
    """Thread-safe buffers per stream, flushed to numbered .npz files.

    Messages are stored as raw CDR bytes as they arrive; they are decoded to arrays at flush time.
    """

    def __init__(self, out: Path, converters: dict) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.converters = converters
        self.buffers: dict[str, list] = {name: [] for name in converters}
        self.counts = {name: 0 for name in converters}
        self.chunks = 0

    def add(self, stream: str, data: bytes, stamp: float, source_ns: int = 0) -> None:
        with self.lock:
            self.buffers[stream].append((stamp, source_ns, data))

    def flush(self) -> None:
        with self.lock:
            buffers, self.buffers = self.buffers, {name: [] for name in self.buffers}
        arrays = {}
        for stream, items in buffers.items():
            self.counts[stream] += len(items)
            if not items:
                continue
            columns = {
                "t": np.array([stamp for stamp, _, _ in items]),
                **self.converters[stream]([data for _, _, data in items]),
                "t_source_ns": np.array([source_ns for _, source_ns, _ in items], dtype=np.int64),
            }
            for key, values in columns.items():
                arrays[f"{stream}/{key}"] = compact(key, values)
        if arrays:
            np.savez_compressed(self.out / f"chunk_{self.chunks:04d}.npz", **arrays)
            self.chunks += 1


def merge_chunks(folder: Path) -> dict[str, np.ndarray]:
    """All chunks of a recording, concatenated per array (used by build_dataset.py)."""
    merged: dict[str, list[np.ndarray]] = {}
    for path in sorted(folder.glob("chunk_*.npz")):
        with np.load(path) as chunk:
            for key in chunk.files:
                merged.setdefault(key, []).append(chunk[key])
    return {key: np.concatenate(parts) for key, parts in merged.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", required=True, help="network interface wired to the robot (lo in sim2sim)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--chunk-s", type=float, default=10.0)
    parser.add_argument("--parent-pid", type=int, help="stop when this process (the launcher) no longer exists")
    return parser.parse_args()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = parse_args()
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    import dds_reader

    writer = ChunkWriter(args.out, CONVERTERS)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: stop.set())

    participant = dds_reader.open_participant(args.iface, args.domain)
    readers = {
        "lowstate": dds_reader.TopicReader(participant, "rt/lowstate", LowState_),
        "lowcmd": dds_reader.TopicReader(participant, "rt/lowcmd", LowCmd_),
        "torso_imu": dds_reader.TopicReader(participant, "rt/secondary_imu", IMUState_),
    }

    started = time.monotonic()
    last_flush = started
    print(
        f"session_recorder: recording rt/lowstate, rt/lowcmd, rt/secondary_imu on {args.iface} -> {args.out}",
        flush=True,
    )
    while not stop.wait(DRAIN_PERIOD_S):
        for stream, reader in readers.items():
            for stamp, source_ns, data, _ in reader.drain_raw():
                writer.add(stream, data, stamp, source_ns)
        if time.monotonic() - last_flush >= args.chunk_s:
            writer.flush()
            last_flush = time.monotonic()
            if args.parent_pid and not process_alive(args.parent_pid):
                print(f"session_recorder: parent {args.parent_pid} is gone; stopping", flush=True)
                break
    for stream, reader in readers.items():
        for stamp, source_ns, data, _ in reader.drain_raw():
            writer.add(stream, data, stamp, source_ns)
    writer.flush()
    summary = {
        "iface": args.iface,
        "started_monotonic": started,
        "stopped_monotonic": time.monotonic(),
        "chunks": writer.chunks,
        "messages": writer.counts,
    }
    (args.out / "recorder.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"session_recorder: {writer.counts} messages in {writer.chunks} chunks", flush=True)


if __name__ == "__main__":
    main()
