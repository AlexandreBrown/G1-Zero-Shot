"""Record the robot's joints, body orientation and the controller's commands during a run, for
gpt_astra/tools/grab_report.py, for debugging guard stops, and for replaying the run in MuJoCo
(gpt_astra/tools/robot_to_sim.py, then gpt_astra/replay.py). Passive DDS subscribers; it sends nothing.

Writes at up to 50 Hz until --seconds pass or Ctrl+C:
  <out>          rt/lowstate: wall time, the 29 body joint angles in motor order, IMU roll/pitch/yaw, IMU quaternion
                 (w, x, y, z), the 29 joint speeds and estimated torques (the first 33 columns are unchanged, so older
                 readers still work)
  <out>_cmd.csv  rt/lowcmd: wall time, the 29 joint targets, kp, kd and feed-forward torques the controller sends

Run it next to run.py with the upstream sim interpreter, which has unitree_sdk2py:
  PYTHONPATH=../GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python:. \\
    ../GR00T-WholeBodyControl/.venv_sim/bin/python -m gpt_astra.tools.record_lowstate --interface <robot NIC> \\
    --out joints.csv
(simulation: --interface lo)
"""
import argparse
import csv
import os
from pathlib import Path
import threading
import time

PERIOD = .02
JOINTS = 29


def parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def command_path(out: Path) -> Path:
    return out.with_name(out.stem + "_cmd.csv")


class Recorder:
    """One CSV written from a DDS callback at most every PERIOD seconds."""
    def __init__(self, path: Path, header: list[str], row):
        self.stream = path.open("w", newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(header)
        self.row, self.lock, self.last, self.rows = row, threading.Lock(), 0., 0

    def handler(self, message) -> None:
        now = time.time()
        with self.lock:
            if now - self.last < PERIOD:
                return
            self.last = now
            self.writer.writerow([f"{now:.4f}", *(f"{v:.5f}" for v in self.row(message))])
            self.rows += 1

    def flush(self) -> None:
        with self.lock:
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            self.stream.close()


def state_row(message) -> list[float]:
    motors = message.motor_state[:JOINTS]
    return [*(m.q for m in motors), *message.imu_state.rpy, *message.imu_state.quaternion,
            *(m.dq for m in motors), *(m.tau_est for m in motors)]


def command_row(message) -> list[float]:
    motors = message.motor_cmd[:JOINTS]
    return [*(m.q for m in motors), *(m.kp for m in motors), *(m.kd for m in motors), *(m.tau for m in motors)]


STATE_HEADER = ["time", *[f"q{i}" for i in range(JOINTS)], "roll", "pitch", "yaw", "qw", "qx", "qy", "qz",
                *[f"dq{i}" for i in range(JOINTS)], *[f"tau{i}" for i in range(JOINTS)]]
COMMAND_HEADER = ["time", *[f"{name}{i}" for name in ("q_target", "kp", "kd", "tau_ff") for i in range(JOINTS)]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", required=True)
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--parent", type=int, help="stop (and save) once this process has exited (run.py passes itself)")
    a = p.parse_args(argv)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    ChannelFactoryInitialize(a.domain, a.interface)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    state = Recorder(a.out, STATE_HEADER, state_row)
    command = Recorder(command_path(a.out), COMMAND_HEADER, command_row)
    subscribers = [ChannelSubscriber("rt/lowstate", LowState_), ChannelSubscriber("rt/lowcmd", LowCmd_)]
    subscribers[0].Init(state.handler, 10)
    subscribers[1].Init(command.handler, 10)
    print(f"Recording rt/lowstate and rt/lowcmd on {a.interface} to {a.out} and {command_path(a.out).name}; "
          "Ctrl+C stops.", flush=True)
    end = time.monotonic() + a.seconds
    try:
        while time.monotonic() < end and (a.parent is None or parent_alive(a.parent)):
            time.sleep(1)
            state.flush()
            command.flush()
    except KeyboardInterrupt:
        pass
    state.close()
    command.close()
    print(f"{state.rows} state samples, {command.rows} command samples", flush=True)
    return 0 if state.rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
