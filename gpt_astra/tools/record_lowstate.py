"""Record the robot's joint angles while run.py --measure grab plays the grab poses, for gpt_astra/tools/grab_report.py.
A passive subscriber to the robot state (rt/lowstate); it sends nothing. Writes wall time, the 29 body joint angles in
motor order and the IMU roll/pitch/yaw at up to 50 Hz until --seconds pass or Ctrl+C.

Run it next to run.py with the upstream sim interpreter, which has unitree_sdk2py:
  PYTHONPATH=../GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python:. \\
    ../GR00T-WholeBodyControl/.venv_sim/bin/python -m gpt_astra.tools.record_lowstate --interface <robot NIC> \\
    --out joints.csv
(simulation: --interface lo)
"""
import argparse
import csv
from pathlib import Path
import threading
import time

PERIOD = .02


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", required=True)
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=900)
    a = p.parse_args(argv)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    ChannelFactoryInitialize(a.domain, a.interface)
    stream = a.out.open("w", newline="")
    writer = csv.writer(stream)
    writer.writerow(["time", *[f"q{i}" for i in range(29)], "roll", "pitch", "yaw"])
    lock, last, rows = threading.Lock(), [0.], [0]
    def handler(message):
        now = time.time()
        with lock:
            if now - last[0] < PERIOD:
                return
            last[0] = now
            writer.writerow([f"{now:.4f}", *[f"{message.motor_state[i].q:.5f}" for i in range(29)],
                             *[f"{v:.5f}" for v in message.imu_state.rpy]])
            rows[0] += 1
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(handler, 10)
    print(f"Recording rt/lowstate on {a.interface} to {a.out}; Ctrl+C stops.", flush=True)
    end = time.monotonic() + a.seconds
    try:
        while time.monotonic() < end:
            time.sleep(1)
            with lock:
                stream.flush()
    except KeyboardInterrupt:
        pass
    with lock:
        stream.close()
    print(f"{rows[0]} samples", flush=True)
    return 0 if rows[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
