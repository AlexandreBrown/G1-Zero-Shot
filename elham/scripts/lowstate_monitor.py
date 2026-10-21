"""Read-only G1 LowState monitor: pre-flight check and live alerts. It NEVER sends commands.

It only reads rt/lowstate and rt/secondary_imu (deep-history readers drained as raw bytes, scripts/dds_reader.py).
Rate and gaps are measured on the publisher's DDS timestamps, so they describe the robot's stream, not this
script's polling.
  --check 5      before launching SONIC: 5 s of LowState and the torso IMU, then PASS/FAIL (rate, gaps, tick,
                 IMU, torso IMU present, joint limits, temperatures, motor error codes). Robot hanging and damped.
  (no --check)   live: one status line per second, a bell + red line on any alert, Ctrl+C to quit.

Alerts follow the abort criteria of docs/CHECKLIST.md (thresholds in scripts/g1_limits.py). The operator acts on them;
this script cannot stop the robot. The deploy program's own stop is O (keyboard) or Select (remote).
Full-rate recording is scripts/session_recorder.py (started by sonic_run.sh for every trial).

Needs unitree_sdk2py + cyclonedds 0.10.2; the SONIC sim environment has both (install_scripts/install_mujoco_sim.sh,
then `source .venv_sim/bin/activate`). Try it first against the MuJoCo sim2sim loop on interface lo.

    python scripts/lowstate_monitor.py --iface enp3s0 --check 5
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g1_cdr  # noqa: E402
import g1_limits  # noqa: E402

RED, YELLOW, RESET, BELL = "\033[31m", "\033[33m", "\033[0m", "\a"
WINDOW = 5000


@dataclass
class Sample:
    """One LowState message, body joints in hardware order."""

    quat_wxyz: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    temperature: np.ndarray
    motor_error: np.ndarray
    mode_machine: int


@dataclass
class Alert:
    level: str
    message: str


class Stream:
    """Recent LowState arrivals (publisher timestamps, ticks) plus the latest message."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self.stamps: collections.deque[float] = collections.deque(maxlen=WINDOW)
        self.ticks: collections.deque[int] = collections.deque(maxlen=WINDOW)
        self.latest = None
        self.latest_taken = 0.0
        self.total = 0

    def poll(self) -> None:
        taken = self.reader.drain_raw()
        if not taken:
            return
        ticks = g1_cdr.decode(g1_cdr.LOW_STATE, [data for _, _, data, _ in taken])["tick"]
        self.stamps.extend(source_ns * 1e-9 for _, source_ns, _, _ in taken)
        self.ticks.extend(int(tick) for tick in ticks)
        self.latest = self.reader.deserialize(taken[-1][2])
        self.latest_taken = taken[-1][0]
        self.total += len(taken)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, object, int]:
        return np.array(self.stamps), np.array(self.ticks), self.latest, self.total


def sample_from_lowstate(msg) -> Sample:
    motors = msg.motor_state[: g1_limits.NUM_JOINTS]
    return Sample(
        quat_wxyz=np.array(msg.imu_state.quaternion, dtype=float),
        q=np.array([m.q for m in motors]),
        dq=np.array([m.dq for m in motors]),
        tau=np.array([m.tau_est for m in motors]),
        temperature=np.array([max(m.temperature) for m in motors], dtype=float),
        motor_error=np.array([m.motorstate for m in motors]),
        mode_machine=int(msg.mode_machine),
    )


def stream_alerts(stamps: np.ndarray, ticks: np.ndarray, thresholds: g1_limits.Thresholds) -> list[Alert]:
    """Rate, gaps and tick continuity of the LowState stream."""
    if len(stamps) < 2:
        return [Alert("stop", "no LowState received")]
    rate = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-9)
    alerts = []
    if rate < thresholds.lowstate_min_hz:
        alerts.append(Alert("stop", f"LowState rate {rate:.0f} Hz < {thresholds.lowstate_min_hz:.0f} Hz"))
    gap = float(np.diff(stamps).max())
    if gap > thresholds.lowstate_max_gap_s:
        alerts.append(Alert("stop", f"LowState gap {gap * 1000:.0f} ms"))
    # The real G1 publishes at ~1040 Hz with a 1 kHz tick, so ~4 % of messages repeat the previous tick (measured
    # 2026-10-21, one writer). Only a tick that goes back means a second source or a restart.
    if np.any(np.diff(ticks) < 0):
        alerts.append(Alert("stop", "LowState tick went backwards"))
    return alerts


def state_alerts(sample: Sample, table: g1_limits.JointTable, thresholds: g1_limits.Thresholds) -> list[Alert]:
    """Abort criteria that one message can show."""
    alerts = []
    tilt = float(g1_limits.tilt_deg(sample.quat_wxyz))
    if tilt > thresholds.tilt_abs_deg:
        alerts.append(Alert("stop", f"base tilt {tilt:.0f} deg"))
    margin = g1_limits.position_margin(sample.q, table)
    j = int(np.argmin(margin))
    if margin[j] < thresholds.position_margin:
        alerts.append(Alert("stop", f"{table.names[j]} {margin[j]:+.3f} rad from its limit"))
    speed = np.abs(sample.dq) / g1_limits.velocity_limit(table, thresholds)
    j = int(np.argmax(speed))
    if speed[j] > 1.0:
        alerts.append(Alert("stop", f"{table.names[j]} speed {abs(sample.dq[j]):.1f} rad/s"))
    torque = np.abs(sample.tau) / table.tau_gate
    j = int(np.argmax(torque))
    if torque[j] > thresholds.torque_fraction:
        alerts.append(Alert("warn", f"{table.names[j]} torque {abs(sample.tau[j]):.0f} Nm = {torque[j]:.0%} of limit"))
    j = int(np.argmax(sample.temperature))
    if sample.temperature[j] >= thresholds.temp_stop:
        alerts.append(Alert("stop", f"{table.names[j]} {sample.temperature[j]:.0f} C: end the session"))
    elif sample.temperature[j] >= thresholds.temp_warn:
        alerts.append(Alert("warn", f"{table.names[j]} {sample.temperature[j]:.0f} C: cool down after this trial"))
    bad = np.flatnonzero(sample.motor_error)
    if bad.size:
        alerts.append(
            Alert("stop", "motor error codes: " + ", ".join(f"{table.names[i]}={sample.motor_error[i]}" for i in bad))
        )
    return alerts


def preflight(
    stamps: np.ndarray,
    ticks: np.ndarray,
    sample: Sample | None,
    table: g1_limits.JointTable,
    thresholds: g1_limits.Thresholds,
) -> list[Alert]:
    """Everything that must hold before the deploy program is launched (robot hanging and damped)."""
    alerts = stream_alerts(stamps, ticks, thresholds)
    if sample is None:
        return alerts
    norm = float(np.linalg.norm(sample.quat_wxyz))
    if abs(norm - 1.0) > 0.01:
        alerts.append(Alert("stop", f"IMU quaternion norm {norm:.3f}"))
    alerts += state_alerts(sample, table, thresholds)
    hottest = float(sample.temperature.max())
    if hottest >= 45.0:
        alerts.append(Alert("warn", f"motors already at {hottest:.0f} C before the session"))
    return alerts


def torso_imu_alerts(messages: int, seconds: float, min_hz: float = 50.0) -> list[Alert]:
    """The deploy program stops into damping at its first control tick (right after ], robot on its feet) when
    rt/secondary_imu is missing, so the pre-flight requires it."""
    if messages < min_hz * seconds:
        return [Alert("stop", f"torso IMU (rt/secondary_imu): {messages} messages in {seconds:.0f} s")]
    return []


def format_status(stamps: np.ndarray, sample: Sample, table: g1_limits.JointTable) -> str:
    recent = stamps[stamps > stamps[-1] - 1.0]
    rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9) if len(recent) > 1 else 0.0
    margin = g1_limits.position_margin(sample.q, table)
    hottest = int(np.argmax(sample.temperature))
    tightest = table.names[int(np.argmin(margin))]
    return (
        f"{rate:4.0f} Hz | tilt {float(g1_limits.tilt_deg(sample.quat_wxyz)):4.1f} deg | "
        f"max |dq| {np.abs(sample.dq).max():4.1f} | min margin {margin.min():+.2f} ({tightest}) | "
        f"hottest {sample.temperature[hottest]:.0f} C ({table.names[hottest]}) | mode_machine {sample.mode_machine}"
    )


def print_alerts(alerts: list[Alert]) -> None:
    for alert in alerts:
        colour = RED if alert.level == "stop" else YELLOW
        print(f"{BELL}{colour}[{alert.level.upper()}] {alert.message}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", required=True, help="network interface wired to the robot (lo for sim2sim)")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--check", type=float, default=0.0, help="pre-flight: listen this many seconds, then PASS/FAIL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_

    import dds_reader

    table = g1_limits.load_joint_table()
    thresholds = g1_limits.Thresholds()
    participant = dds_reader.open_participant(args.iface, args.domain)
    stream = Stream(dds_reader.TopicReader(participant, "rt/lowstate", LowState_))
    torso_imu = dds_reader.TopicReader(participant, "rt/secondary_imu", IMUState_)

    if args.check:
        deadline = time.monotonic() + args.check
        torso_messages = 0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            stream.poll()
            torso_messages += len(torso_imu.drain_raw())
        stamps, ticks, latest, total = stream.snapshot()
        sample = sample_from_lowstate(latest) if latest is not None else None
        alerts = preflight(stamps, ticks, sample, table, thresholds)
        alerts += torso_imu_alerts(torso_messages, args.check)
        if sample is not None:
            print(format_status(stamps, sample, table))
        print_alerts(alerts)
        failed = any(a.level == "stop" for a in alerts)
        print(f"PRE-FLIGHT: {'FAIL' if failed else 'PASS'} ({total} messages in {args.check:.0f} s)")
        sys.exit(1 if failed else 0)

    last_print = 0.0
    shown: set[str] = set()
    try:
        while True:
            time.sleep(0.1)
            stream.poll()
            stamps, ticks, latest, _ = stream.snapshot()
            if latest is None:
                continue
            sample = sample_from_lowstate(latest)
            recent = stamps > stamps[-1] - 1.0
            alerts = state_alerts(sample, table, thresholds)
            if time.monotonic() - stream.latest_taken > 0.5:
                alerts.append(Alert("stop", "LowState stopped arriving"))
            if recent.sum() > 1:
                alerts += stream_alerts(stamps[recent], ticks[recent], thresholds)
            due = time.monotonic() - last_print > 1.0
            fresh = [a for a in alerts if a.message not in shown]
            if fresh or (alerts and due):
                print_alerts(alerts if due else fresh)
            shown = {a.message for a in alerts}
            if due:
                age = time.monotonic() - stream.latest_taken
                print(f"{format_status(stamps, sample, table)} | last message {age:.1f} s ago", flush=True)
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
