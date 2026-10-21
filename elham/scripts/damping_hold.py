"""Hold the G1 in damping for a few seconds: the same command the deploy program sends once when it stops.

The deploy program (g1_deploy_onnx_ref) stops by writing ONE damping LowCmd (kp 0, kd 8, mode 1 on the 29 body
motors) and exiting. If that one packet is lost, or the program dies without it (Ctrl+C, crash, killed terminal),
the motors may keep the policy's last targets at full gains. scripts/sonic_run.sh therefore runs this script for
2 s after every exit of the deploy program, and an operator can run it by hand on a robot that stays stiff:

    python scripts/damping_hold.py --iface enp58s0 --seconds 5

It first reads mode_machine from LowState (required in every LowCmd), then publishes the damping command at
500 Hz with Unitree's CRC. Never run it while the deploy program is running: two publishers would fight.
Needs unitree_sdk2py from the SONIC tree (its CRC library), i.e. the .venv_sim environment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dds_reader  # noqa: E402

NUM_BODY_MOTORS = 29
DAMPING_KD = 8.0
RATE_HZ = 500.0


def damping_command(mode_machine: int):
    """LowCmd identical to g1_deploy_onnx_ref's CreateDampingCommand() + LowCommandWriter(), with CRC."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = mode_machine
    for i in range(NUM_BODY_MOTORS):
        motor = cmd.motor_cmd[i]
        motor.mode = 1
        motor.q = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = DAMPING_KD
    cmd.crc = CRC().Crc(cmd)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", required=True, help="network interface wired to the robot (lo in sim2sim)")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--state-timeout", type=float, default=1.0, help="seconds to wait for LowState")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    participant = dds_reader.open_participant(args.iface, args.domain)
    state = dds_reader.TopicReader(participant, "rt/lowstate", LowState_)
    deadline = time.monotonic() + args.state_timeout
    latest = None
    while latest is None and time.monotonic() < deadline:
        time.sleep(0.01)
        samples = state.drain_raw()
        if samples:
            latest = state.deserialize(samples[-1][2])
    if latest is None:
        sys.exit(f"damping_hold: no LowState on {args.iface} within {args.state_timeout} s; nothing sent")

    cmd = damping_command(int(latest.mode_machine))
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    period = 1.0 / RATE_HZ
    start = time.monotonic()
    sent = 0
    while time.monotonic() - start < args.seconds:
        publisher.Write(cmd)
        sent += 1
        time.sleep(max(0.0, start + sent * period - time.monotonic()))
    print(
        f"damping_hold: sent {sent} damping commands (kp 0, kd {DAMPING_KD}) in {args.seconds:.1f} s "
        f"(mode_machine {cmd.mode_machine})"
    )


if __name__ == "__main__":
    main()
