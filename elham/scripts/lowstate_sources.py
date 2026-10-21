"""Who publishes rt/lowstate, and how fast? Read-only: it only listens, like lowstate_monitor.py.

Use when the pre-flight reports an unexpected LowState rate or a tick that goes back. Per DDS writer it prints the
message rate and the tick steps between its consecutive messages. A healthy G1 has ONE writer at about 1040 Hz
(1 kHz tick, ~4 % repeated ticks; measured on our G1 2026-10-21). Two writers mean a second program publishes the
robot state, for example on PC2.

    python scripts/lowstate_sources.py --iface enp58s0
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dds_reader  # noqa: E402
import g1_cdr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--discovery-s", type=float, default=5.0, help="wait this long for discovery, then listen")
    parser.add_argument("--listen-s", type=float, default=2.0)
    return parser.parse_args()


def describe(samples: list[tuple[float, int, bytes, object]], seconds: float) -> list[str]:
    """One line per writer: rate, tick steps, publisher-clock spacing, and the joint positions it reports."""
    by_writer: dict[int, list[tuple[int, bytes]]] = collections.defaultdict(list)
    for _, source_ns, data, info in samples:
        by_writer[info.publication_handle].append((source_ns, data))
    lines = [f"{len(samples)} messages in {seconds:.0f} s from {len(by_writer)} writer(s)"]
    for handle, rows in by_writer.items():
        decoded = g1_cdr.decode(g1_cdr.LOW_STATE, [data for _, data in rows])
        ticks = decoded["tick"].astype(np.int64)
        stamps = np.array([source_ns for source_ns, _ in rows], dtype=np.int64)
        steps = collections.Counter(np.diff(ticks).tolist()).most_common(4)
        spacing_ms = np.diff(stamps).mean() / 1e6 if len(stamps) > 1 else float("nan")
        q = decoded["motor_state.q"][-1, :3]
        lines.append(
            f"  writer {handle}: {len(rows) / seconds:.0f} Hz | tick steps (step: count) {steps} | "
            f"ticks {ticks[0]}..{ticks[-1]} | publisher spacing {spacing_ms:.2f} ms | first 3 q {np.round(q, 3)}"
        )
    return lines


def main() -> None:
    args = parse_args()
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    reader = dds_reader.TopicReader(dds_reader.open_participant(args.iface, args.domain), "rt/lowstate", LowState_)
    time.sleep(args.discovery_s)
    reader.drain_raw()
    time.sleep(args.listen_s)
    print("\n".join(describe(reader.drain_raw(), args.listen_s)))


if __name__ == "__main__":
    main()
