"""Is this machine fast enough to run SONIC? Reads the "Loop timing" lines the controller prints every second in a
run's controller.log and its guard latches, for one or more run folders (robot or sim).

Reference (2026-10): real-time sim on a Mila L40S: robot state 1.5-4.4 ms old, observation->command 0.25-0.33 ms,
no latches. The GPU laptop on a weaker, non-original charger: observation 1.0 ms (0.36 ms on its own charger),
robot state 7-26 ms old, and SONIC's Start oscillated into a guard latch in ~9 of 10 runs. A run fails here when
the state age or the observation->command time reaches the slow laptop's range, or the guard latched for any
reason other than the normal end of a session.

Usage: python -m gpt_astra.tools.timing_report <run folder> [<run folder> ...]
"""
from pathlib import Path
import re
import sys

import numpy as np

AGES = ("LowState age", "IMU age")
DURATIONS = ("Obs", "Policy", "Obs 2 Motor Command")
LIMITS = {"LowState age": 7.0, "Obs 2 Motor Command": 1000.0}   # p95, ms / us
NORMAL_END = "UI stop / heartbeat lost"


def read(log: Path) -> tuple[dict, list[str]]:
    values = {name: [] for name in AGES + DURATIONS}
    latches = []
    for line in log.read_text(errors="replace").splitlines():
        if "Loop timing" in line:
            for name in AGES:
                found = re.search(re.escape(name) + r": ([\d.]+)ms", line)
                if found:
                    values[name].append(float(found.group(1)))
            for name in DURATIONS:
                found = re.search(r"(?:^|, |- )" + re.escape(name) + r": (\d+)us", line)
                if found:
                    values[name].append(float(found.group(1)))
        if "SONIC GUARD LATCHED:" in line:
            latches.append(line.split("SONIC GUARD LATCHED:", 1)[1].strip())
    return values, latches


def report(run: Path) -> bool:
    values, latches = read(run / "controller.log")
    ok = True
    print(f"{run}: {len(values['LowState age'])} timing samples (1 per second)")
    for name in AGES + DURATIONS:
        data = np.array(values[name])
        if not len(data):
            continue
        unit = "ms" if name in AGES else "us"
        p95 = np.percentile(data, 95)
        flag = ""
        if name in LIMITS and p95 > LIMITS[name]:
            flag, ok = f"  TOO SLOW (p95 > {LIMITS[name]:g} {unit})", False
        print(f"  {name:20s} median {np.median(data):8.2f}  p95 {p95:8.2f}  max {data.max():8.2f} {unit}{flag}")
    # The heartbeat latch is the normal end only as the last latch of a session that reached its end; mid-run it
    # means the laptop stalled for >300 ms.
    events = run / "events.jsonl"
    finished = events.is_file() and '"event": "finished"' in events.read_text()
    faults = [reason for i, reason in enumerate(latches)
              if not (finished and i == len(latches) - 1 and reason.startswith(NORMAL_END))]
    for reason in faults:
        print(f"  GUARD LATCHED: {reason}")
    ok = ok and not faults and bool(values["LowState age"])
    print("  verdict: " + ("OK" if ok else "NOT OK: fix power/thermals/load before running on the robot"))
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [report(Path(arg)) for arg in argv]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
