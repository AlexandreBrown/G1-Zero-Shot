"""How the guarded walk really moves, from a run.py --measure walk run: per move, the distance walked, how far the
robot kept going after the last move request (the follower's stop margin, walk.coast_m), how often a walk never
started, and how far leaving the planner moved the robot (grab.settle_back_m). --write stores and dates them.

Usage: python -m gpt_astra.tools.walk_report <run folder> [--calibration gpt_astra/calibration/robot.json] [--write]
"""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpt_astra import calibration

# The balloon coordinate a move changes and its sign: walking forward brings the balloon closer (x falls), walking
# left moves it to the robot's right (y falls).
AXIS = {"forward": (0, -1), "back": (0, 1), "left": (1, -1), "right": (1, 1)}
REQUIRED = ("forward", "back", "left", "right", "forward_fine", "back_fine")   # every move the follower uses
STALLED = .02  # m: a walk that moved less never started stepping


def probes(run: Path) -> tuple[list[dict], list[dict]]:
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    return ([e for e in events if e["event"] == "walk_probe" and e["after"] is not None and not e["aborted"]],
            [e for e in events if e["event"] == "settle_probe" and e["before"] and e["after"]])


def summarize(walks: list[dict]) -> dict:
    """Per move: median distance walked and coast after the last request (m), stalls, probes."""
    result = {}
    for move in sorted({w["move"] for w in walks}):
        index, sign = AXIS[move.removesuffix("_fine")]
        rows = [w for w in walks if w["move"] == move]
        moved = [sign * (w["after"][index] - w["before"][index]) for w in rows]
        coast = [sign * (w["after"][index] - w["at_stop"][index]) for w in rows]
        going = [c for c, m in zip(coast, moved) if m >= STALLED]
        result[move] = {"probes": len(rows), "stalled": sum(m < STALLED for m in moved),
                        "moved_m": round(float(np.median(moved)), 3),
                        "coast_m": round(float(max(np.median(going), 0.)), 3) if going else None,
                        "seconds": round(float(np.median([w["seconds"] for w in rows])), 2)}
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--calibration", type=Path, default=calibration.for_backend("robot"))
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    cal = calibration.load(a.calibration)
    walks, settles = probes(a.run)
    moves = summarize(walks)
    settle_back = round(float(np.median([s["after"][0] - s["before"][0] for s in settles])), 3) if settles else None
    for move, row in moves.items():
        coast = "-" if row["coast_m"] is None else f"{row['coast_m']:.2f} m"
        print(f"{move:13s} {row['probes']} probes of {row['seconds']} s: walked {row['moved_m']:.2f} m, coast "
              f"{coast}, stalled {row['stalled']}")
    print(f"leaving the planner moved the balloon {settle_back} m farther (settle back)")
    problems = [f"{move}: every probe stalled" for move, row in moves.items() if row["coast_m"] is None]
    problems += [f"{move} not measured" for move in REQUIRED if move not in moves]
    problems += ["no settle probe"] if settle_back is None else []
    # The follower's fence dead-reckons commanded speed x time plus the coast: a robot that walks farther than that
    # breaks the bound, so the walk section is not accepted.
    for move, row in moves.items():
        commanded = (cal.fine_walk_speed if move.endswith("_fine") else cal.walk_speed) * row["seconds"]
        if row["coast_m"] is not None and row["moved_m"] > commanded + row["coast_m"] + .05:
            problems.append(f"{move} walked {row['moved_m']:.2f} m, more than commanded {commanded:.2f} m + coast")
    problems += [problem] if (problem := calibration.platform_problem(a.run, cal)) else []
    for problem in problems:
        print("FAIL: " + problem)
    if a.write and not problems:
        data = json.loads(a.calibration.read_text())
        coast = data["walk"]["coast_m"]
        for move, row in moves.items():
            coast[move] = row["coast_m"]
        for side in ("left", "right"):
            coast[side + "_fine"] = coast[side]   # the follower walks sideways at full speed only
        stalls = sum(row["stalled"] for row in moves.values())
        data["walk"]["measured"] = (f"{date.today().isoformat()} walk_report: {len(walks)} probes, {stalls} stalled; "
                                    + ", ".join(f"{m} walked {r['moved_m']:.2f} m in {r['seconds']} s"
                                                for m, r in moves.items()))
        data["grab"]["settle_back_m"] = settle_back
        a.calibration.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote walk.coast_m and grab.settle_back_m of {a.calibration}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
