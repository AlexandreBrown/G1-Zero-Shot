#!/usr/bin/env python3
"""Score a follow_balloon run: per place, seconds from the balloon stopping to the grab, from the pull to the
robot letting go, and how often the balloon came within 0.22 m of the pelvis (bumping the body).

Usage: python -m gpt_astra.tools.analyze_follow <run-folder>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main(run: Path) -> None:
    t0 = json.loads((run / "trajectory_meta.json").read_text())["started_at_unix"]
    rows = [list(map(float, r)) for r in list(csv.reader((run / "trajectory.csv").open()))[1:]]
    task = [json.loads(line) for line in (run / "task_events.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    lets = [e["time"] for e in events if e["event"] == "let_go"]
    for i, e in enumerate(task):
        if e["event"] == "stopped":
            nxt = next((t for t in task[i+1:] if t["event"] in ("grabbed", "grab_timeout")), None)
            print(f"place {e['place']}: {nxt['event'] if nxt else 'unfinished'}"
                  + (f" {nxt['wall_time'] - e['wall_time']:.1f} s after the balloon stopped" if nxt else ""))
        if e["event"] == "pull_away":
            let = next((t for t in lets if t > e["wall_time"]), None)
            print(f"   pulled away; robot let go {let - e['wall_time']:.1f} s later" if let else "   pulled away; never let go")
    close, previous = [], -1
    for r in rows:
        q = r[2:59]
        w, x, y, z = q[3:7]
        rot = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        ball = rot.T @ (np.array(q[50:53]) - np.array(q[0:3]))
        if -1 < ball[0] < .22 and r[0] - previous > 1:
            close.append(round(r[0], 1))
        if -1 < ball[0] < .22:
            previous = r[0]
    print("balloon within 0.22 m of the pelvis (body bump) at video s:", close)
    pelvis = np.array([[r[0], *r[2:4]] for r in rows])
    walks = [e for e in events if e["event"] == "steer"]
    bouts = []
    for start, stop in zip(walks, walks[1:]):
        if start["direction"] is None:
            continue
        i, j = (np.searchsorted(pelvis[:, 0], e["time"] - t0) for e in (start, stop))
        j = min(np.searchsorted(pelvis[:, 0], stop["time"] - t0 + 1.0), len(pelvis) - 1)
        bouts.append((start["direction"], stop["time"] - start["time"], np.linalg.norm(pelvis[j, 1:] - pelvis[i, 1:])))
    for label, keep in (("fast", lambda d: d < 4), ("fine", lambda d: d >= 4)):
        chosen = [(s, m) for d, s, m in bouts if keep(d)]
        if chosen:
            print(f"{label} walks: {len(chosen)}, median {np.median([s for s, _ in chosen]):.1f} s, "
                  f"moved median {np.median([m for _, m in chosen]):.2f} m (range {min(m for _, m in chosen):.2f}"
                  f"-{max(m for _, m in chosen):.2f})")
    for e in (e for e in events if e["event"] == "catch"):
        r = rows[min(np.searchsorted([r[0] for r in rows], e["time"] - t0), len(rows) - 1)]
        q = r[2:59]
        w, x, y, z = q[3:7]
        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        d = np.array(q[50:52]) - np.array(q[0:2])
        truth = (d[0] * np.cos(heading) + d[1] * np.sin(heading), -d[0] * np.sin(heading) + d[1] * np.cos(heading))
        ball = e["ball"] or [np.nan, np.nan]
        print(f"catch: grab {e['grab']}, estimate ({ball[0]:.3f}, {ball[1]:+.3f}), truth ({truth[0]:.3f}, {truth[1]:+.3f})")
    decisions = [e for e in events if e["event"] == "decision"]
    print("policy decisions:", {a: sum(d["action"] == a for d in decisions) for a in {d["action"] for d in decisions}})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
