"""CPU-only IPC stand-in for SONIC. Never opens DDS, CUDA, or robot sockets."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonic_laptop.session import atomic_write


def run(directory):
    directory = Path(directory)
    clips = json.loads((directory / "catalog.json").read_text())
    frames = {c["index"]: c["frames"] for c in clips}
    sequence, state, index, frame = 0, 1, 0, 0
    playing, fault, result, reason = False, False, "ready", ""
    started = time.monotonic()
    motion_start = started
    while True:
        now = time.monotonic()
        try:
            stamp, run = (directory / "lease").read_text().split()
            healthy = run == "1" and -.01 <= now-float(stamp) <= .3
        except (ValueError, OSError):
            healthy = False
        if not healthy:
            fault, playing, reason = True, False, "MOCK latched: lease expired/stop"
        ready = state == 1 and now-started >= 2 and not fault
        try:
            seq, command, arg, stamp = (directory / "request").read_text().split()
            seq, arg, stamp = int(seq), int(arg), float(stamp)
            if seq > sequence and not fault:
                sequence, result = seq, "rejected: invalid command/state"
                if not -.01 <= now-stamp <= .15:
                    result = "rejected: expired command"
                elif command == "grounded_start" and ready:
                    state, result = 2, "ok"
                elif command == "motion" and state == 2 and not playing and arg in frames:
                    index, frame, playing, motion_start, result = arg, 0, True, now, "ok"
        except (ValueError, OSError):
            pass
        if playing:
            frame = min(int((now-motion_start)*50), frames[index]-1)
            if frame == frames[index]-1:
                playing = False
        atomic_write(directory / "status",
                     f"{sequence} {state} {index} {frame} {int(playing)} 0 {int(fault)} {int(ready)}\n"
                     f"{result}\n{reason}\nMOCK state: perfect tracking, no motor checks\n")
        time.sleep(.02)


if __name__ == "__main__":
    run(sys.argv[1])
