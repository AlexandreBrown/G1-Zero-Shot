"""Pack a run folder into a compact but complete copy for sharing (e.g. in git): every log and the GPT input frames
as they are, the robot state/command CSVs gzipped, and the per-frame camera images (camera/) as one MP4 plus a table
of each frame's times. Leaves out what the code regenerates (the staged motion library) and live IPC files.

Usage: python -m gpt_astra.tools.compact_run <run folder> <output folder>
Reconstruct: gunzip -k <out>/robot_state.csv.gz before gpt_astra/tools/robot_to_sim.py; camera.mp4 frame k is row k
of camera_frames.csv (laptop receive time and the robot's capture time; the same frames appear in trace.jsonl).
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

FILES = ("events.jsonl", "trace.jsonl", "controller.log", "recorder.log", "simulator.log", "status", "init.txt",
         "catalog.json", "acceptance-limits.json", "trajectory_meta.json", "simulation.mp4")
FOLDERS = ("frames", "task")
COMPRESSED = ("robot_state.csv", "robot_state_cmd.csv", "trajectory.csv")


def camera_video(run: Path, out: Path) -> int:
    """camera/*.jpg -> camera.mp4 (written hidden, then renamed) and camera_frames.csv; returns the frame count."""
    frames = sorted((run / "camera").glob("*.jpg"))
    if not frames:
        return 0
    trace = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    times = {e["image"]: e for e in trace if e["event"] == "camera"}
    stamps = [times[f"camera/{f.name}"]["time"] for f in frames if f"camera/{f.name}" in times]
    rate = max(1, round((len(stamps) - 1) / (stamps[-1] - stamps[0]))) if len(stamps) > 1 else 10
    listing = out / ".camera_frames.txt"
    listing.write_text("".join(f"file '{f.resolve()}'\nduration {1 / rate}\n" for f in frames))
    partial = out / ".camera.mp4.part"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-r", str(rate),
                    "-f", "mp4", str(partial)], check=True)
    partial.rename(out / "camera.mp4")
    listing.unlink()
    with (out / "camera_frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mp4_frame", "image", "time", "received_monotonic", "captured"])
        for k, f in enumerate(frames):
            e = times.get(f"camera/{f.name}", {})
            writer.writerow([k, f.name, e.get("time"), e.get("received_monotonic"), e.get("captured")])
    return len(frames)


def compact(run: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (run / name).is_file():
            shutil.copy2(run / name, out / name)
    for name in FOLDERS:
        if (run / name).is_dir():
            shutil.copytree(run / name, out / name, dirs_exist_ok=True)
    for name in COMPRESSED:
        if (run / name).is_file():
            with (run / name).open("rb") as source, gzip.open(out / (name + ".gz"), "wb", compresslevel=9) as target:
                shutil.copyfileobj(source, target)
    return {"camera_frames": camera_video(run, out) if (run / "camera").is_dir() else 0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args(argv)
    result = compact(a.run, a.out)
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file())
    print(f"{a.run.name}: {result['camera_frames']} camera frames, {size / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
