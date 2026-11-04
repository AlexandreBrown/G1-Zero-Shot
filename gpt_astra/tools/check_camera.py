"""Measure the head camera and the balloon detection before follow mode, with the robot standing still (SONIC
holding the ready pose, or in the gantry). Nothing moves; only the camera stream is read.

1. Empty view: no balloon in sight (then, with --red-object, a person in red clothing in view): locate() must find
   nothing.
2. Spots: put the balloon's CENTRE at each printed spot (pelvis frame: x ahead of the pelvis, y to the robot's left,
   z above the pelvis; mark the floor from the point under the pelvis and use the measured pelvis height) and press
   ENTER. The default spots span where the follower stands the balloon (0.4-0.7 m ahead, +-0.12 m).
3. It prints locate()'s error at every spot with the current calibration, fits the camera pose and field of view
   (the balloon diameter is taped, not fitted), suggests colour and shape thresholds from the balloon's pixels,
   and with --write stores and dates the camera and balloon sections. Frames and numbers go to --out.

Usage: python -m gpt_astra.tools.check_camera --camera-host <robot ip> --balloon-diameter 0.15 \\
           [--calibration gpt_astra/calibration/robot.json] [--write] [--red-object] [--out DIR]
"""
import argparse
from dataclasses import replace
from datetime import date
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from gpt_astra import calibration
from gpt_astra.follow import Blob, blob_position, fit_blob, locate

SPOTS = [(x, y, .16) for x in (.4, .55, .7) for y in (-.12, 0., .12)]


def rotation(omega: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector omega (Rodrigues)."""
    angle = np.linalg.norm(omega)
    if angle < 1e-12:
        return np.eye(3)
    k = omega / angle
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def project(point: np.ndarray, position: np.ndarray, rot: np.ndarray, fovy_deg: float, radius: float,
            width: int, height: int) -> np.ndarray:
    """Pixel centre (u, v) and apparent radius of a sphere at point (pelvis frame), as locate() inverts it."""
    camera = rot.T @ (point - position)
    focal = height / 2 / np.tan(np.radians(fovy_deg) / 2)
    depth = -camera[2]
    return np.array([width / 2 + focal * camera[0] / depth, height / 2 - focal * camera[1] / depth,
                     focal * radius / depth])


def fit_camera(points: list, observed: list, cal: calibration.Calibration, radius: float, width: int,
               height: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Camera position, rotation and vertical field of view that best explain the observed (u, v, r) of a sphere
    of the given radius at the known points (Levenberg-Marquardt from the calibration's camera). Also returns the
    RMS pixel error of the fit."""
    def unpack(p):
        return cal.camera_position + p[3:6], cal.camera_rotation @ rotation(p[:3]), cal.fovy_deg * np.exp(p[6])
    def residuals(p):
        position, rot, fovy = unpack(p)
        return np.concatenate([project(np.array(q), position, rot, fovy, radius, width, height) - np.array(o)
                               for q, o in zip(points, observed)])
    p, damping = np.zeros(7), 1e-3
    for _ in range(iterations):
        r = residuals(p)
        jacobian = np.column_stack([(residuals(p + e) - r) / 1e-6 for e in np.eye(7) * 1e-6])
        normal = jacobian.T @ jacobian
        step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), -jacobian.T @ r)
        if np.sum(residuals(p + step) ** 2) < np.sum(r ** 2):
            p, damping = p + step, damping / 3
        else:
            damping *= 5
    position, rot, fovy = unpack(p)
    u, s, vt = np.linalg.svd(rot)
    return position, u @ vt, float(fovy), float(np.sqrt(np.mean(residuals(p) ** 2)))


def balloon_pixels(jpeg: bytes, blob: Blob) -> np.ndarray:
    """RGB of the reddish pixels inside the fitted circle (shrunk by 15% to stay off the edge): the specular
    highlight is not red and must not loosen the thresholds; the hole fill in fit_blob already bridges it."""
    image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).astype(float)
    rows, cols = np.mgrid[:image.shape[0], :image.shape[1]]
    inside = np.hypot(cols - blob.u, rows - blob.v) < .85 * blob.radius
    reddish = (image[..., 0] > image[..., 1]) & (image[..., 0] > image[..., 2])
    return image[inside & reddish]


def summarize(jpegs: list[bytes], cal: calibration.Calibration) -> dict:
    """Median blob and locate() result of one spot's frames, with colour and shape statistics."""
    blobs = [(jpeg, fit_blob(jpeg, cal)) for jpeg in jpegs]
    blobs = [(jpeg, blob) for jpeg, blob in blobs if blob is not None]
    if not blobs:
        return {"frames": len(jpegs), "blobs": 0}
    located = [p for p in (locate(jpeg, cal) for jpeg, _ in blobs) if p is not None]
    pixels = np.concatenate([balloon_pixels(jpeg, blob) for jpeg, blob in blobs])
    ratio = np.minimum(pixels[:, 0] / np.maximum(pixels[:, 1], 1), pixels[:, 0] / np.maximum(pixels[:, 2], 1))
    return {"frames": len(jpegs), "blobs": len(blobs), "located": len(located),
            "uvr": np.median([[b.u, b.v, b.radius] for _, b in blobs], axis=0).tolist(),
            "size": [blobs[0][1].width, blobs[0][1].height],
            "residuals": [b.residual for _, b in blobs],
            "position": np.median(located, axis=0).tolist() if located else None,
            "red_p5": float(np.percentile(pixels[:, 0], 5)), "ratio_p5": float(np.percentile(ratio, 5))}


# Floors and cap for the suggested thresholds: looser colours or circle test would accept clothing and clutter
# (a red square fits a circle with residual 0.11).
RED_MIN_FLOOR, RED_RATIO_FLOOR, RESIDUAL_CAP = 50, 1.5, .1


def analyze(spots: list, results: list[dict], cal: calibration.Calibration, radius: float,
            empty_frames: list[bytes], spot_frames: list[bytes]) -> dict:
    """Errors with the current calibration, the fitted camera, the suggested balloon thresholds, how often the
    suggested calibration still 'sees' a balloon in the empty views (must be never) and finds it at the spots."""
    seen = [(np.array(spot), r) for spot, r in zip(spots, results) if r.get("blobs")]
    if len(seen) < 4:
        raise ValueError(f"Only {len(seen)} spots had a red blob; need at least 4 to fit the camera")
    errors = [np.linalg.norm(np.array(r["position"]) - spot) if r["position"] else np.inf for spot, r in seen]
    width, height = seen[0][1]["size"]
    position, rot, fovy, rms = fit_camera([s for s, _ in seen], [r["uvr"] for _, r in seen], cal, radius,
                                          width, height)
    fitted = replace(cal, camera_position=position, camera_rotation=rot, fovy_deg=fovy, balloon_radius=radius)
    after = [float(np.linalg.norm(blob_position(Blob(*r["uvr"], 0., 0, width, height), fitted) - spot))
             for spot, r in seen]
    residual = float(np.percentile([x for _, r in seen for x in r["residuals"]], 95))
    heights = [s[2] for s, _ in seen]
    suggested = replace(
        fitted,
        red_min=int(max(RED_MIN_FLOOR, min(cal.red_min, .8 * min(r["red_p5"] for _, r in seen)))),
        red_ratio=round(float(max(RED_RATIO_FLOOR, min(cal.red_ratio, .8 * min(r["ratio_p5"] for _, r in seen)))), 2),
        max_fit_residual=round(float(min(RESIDUAL_CAP, max(cal.max_fit_residual, 1.5 * residual))), 3),
        height_range=(round(min(heights) - .2, 2), round(max(heights) + .2, 2)))
    return {"empty_detections": sum(locate(jpeg, suggested) is not None for jpeg in empty_frames),
            "spot_detection_rate": round(sum(locate(jpeg, suggested) is not None for jpeg in spot_frames)
                                         / max(len(spot_frames), 1), 3),"spots": len(seen), "error_before_m": [round(float(e), 3) for e in errors],
            "error_after_m": [round(e, 3) for e in after], "fit_rms_px": round(rms, 2),
            "camera": {"position_m": position.round(4).tolist(), "rotation": rot.round(4).tolist(),
                       "fovy_deg": round(fovy, 2)},
            "balloon": {"radius_m": radius, "red_min": suggested.red_min, "red_ratio": suggested.red_ratio,
                        "max_fit_residual": suggested.max_fit_residual,
                        "height_range_m": list(suggested.height_range)}}


def write(path: Path, result: dict) -> None:
    data = json.loads(path.read_text())
    stamp = (f"{date.today().isoformat()} check_camera: {result['spots']} spots, locate error max "
             f"{max(result['error_after_m']) * 100:.1f} cm after the fit (before "
             f"{min(max(result['error_before_m']), 9.99) * 100:.1f} cm), fit {result['fit_rms_px']} px rms")
    data["camera"].update(result["camera"], measured=stamp)
    data["balloon"].update(result["balloon"], measured=stamp + ", no detections in the empty views")
    path.write_text(json.dumps(data, indent=2) + "\n")


def collect(camera, count: int, out: Path, label: str) -> list[bytes]:
    """count fresh frames from the stream, saved as <out>/<label>_<i>.jpg."""
    frames, last = [], None
    deadline = time.monotonic() + 10 + count
    while len(frames) < count:
        if time.monotonic() > deadline:
            raise TimeoutError("Camera stream stalled")
        camera.poll()
        if camera.history and camera.history[-1].received != last:
            last = camera.history[-1].received
            frames.append(camera.history[-1].jpeg)
            (out / f"{label}_{len(frames):02d}.jpg").write_bytes(frames[-1])
        time.sleep(.02)
    return frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-host", required=True)
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--camera-key", default="ego_view")
    p.add_argument("--calibration", type=Path, default=calibration.for_backend("robot"))
    p.add_argument("--balloon-diameter", type=float, required=True, help="Taped diameter of the balloon (m)")
    p.add_argument("--pelvis-height", type=float, default=.79, help="Measured pelvis height above the floor (m)")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--red-object", action="store_true", help="Also check a person in red clothing")
    p.add_argument("--out", type=Path, default=Path(f"check_camera_{time.strftime('%Y%m%d-%H%M%S')}"))
    p.add_argument("--write", action="store_true", help="Store and date the fitted camera and balloon sections")
    a = p.parse_args(argv)
    from gpt_astra.camera import ZmqCamera
    cal = calibration.load(a.calibration)
    a.out.mkdir(parents=True)
    camera = ZmqCamera(a.camera_host, a.camera_port, a.camera_key)
    try:
        empty_frames = []
        steps = [("empty", "Take the balloon OUT of view.")]
        if a.red_object:
            steps.append(("red_object", "Balloon out of view; a person in red clothing stands 0.5-1.5 m in front."))
        for label, text in steps:
            input(f"\n{text} ENTER when ready.")
            empty_frames += collect(camera, a.frames, a.out, label)
            found = sum(locate(jpeg, cal) is not None for jpeg in empty_frames[-a.frames:])
            print(f"  {label}: balloon detected in {found}/{a.frames} frames (must be 0)")
        results, spot_frames = [], []
        for i, spot in enumerate(SPOTS):
            x, y, z = spot
            input(f"\nSpot {i + 1}/{len(SPOTS)}: balloon centre {x:.2f} m ahead of the pelvis, {abs(y):.2f} m to the "
                  f"robot's {'left' if y >= 0 else 'right'}, {a.pelvis_height + z:.2f} m above the floor. ENTER.")
            spot_frames += collect(camera, a.frames, a.out, f"spot{i + 1}")
            results.append(summarize(spot_frames[-a.frames:], cal))
            r = results[-1]
            print(f"  blob in {r.get('blobs', 0)}/{a.frames} frames, located {r.get('located', 0)}; estimate "
                  f"{np.round(r['position'], 3).tolist() if r.get('position') else None}")
    finally:
        camera.close()
    result = analyze(SPOTS, results, cal, a.balloon_diameter / 2, empty_frames, spot_frames)
    (a.out / "result.json").write_text(json.dumps({"spots": SPOTS, "per_spot": results, "result": result},
                                                  indent=2))
    print(json.dumps(result, indent=2))
    failed = result["empty_detections"] or max(result["error_after_m"]) > .03 or result["spot_detection_rate"] < .9
    if result["empty_detections"]:
        print("FAIL: the suggested thresholds 'see' a balloon in an empty view; check the scene and repeat.")
    if result["spot_detection_rate"] < .9:
        print("FAIL: the suggested thresholds miss the balloon in >10% of the spot frames (lighting? balloon colour?).")
    if max(result["error_after_m"]) > .03:
        print("FAIL: >3 cm error at some spot even after the fit; re-measure the spots and the pelvis height.")
    if a.write and not failed:
        write(a.calibration, result)
        print(f"Wrote the camera and balloon sections of {a.calibration}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
