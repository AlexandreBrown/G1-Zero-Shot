#!/usr/bin/env python3
"""Preview Linux video devices; use left/right arrows to switch."""

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import time

import cv2
import numpy as np

WINDOW = 'G1 video devices'


def devices():
    return sorted(glob.glob('/dev/video[0-9]*'), key=lambda p: int(p[10:]))


def interface_name(path):
    try:
        name = Path('/sys/class/video4linux', Path(path).name, 'device').resolve() / 'interface'
        return name.read_text().strip()
    except OSError:
        return 'Unknown camera interface'


def capture(path, messages):
    """Keep camera I/O in a process so a stuck device never freezes the UI."""
    camera = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            messages.put(('error', 'Cannot open device (metadata, unavailable, or busy).'))
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                messages.put(('error', 'Device opened, but could not read an image.'))
                return
            ok, encoded = cv2.imencode('.jpg', frame)
            if ok:
                try:
                    messages.put_nowait(('frame', encoded.tobytes()))
                except queue.Full:
                    pass
    finally:
        camera.release()


class Preview:
    def __init__(self, path):
        ctx = mp.get_context('spawn')
        self.messages = ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=capture, args=(path, self.messages), daemon=True)
        self.process.start()
        self.started = time.monotonic()
        self.frame = None
        self.status = 'Opening device...'

    def update(self):
        for _ in range(2):
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == 'frame':
                self.frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.status = ''
            else:
                self.status = value
        if self.frame is None and self.status == 'Opening device...' and time.monotonic() - self.started > 3:
            self.status = 'No image yet; device may be unavailable or busy.'

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.messages.close()
        self.messages.join_thread()


def draw(path, preview, position, count):
    frame = preview.frame
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (640, 480))
    cv2.rectangle(frame, (0, 0), (639, 90), (0, 0, 0), -1)
    lines = [f'{path}  ({position + 1}/{count})', interface_name(path),
             'Left/Right: switch   Q/Esc: quit']
    for i, line in enumerate(lines):
        cv2.putText(frame, line[:68], (10, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if preview.status:
        cv2.putText(frame, preview.status[:65], (10, 235), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, frame)


def probe(path):
    preview = Preview(path)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            preview.update()
            if preview.frame is not None:
                print(f'{path}: frame {preview.frame.shape}')
                return 0
            if not preview.process.is_alive():
                preview.update()
                break
            time.sleep(0.05)
        print(f'{path}: {preview.status}')
        return 1
    finally:
        preview.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=int, help='Check one /dev/video number without opening a window')
    args = parser.parse_args()
    if args.probe is not None:
        return probe(f'/dev/video{args.probe}')
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise SystemExit('No display found. Connect from your laptop with: ssh -X unitree@192.168.123.164')
    available = devices()
    if not available:
        raise SystemExit('No /dev/video devices found.')
    position = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    preview = Preview(available[position])
    try:
        while True:
            preview.update()
            draw(available[position], preview, position, len(available))
            key = cv2.waitKeyEx(30)
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (65361, 2424832, 81, ord('a'), ord('A')):
                step = -1
            elif key in (65363, 2555904, 83, ord('d'), ord('D')):
                step = 1
            else:
                continue
            preview.close()
            position = (position + step) % len(available)
            preview = Preview(available[position])
    finally:
        preview.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
