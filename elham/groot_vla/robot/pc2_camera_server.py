#!/usr/bin/env python3
"""Standalone ego-view camera server for the G1's head RealSense, run on PC2 (192.168.123.164).

Publishes the same wire format as gear_sonic's RealSense driver
(gear_sonic/camera/drivers/realsense.py + sensor_server.py), so run_vla_inference.py
can subscribe with --camera-host 192.168.123.164 --camera-port 5555.

Needs only: pyrealsense2, numpy, opencv (cv2), pyzmq, msgpack.

    python3 pc2_camera_server.py [--port 5555] [--fps 30]
"""

import argparse
import base64
import time

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found")
    for device in devices:
        print(f"RealSense: {device.get_info(rs.camera_info.name)} "
              f"serial={device.get_info(rs.camera_info.serial_number)} "
              f"fw={device.get_info(rs.camera_info.firmware_version)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipeline.start(config)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Publishing ego_view {args.width}x{args.height}@{args.fps} on tcp://*:{args.port}")

    sent = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            # Same as gear_sonic ImageUtils.encode_image: the RGB array goes straight into
            # imencode and the client's imdecode returns it in the same channel order.
            _, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            message = {
                "timestamps": {"ego_view": time.time()},
                "images": {"ego_view": base64.b64encode(jpeg).decode("utf-8")},
            }
            try:
                socket.send(msgpack.packb(message, use_bin_type=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            sent += 1
            if sent % 300 == 0:
                print(f"sent {sent} frames")
    finally:
        pipeline.stop()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
