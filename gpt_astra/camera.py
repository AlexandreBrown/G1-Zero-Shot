"""Fresh, bounded JPEG observations from SONIC's camera-server wire protocol."""
import base64
from collections import deque
from dataclasses import dataclass
import io
import math
import time
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    received: float
    name: str


def jpeg_thumbnail(image):
    image = image.convert("RGB")
    image.thumbnail((640, 480))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80)
    return out.getvalue()


class Frames:
    def __init__(self):
        self.history = deque(maxlen=2)

    def add(self, frame):
        # Retain a short temporal pair, rather than two adjacent 50 Hz frames.
        if len(self.history) == 2 and frame.received - self.history[0].received < .2:
            self.history[-1] = frame
        else:
            self.history.append(frame)

    def latest(self, max_age=.5):
        if not self.history or time.monotonic() - self.history[-1].received > max_age:
            raise TimeoutError("Camera missing/stale; no motion command accepted")
        return list(self.history)


class ZmqCamera(Frames):
    def __init__(self, host, port=5555, key="ego_view"):
        super().__init__()
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
        self.socket.connect(f"tcp://{host}:{port}")
        self.key, self.last_stamp = key, -math.inf

    def poll(self, state=None):
        import msgpack
        try:
            raw = self.socket.recv(self.zmq.NOBLOCK)
        except self.zmq.Again:
            return
        message = msgpack.unpackb(raw, raw=False)
        images, timestamps = message.get("images", {}), message.get("timestamps", {})
        if self.key not in images:
            raise ValueError(f"Camera key {self.key!r} missing; available: {sorted(images)}")
        stamp = timestamps.get(self.key)
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("Camera timestamp missing or invalid")
        # Requires synchronized laptop/robot wall clocks. Reject replayed/cached frames.
        age = time.time() - stamp
        if age > .5 or age < -.5:
            raise ValueError("Camera capture timestamp >0.5s old/future; synchronize clocks/check stream")
        if stamp <= self.last_stamp:
            return
        value = images[self.key]
        if isinstance(value, str):
            value = base64.b64decode(value, validate=True)
        if not isinstance(value, bytes):
            raise ValueError("Expected raw JPEG bytes or base64 JPEG from composed_camera server")
        with Image.open(io.BytesIO(value)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Camera image too large")
            jpeg = jpeg_thumbnail(image)
        self.last_stamp = stamp
        self.add(Frame(jpeg, time.monotonic(), self.key))

    def close(self):
        self.socket.close()
        self.ctx.term()


class MockCamera(Frames):
    """Synthetic scene with command feedback; NOT a physics/contact simulation."""
    def __init__(self, image_path=None):
        super().__init__()
        self.start = time.monotonic()
        self.image = Image.open(image_path).convert("RGB") if image_path else None

    def poll(self, state=None):
        now = time.monotonic()
        if self.history and now-self.history[-1].received < .05:
            return
        if self.image:
            image = self.image.copy()
        else:
            image = Image.new("RGB", (640, 480), "#e5eef5")
            draw = ImageDraw.Draw(image)
            draw.text((12, 12), "MOCK EGO VIEW - no physics / no SONIC inference", fill="black")
            x = int(320 + 140 * math.sin((now-self.start)*.65))
            y = int(200 + 65 * math.sin(now-self.start))
            draw.rectangle((x-30, y-30, x+30, y+30), fill="#497fb5")
            draw.ellipse((130, 350, 220, 420), fill="#b3b8be")
            draw.ellipse((420, 350, 510, 420), fill="#b3b8be")
            draw.text((120, 430), "robot LEFT", fill="black")
            draw.text((420, 430), "robot RIGHT", fill="black")
            draw.text((12, 40), f"command index={(state or {}).get('index', 0)}; "
                      f"frame={(state or {}).get('frame', 0)}", fill="black")
        self.add(Frame(jpeg_thumbnail(image), now, "MOCK ego_view (synthetic, no physics)"))

    def close(self):
        if self.image:
            self.image.close()
