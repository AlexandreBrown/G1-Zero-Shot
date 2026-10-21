"""Fast numpy decoding of the G1 DDS messages (unitree_hg LowState_, LowCmd_, IMUState_) from raw CDR bytes.

The Python IDL deserializer needs about 3 ms per LowState_ (measured on the deploy laptop), so one Python process
keeps up with at most ~300 messages/s; the real G1 publishes rt/lowstate at ~1040 Hz. These messages are fixed-size
"final" structs in little-endian CDR (4-byte encapsulation header, then every primitive aligned to its own size
relative to the start of the body), so a batch of n messages is an (n, size) byte array and every field is a
strided view of it. tests/test_cdr.py checks the layouts against the official serializer.
"""

from __future__ import annotations

import numpy as np

HEADER = 4

Spec = list[tuple[str, "str | Spec", int]]

MOTOR_STATE: Spec = [
    ("mode", "u1", 1),
    ("q", "f4", 1),
    ("dq", "f4", 1),
    ("ddq", "f4", 1),
    ("tau_est", "f4", 1),
    ("temperature", "i2", 2),
    ("vol", "f4", 1),
    ("sensor", "u4", 2),
    ("motorstate", "u4", 1),
    ("reserve", "u4", 4),
]
IMU_STATE: Spec = [
    ("quaternion", "f4", 4),
    ("gyroscope", "f4", 3),
    ("accelerometer", "f4", 3),
    ("rpy", "f4", 3),
    ("temperature", "i2", 1),
]
LOW_STATE: Spec = [
    ("version", "u4", 2),
    ("mode_pr", "u1", 1),
    ("mode_machine", "u1", 1),
    ("tick", "u4", 1),
    ("imu_state", IMU_STATE, 1),
    ("motor_state", MOTOR_STATE, 35),
    ("wireless_remote", "u1", 40),
    ("reserve", "u4", 4),
    ("crc", "u4", 1),
]
MOTOR_CMD: Spec = [
    ("mode", "u1", 1),
    ("q", "f4", 1),
    ("dq", "f4", 1),
    ("tau", "f4", 1),
    ("kp", "f4", 1),
    ("kd", "f4", 1),
    ("reserve", "u4", 1),
]
LOW_CMD: Spec = [
    ("mode_pr", "u1", 1),
    ("mode_machine", "u1", 1),
    ("motor_cmd", MOTOR_CMD, 35),
    ("reserve", "u4", 4),
    ("crc", "u4", 1),
]


def leaves(spec: Spec, start: int = 0, prefix: str = "") -> tuple[list[tuple[str, str, int, int]], int]:
    """Every primitive field of one struct placed at body offset `start`: (path, dtype, count, offset), and the
    end offset. An array of structs gives one entry per element, with paths "name[k].member"."""
    out = []
    pos = start
    for name, kind, count in spec:
        if isinstance(kind, list):
            for k in range(count):
                path = f"{prefix}{name}[{k}]." if count > 1 else f"{prefix}{name}."
                sub, pos = leaves(kind, pos, path)
                out.extend(sub)
        else:
            size = np.dtype(kind).itemsize
            pos += -pos % size
            out.append((prefix + name, kind, count, pos))
            pos += size * count
    return out, pos


def message_size(spec: Spec) -> int:
    return HEADER + leaves(spec)[1]


def decode(spec: Spec, messages: list[bytes]) -> dict[str, np.ndarray]:
    """Batch-decode serialized messages. Arrays of structs come back per member with the element axis second,
    e.g. LOW_STATE gives "motor_state.q" of shape (n, 35) and "imu_state.quaternion" of shape (n, 4)."""
    size = message_size(spec)
    raw = np.frombuffer(b"".join(messages), dtype=np.uint8).reshape(len(messages), -1)
    if raw.shape[1] != size:
        raise ValueError(f"messages are {raw.shape[1]} bytes, the layout says {size}")
    grouped: dict[str, list[np.ndarray]] = {}
    for path, kind, count, offset in leaves(spec)[0]:
        width = np.dtype(kind).itemsize * count
        column = np.ascontiguousarray(raw[:, HEADER + offset : HEADER + offset + width]).view("<" + kind)
        member = path.split("[")[0] + ("." + path.split("].")[1] if "[" in path else "")
        grouped.setdefault(member, []).append(column if count > 1 else column[:, 0])
    return {name: parts[0] if len(parts) == 1 else np.stack(parts, axis=1) for name, parts in grouped.items()}
