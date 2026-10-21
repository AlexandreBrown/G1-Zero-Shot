"""Quaternion helpers (w, x, y, z) shared by the preview; the same conventions as SONIC's deploy program."""

from __future__ import annotations

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def rotation_6d(q: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns, row-major (SONIC's anchor-orientation encoding)."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w)), axis=-1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z)), axis=-1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w)), axis=-1),
        ),
        axis=-2,
    )
    return matrix.reshape(*q.shape[:-1], 6)
