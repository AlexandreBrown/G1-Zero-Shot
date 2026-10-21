"""Readers for the files the deploy program and sonic_run.sh write into a trial folder."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_COLUMNS = 5


@dataclass(frozen=True)
class Logged:
    """One state-logger CSV: entry index, monotonic time (s) and the data columns."""

    index: np.ndarray
    t_monotonic: np.ndarray
    t_realtime: np.ndarray
    data: np.ndarray


def is_trial(path: Path) -> bool:
    """A trial folder with at least one logged control tick (a launch stopped before ] has empty CSVs)."""
    q = path / "state" / "q.csv"
    return q.is_file() and len(q.read_text().splitlines()) > 1


def read_logged(path: Path) -> Logged:
    """index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,<data...> (deploy FileSink format)."""
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return Logged(
        index=table[:, 0].astype(np.int64),
        t_monotonic=table[:, 3] / 1000.0,
        t_realtime=table[:, 2] / 1000.0,
        data=table[:, META_COLUMNS:],
    )


def read_motion_names(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    return np.array([int(r[0]) for r in rows], dtype=np.int64), [r[-1] for r in rows]


def read_unindexed(path: Path) -> np.ndarray:
    """Header-less CSV rows with a trailing comma (--target-motion-logfile, --policy-input-logfile)."""
    rows = []
    for line in path.read_text().splitlines():
        values = [v for v in line.strip().rstrip(",").split(",") if v != ""]
        if values:
            rows.append([float(v) for v in values])
    return np.array(rows, dtype=np.float64)


def align(reference_index: np.ndarray, logged: Logged, fill: float = np.nan) -> np.ndarray:
    """Rows of ``logged`` placed at the positions of ``reference_index`` (missing entries = ``fill``)."""
    out = np.full((len(reference_index), logged.data.shape[1]), fill)
    position = {int(i): k for k, i in enumerate(reference_index)}
    for row, i in enumerate(logged.index):
        k = position.get(int(i))
        if k is not None:
            out[k] = logged.data[row]
    return out
