"""Lossless read-only DDS readers for the G1 topics (used by session_recorder.py and lowstate_monitor.py).

unitree_sdk2py's ChannelSubscriber hands each sample to a Python callback one at a time from a shallow reader,
so at 500 Hz on a busy computer older samples are overwritten before Python takes them (measured: ~10-60 %
loss under load). These readers keep a deep history (KeepLast 4096, best effort like the robot's publishers) and
are drained in batches, so nothing is lost as long as the drain loop runs at least every few seconds.

drain() deserializes every sample with the Python IDL code: about 3 ms per LowState_, so at most ~300 messages/s,
while the real G1 sends ~1040/s (2026-10-21: the recorder kept ~40 % and the monitor saw false 1-2 s gaps).
High-rate readers use drain_raw(), which returns the CDR bytes for scripts/g1_cdr.py to decode in bulk.
"""

from __future__ import annotations

import time

HISTORY_DEPTH = 4096


def open_participant(iface: str, domain: int = 0):
    """A DDS participant bound to ``iface`` through unitree_sdk2py's own domain configuration."""
    from cyclonedds.domain import DomainParticipant
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(domain, iface)
    return DomainParticipant(domain)


class TopicReader:
    """Read-only reader of one topic; never creates a writer."""

    def __init__(self, participant, topic: str, kind) -> None:
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        qos = Qos(Policy.History.KeepLast(HISTORY_DEPTH), Policy.Reliability.BestEffort)
        self.kind = kind
        self.reader = DataReader(participant, Topic(participant, topic, kind), qos=qos)

    def drain(self) -> list[tuple[float, int, object]]:
        """All samples received since the last call: (take time on CLOCK_MONOTONIC, source timestamp ns, sample)."""
        from cyclonedds.internal import InvalidSample

        stamp = time.monotonic()
        out = []
        while True:
            samples = self.reader.take(N=HISTORY_DEPTH)
            for sample in samples:
                if isinstance(sample, InvalidSample):
                    continue
                out.append((stamp, int(sample.sample_info.source_timestamp), sample))
            if len(samples) < HISTORY_DEPTH:
                return out

    def drain_raw(self) -> list[tuple[float, int, bytes, object]]:
        """Like drain(), without deserializing: (take time, source timestamp ns, CDR bytes, sample info)."""
        from cyclonedds._clayer import ddspy_take
        from cyclonedds.core import DDSException

        stamp = time.monotonic()
        out = []
        while True:
            taken = ddspy_take(self.reader._ref, HISTORY_DEPTH)
            if isinstance(taken, int):
                raise DDSException(taken, f"while taking from {self.reader!r}")
            out.extend((stamp, int(info.source_timestamp), data, info) for data, info in taken if info.valid_data)
            if len(taken) < HISTORY_DEPTH:
                return out

    def deserialize(self, data: bytes):
        return self.kind.deserialize(data)
