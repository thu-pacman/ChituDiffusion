from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping


class RankTimelineRecorder:
    """Buffer rank-local wall-clock spans and flush once during shutdown."""

    def __init__(
        self,
        *,
        rank: int,
        output_root: str | Path,
        enabled: bool,
    ) -> None:
        self.rank = int(rank)
        self.enabled = bool(enabled)
        self.path = Path(output_root) / f"timeline-rank{self.rank}.jsonl"
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False

    def record(
        self,
        stage: str,
        *,
        start_unix_ns: int,
        end_unix_ns: int,
        request_id: str | None = None,
        lane_ranks: tuple[int, ...] | list[int] | None = None,
        resource: str = "gpu",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        start_ns = int(start_unix_ns)
        end_ns = int(end_unix_ns)
        if end_ns < start_ns:
            raise ValueError("timeline span ends before it starts")
        event: dict[str, Any] = {
            "schema_version": 1,
            "rank": self.rank,
            "stage": str(stage),
            "resource": str(resource),
            "start_unix_ns": start_ns,
            "end_unix_ns": end_ns,
        }
        if request_id is not None:
            event["request_id"] = str(request_id)
        if lane_ranks is not None:
            event["lane_ranks"] = [int(rank) for rank in lane_ranks]
        if metadata:
            event["metadata"] = dict(metadata)
        with self._lock:
            if self._closed:
                return
            self._events.append(event)

    def instant(
        self,
        stage: str,
        *,
        unix_ns: int,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.record(
            stage,
            start_unix_ns=unix_ns,
            end_unix_ns=unix_ns,
            request_id=request_id,
            metadata=metadata,
        )

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            events = sorted(
                self._events,
                key=lambda event: (
                    event["start_unix_ns"],
                    event["end_unix_ns"],
                    event["stage"],
                ),
            )
            self._events = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        temporary.replace(self.path)


__all__ = ["RankTimelineRecorder"]
