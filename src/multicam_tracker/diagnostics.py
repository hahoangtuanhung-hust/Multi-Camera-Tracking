"""Bounded, optional JSONL diagnostics for identity lifecycle decisions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class IdentityEventWriter:
    """Stream identity events to disk with constant in-process memory."""

    def __init__(
        self,
        path: Path | None,
        *,
        max_events: int = 250_000,
        flush_every: int = 128,
    ) -> None:
        self.path = None if path is None else Path(path)
        self.max_events = max(0, int(max_events))
        self.flush_every = max(1, int(flush_every))
        self.events_written = 0
        self.events_dropped = 0
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8", buffering=64 * 1024)

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    def write(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            return
        if self.max_events and self.events_written >= self.max_events:
            self.events_dropped += 1
            return
        record = {"event": event, **fields}
        self._handle.write(
            json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        self.events_written += 1
        if self.events_written % self.flush_every == 0:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps({
            "event": "diagnostic_summary",
            "events_written": self.events_written,
            "events_dropped": self.events_dropped,
        }) + "\n")
        self._handle.close()
        self._handle = None
