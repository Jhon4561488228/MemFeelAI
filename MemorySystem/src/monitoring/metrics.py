"""
Lightweight in-memory metrics for AIRI Memory System.

Not production-grade, but sufficient for local visibility and tests.
"""

from typing import Dict, List
from time import time

_counters: Dict[str, int] = {}
_gauges: Dict[str, float] = {}
_started_at: float = time()
_events: List[Dict[str, object]] = []  # recent add/search events
_events_max = 200

def inc(name: str, value: int = 1) -> None:
    _counters[name] = _counters.get(name, 0) + value

def set_gauge(name: str, value: float) -> None:
    _gauges[name] = float(value)

def snapshot() -> Dict[str, object]:
    return {
        "counters": dict(_counters),
        "gauges": dict(_gauges),
        "uptime_sec": time() - _started_at,
        "recent_events": list(_events),
    }

def time_block(metric_name: str):
    """Context manager to time blocks and increment a counter."""
    class _Timer:
        def __enter__(self):
            self._start = time()
            return self
        def __exit__(self, exc_type, exc, tb):
            duration = time() - self._start
            set_gauge(f"{metric_name}_seconds", duration)
            inc(f"{metric_name}_total")
    return _Timer()

def record_event(kind: str, data: Dict[str, object]) -> None:
    """Append a recent event (add/search/etc.) to the ring buffer."""
    data = dict(data)
    data["kind"] = kind
    data["ts"] = time()
    _events.append(data)
    if len(_events) > _events_max:
        del _events[: len(_events) - _events_max]

def merge_counters(prefix: str, other: Dict[str, int]) -> None:
    """Слить внешние счётчики под префиксом (для объединения legacy/новых)."""
    for k, v in (other or {}).items():
        _counters[f"{prefix}.{k}"] = _counters.get(f"{prefix}.{k}", 0) + int(v)
