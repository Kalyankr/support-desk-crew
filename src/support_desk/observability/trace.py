"""One structured trace per ticket: which node ran, for how long, and what it produced.

Phase 4 runs two nodes concurrently, so spans are appended under a lock. Without it the
trace would be the one place in the system that quietly loses data under the exact
conditions you most want to debug.

Deliberately not a vendor SDK. A trace is a dict; `to_dict()` is the whole contract, so
pointing it at LangSmith or Phoenix later is a serialisation detail, not a rewrite.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from support_desk import config


@dataclass(frozen=True)
class Span:
    node: str
    started_at: float
    duration_ms: float
    status: str
    detail: str = ""


@dataclass
class TicketTrace:
    ticket_id: str
    spans: list[Span] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _origin: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def span(self, node: str):
        start = time.perf_counter()
        status, detail = "ok", ""
        try:
            yield
        except Exception as exc:
            status, detail = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(node, start, status, detail)

    def _record(self, node: str, start: float, status: str, detail: str) -> None:
        span = Span(
            node=node,
            started_at=round((start - self._origin) * 1000, 2),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            status=status,
            detail=detail,
        )
        with self._lock:
            self.spans.append(span)

    @property
    def total_ms(self) -> float:
        with self._lock:
            if not self.spans:
                return 0.0
            return round(max(s.started_at + s.duration_ms for s in self.spans), 2)

    def nodes(self) -> list[str]:
        with self._lock:
            return [s.node for s in sorted(self.spans, key=lambda s: s.started_at)]

    def failures(self) -> list[Span]:
        with self._lock:
            return [s for s in self.spans if s.status == "error"]

    def to_dict(self, meter: Any | None = None) -> dict[str, Any]:
        with self._lock:
            spans = [asdict(s) for s in sorted(self.spans, key=lambda s: s.started_at)]
        payload: dict[str, Any] = {
            "ticket_id": self.ticket_id,
            "total_ms": self.total_ms,
            "spans": spans,
        }
        if meter is not None:
            payload["cost"] = {
                "calls": len(meter.entries),
                "tokens": meter.total_tokens,
                "usd": round(meter.total_cost_usd, 6),
                "by_agent": meter.by_agent(),
            }
        return payload

    def write(self, directory: Path | None = None, meter: Any | None = None) -> Path:
        target = directory or config.TRACE_DIR
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.ticket_id}.json"
        path.write_text(json.dumps(self.to_dict(meter), indent=2), encoding="utf-8")
        return path
