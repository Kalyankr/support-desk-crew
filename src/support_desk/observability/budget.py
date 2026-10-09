"""Per-ticket token and cost meter.

Exists in Phase 0, before any agent, so that every later phase is measured from the start.
The ceiling is enforced across a whole ticket rather than per call: runaway cost in agent
systems comes from loop count, not from any single oversized request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from support_desk import config


class BudgetExceeded(RuntimeError):
    """Raised when a ticket exhausts its token or dollar ceiling."""


@dataclass(frozen=True)
class Usage:
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        prompt_rate, completion_rate = config.MODEL_PRICING.get(self.model, config.FALLBACK_PRICING)
        return (
            self.prompt_tokens * prompt_rate + self.completion_tokens * completion_rate
        ) / 1_000_000


@dataclass
class BudgetMeter:
    ticket_id: str
    max_cost_usd: float = config.MAX_COST_PER_TICKET_USD
    max_tokens: int = config.MAX_TOKENS_PER_TICKET
    entries: list[Usage] = field(default_factory=list)
    # The graph fans out to concurrent nodes, so appending and checking must not interleave.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, agent: str, model: str, prompt_tokens: int, completion_tokens: int) -> Usage:
        usage = Usage(agent, model, prompt_tokens, completion_tokens)
        with self._lock:
            self.entries.append(usage)
            self._enforce()
        return usage

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def by_agent(self) -> dict[str, float]:
        """Cost attribution — the number that tells you which agent to cut."""
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.agent] = totals.get(entry.agent, 0.0) + entry.cost_usd
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def _enforce(self) -> None:
        if self.total_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"{self.ticket_id}: {self.total_tokens} tokens exceeds cap {self.max_tokens}"
            )
        if self.total_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"{self.ticket_id}: ${self.total_cost_usd:.4f} exceeds cap ${self.max_cost_usd:.4f}"
            )

    def summary(self) -> str:
        lines = [
            f"ticket={self.ticket_id} calls={len(self.entries)} "
            f"tokens={self.total_tokens} cost=${self.total_cost_usd:.4f}"
        ]
        lines.extend(f"  {agent:<12} ${cost:.4f}" for agent, cost in self.by_agent().items())
        return "\n".join(lines)
