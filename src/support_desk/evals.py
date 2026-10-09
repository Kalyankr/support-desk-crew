"""The scorecard: what this system is actually worth, in numbers.

Rows are marked `blocked` rather than filled with plausible-looking figures when they
cannot be measured here. Triage and Resolver quality need a live model; every other row
is deterministic and runs offline. A scorecard that quietly substitutes a scripted model's
output for a real one is worse than no scorecard, because it reads as evidence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from statistics import mean
from types import SimpleNamespace
from typing import Any

from support_desk import config
from support_desk.agents.account import account_lookup
from support_desk.agents.knowledge import knowledge_lookup
from support_desk.data.golden import load_golden
from support_desk.graph import build_graph
from support_desk.guard import policy_guard
from support_desk.observability.budget import BudgetMeter
from support_desk.observability.trace import TicketTrace
from support_desk.schemas import ProposedAction
from support_desk.state import initial_state
from support_desk.tools import kb_search

BLOCKED = "blocked"

# What a correct Resolver would declare. Supplied here because nothing derives it without
# a live model; see tests/test_policy_guard.py for the same caveat.
GOLDEN_BASIS = {
    "T-01": "duplicate_charge",
    "T-02": "duplicate_charge",
    "T-17": "warranty",
    "T-25": "expedited_fee",
}


@dataclass
class Row:
    metric: str
    value: Any
    target: str
    note: str = ""

    @property
    def blocked(self) -> bool:
        return self.value == BLOCKED


@dataclass
class Scorecard:
    rows: list[Row] = field(default_factory=list)
    calls_by_agent: dict[str, float] = field(default_factory=dict)

    def get(self, metric: str) -> Row:
        return next(r for r in self.rows if r.metric == metric)

    def render(self) -> str:
        width = max(len(r.metric) for r in self.rows)
        header = f"{'metric'.ljust(width)}  {'result':>10}  {'target':>10}  note"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            value = row.value if isinstance(row.value, str) else f"{row.value}"
            lines.append(f"{row.metric.ljust(width)}  {value:>10}  {row.target:>10}  {row.note}")
        lines.append("")
        lines.append("model calls per ticket, by agent (what you would pay for):")
        for agent, calls in sorted(self.calls_by_agent.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {agent:<10} {calls:.2f}")
        return "\n".join(lines)


def _scripted(payload: str) -> Any:
    return SimpleNamespace(
        invoke=lambda _m: SimpleNamespace(
            content=payload, usage_metadata={"input_tokens": 100, "output_tokens": 25}
        )
    )


def build_scorecard(conn: sqlite3.Connection, collection: Any) -> Scorecard:
    tickets = load_golden()
    card = Scorecard()

    # --- Account resolution (deterministic) ---------------------------------
    account_hits = 0
    for ticket in tickets:
        result = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        resolved = result.order.id if result.order else None
        account_hits += resolved == ticket.order_id
    card.rows.append(
        Row("account resolution", f"{account_hits / len(tickets):.0%}", ">=90%", "regex + SQL")
    )

    # --- Retrieval recall, topical documents only ---------------------------
    topical = [t for t in tickets if set(t.expected_citations) - set(kb_search.ALWAYS_ON_DOCS)]
    recall_hits = 0
    for ticket in topical:
        found = knowledge_lookup(ticket.text, ticket.expected_category, collection=collection)
        docs = {c.split("#")[0] for c in found.citations}
        recall_hits += (set(ticket.expected_citations) - set(kb_search.ALWAYS_ON_DOCS)).issubset(
            docs
        )
    card.rows.append(
        Row(
            "retrieval recall",
            f"{recall_hits / len(topical):.0%}",
            ">=90%",
            "topical docs; escalation matrix is always-on",
        )
    )

    # --- Hard rules ----------------------------------------------------------
    violations = 0
    unflagged = 0
    for ticket in tickets:
        if ticket.expected_action.type != "refund":
            continue
        account = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        action = ProposedAction(
            type="refund",
            amount=ticket.expected_action.amount,
            basis=GOLDEN_BASIS.get(ticket.id, "change_of_mind"),
        )
        decision = policy_guard.evaluate(action, account, ["refund-policy.md#x"])
        violations += decision.vetoed
        if ticket.expected_approval_required and not decision.needs_approval:
            unflagged += 1
    card.rows.append(Row("hard-rule violations", violations, "0", "guard vs golden refunds"))
    card.rows.append(Row("high risk unflagged", unflagged, "0", "would skip human review"))

    # --- Orchestration: latency and call volume ------------------------------
    latencies: list[float] = []
    calls: list[dict[str, int]] = []
    for ticket in tickets:
        meter = BudgetMeter(ticket.id, max_cost_usd=10.0, max_tokens=10_000_000)
        tracer = TicketTrace(ticket.id)
        graph = build_graph(
            triage_llm=_scripted(
                json.dumps(
                    {
                        "category": ticket.expected_category,
                        "urgency": "medium",
                        "intent": "x",
                        "confidence": 0.9,
                    }
                )
            ),
            resolver_llm=_scripted(
                json.dumps({"action": {"type": "reply_only"}, "reply": "Thanks.", "citations": []})
            ),
            critic_llm=_scripted(json.dumps({"verdict": "approve"})),
            conn=conn,
            collection=collection,
            meter=meter,
            tracer=tracer,
        )
        graph.invoke(initial_state(ticket.id, ticket.text, ticket.customer_email))
        latencies.append(tracer.total_ms)
        per_agent: dict[str, int] = {}
        for entry in meter.entries:
            per_agent[entry.agent] = per_agent.get(entry.agent, 0) + 1
        calls.append(per_agent)

    latencies.sort()
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    card.rows.append(Row("p95 latency (ms)", f"{p95:.0f}", "<20000", "orchestration only"))
    card.rows.append(
        Row("mean model calls", f"{mean(sum(c.values()) for c in calls):.2f}", "-", "per ticket")
    )

    agents = {a for c in calls for a in c}
    card.calls_by_agent = {agent: mean(c.get(agent, 0) for c in calls) for agent in agents}

    # --- Rows that need a live model ----------------------------------------
    card.rows.append(Row("triage accuracy", BLOCKED, ">=85%", "needs a live model"))
    card.rows.append(Row("action accuracy", BLOCKED, ">=85%", "needs a live model"))
    card.rows.append(Row("draft groundedness", BLOCKED, ">=95%", "needs a live model"))

    return card


def main() -> None:
    from support_desk.data import db

    conn = db.build(config.LOCAL_DIR / "scorecard.db")
    collection = kb_search.build_index(persist=False, reset=True)
    print(build_scorecard(conn, collection).render())


if __name__ == "__main__":
    main()
