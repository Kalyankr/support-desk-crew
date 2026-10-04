"""Graph tests: routing, handoffs, and a full offline run over the golden set.

Uses fake models throughout. What is verified here is the *orchestration* — that the right
node runs, that state reaches the next node intact, and that nothing crashes — not the
quality of any model's judgment, which needs live access to measure.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from support_desk import config
from support_desk.data import db
from support_desk.data.golden import load_golden
from support_desk.graph import build_graph, route_after_triage
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import TriageResult
from support_desk.state import initial_state


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_graph.db")
    yield connection
    connection.close()


class _ScriptedLLM:
    """Always answers with the same canned payload, whatever it is asked."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, _messages: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            content=self.payload, usage_metadata={"input_tokens": 100, "output_tokens": 25}
        )


def _triage_llm(category: str) -> _ScriptedLLM:
    return _ScriptedLLM(
        json.dumps({"category": category, "urgency": "medium", "intent": "x", "confidence": 0.9})
    )


def _resolver_llm(action: str = "reply_only") -> _ScriptedLLM:
    return _ScriptedLLM(
        json.dumps({"action": {"type": action}, "reply": "Thanks for writing in.", "citations": []})
    )


# --- Routing is deterministic code, not a model decision ----------------------


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("billing", "account"),
        ("shipping", "account"),
        ("product", "knowledge"),
        ("unknown", "resolver"),
    ],
)
def test_route_after_triage(category, expected):
    state = initial_state("T-X", "text", "a@example.com")
    state["triage"] = TriageResult(category=category, urgency="low", intent="x", confidence=0.5)
    assert route_after_triage(state) == expected


def test_missing_triage_routes_straight_to_resolver():
    state = initial_state("T-X", "text", "a@example.com")
    assert route_after_triage(state) == "resolver"


# --- End-to-end through the compiled graph ------------------------------------


def test_billing_ticket_visits_account_not_knowledge(conn):
    graph = build_graph(triage_llm=_triage_llm("billing"), resolver_llm=_resolver_llm(), conn=conn)
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    assert final["account"] is not None
    assert final["knowledge"] is None
    assert final["account"].order.id == "L-10422"


def test_product_ticket_visits_knowledge_not_account(conn):
    graph = build_graph(triage_llm=_triage_llm("product"), resolver_llm=_resolver_llm(), conn=conn)
    final = graph.invoke(
        initial_state("T-16", "how do I factory reset?", "elena.marchetti@example.com")
    )
    assert final["knowledge"] is not None
    assert final["account"] is None
    assert final["knowledge"].citations == ["warranty-policy.md"]


def test_unknown_category_skips_both_lookups(conn):
    graph = build_graph(
        triage_llm=_triage_llm("unknown"),
        resolver_llm=_resolver_llm("escalate_to_human"),
        conn=conn,
    )
    final = graph.invoke(initial_state("T-X", "???", "ada.mercer@example.com"))
    assert final["account"] is None
    assert final["knowledge"] is None
    assert final["resolution"].action.type == "escalate_to_human"


def test_state_accumulates_a_trace_in_order(conn):
    graph = build_graph(triage_llm=_triage_llm("billing"), resolver_llm=_resolver_llm(), conn=conn)
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    assert len(final["trace"]) == 3
    assert final["trace"][0].startswith("triage ->")
    assert final["trace"][1].startswith("account ->")
    assert final["trace"][2].startswith("resolver ->")


def test_account_facts_are_handed_to_the_resolver(conn):
    resolver_llm = _resolver_llm()
    graph = build_graph(triage_llm=_triage_llm("billing"), resolver_llm=resolver_llm, conn=conn)
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    assert resolver_llm.calls == 1
    assert final["attempts"] == 1


def test_budget_is_metered_across_the_whole_graph(conn):
    meter = BudgetMeter("T-01")
    graph = build_graph(
        triage_llm=_triage_llm("billing"),
        resolver_llm=_resolver_llm(),
        conn=conn,
        meter=meter,
    )
    graph.invoke(initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com"))
    assert {e.agent for e in meter.entries} == {"triage", "resolver"}


# --- Phase 3 acceptance: all 30 tickets, no crashes ---------------------------


def test_every_golden_ticket_runs_end_to_end(conn):
    failures = []
    for ticket in load_golden():
        graph = build_graph(
            triage_llm=_triage_llm(ticket.expected_category),
            resolver_llm=_resolver_llm(),
            conn=conn,
        )
        try:
            final = graph.invoke(initial_state(ticket.id, ticket.text, ticket.customer_email))
        except Exception as exc:
            failures.append((ticket.id, repr(exc)))
            continue
        if final["resolution"] is None:
            failures.append((ticket.id, "no resolution produced"))

    assert not failures, f"graph failed on {len(failures)} tickets: {failures}"
