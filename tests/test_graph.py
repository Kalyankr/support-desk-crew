"""Graph tests: routing, handoffs, and a full offline run over the golden set.

Uses fake models throughout. What is verified here is the *orchestration* — that the right
node runs, that state reaches the next node intact, and that nothing crashes — not the
quality of any model's judgment, which needs live access to measure.
"""

from __future__ import annotations

import json
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from support_desk import config
from support_desk import graph as graph_mod
from support_desk.data import db
from support_desk.data.golden import load_golden
from support_desk.graph import build_graph, route_after_triage
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import TriageResult
from support_desk.state import TicketState, initial_state
from support_desk.tools import kb_search


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_graph.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def collection():
    return kb_search.build_index(persist=False, reset=True)


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


def _critic_llm() -> _ScriptedLLM:
    return _ScriptedLLM(json.dumps({"verdict": "approve"}))


# --- Routing is deterministic code, not a model decision ----------------------


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("billing", ["account", "knowledge"]),
        ("shipping", ["account", "knowledge"]),
        ("product", ["account", "knowledge"]),
        ("unknown", ["resolver"]),
    ],
)
def test_route_after_triage(category, expected):
    state = initial_state("T-X", "text", "a@example.com")
    state["triage"] = TriageResult(category=category, urgency="low", intent="x", confidence=0.5)
    assert route_after_triage(state) == expected


def test_missing_triage_routes_straight_to_resolver():
    state = initial_state("T-X", "text", "a@example.com")
    assert route_after_triage(state) == ["resolver"]


# --- End-to-end through the compiled graph ------------------------------------


def test_billing_ticket_gathers_both_account_and_policy(conn, collection):
    graph = build_graph(
        triage_llm=_triage_llm("billing"),
        resolver_llm=_resolver_llm(),
        conn=conn,
        collection=collection,
    )
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    assert final["account"].order.id == "L-10422"
    assert final["knowledge"].citations


def test_product_ticket_retrieves_warranty_policy(conn, collection):
    graph = build_graph(
        triage_llm=_triage_llm("product"),
        resolver_llm=_resolver_llm(),
        conn=conn,
        collection=collection,
    )
    final = graph.invoke(
        initial_state("T-14", "the lamp flickers on the warm setting", "ada.mercer@example.com")
    )
    docs = {c.split("#")[0] for c in final["knowledge"].citations}
    assert "warranty-policy.md" in docs


def test_unknown_category_skips_both_lookups(conn, collection):
    graph = build_graph(
        triage_llm=_triage_llm("unknown"),
        resolver_llm=_resolver_llm("escalate_to_human"),
        conn=conn,
        collection=collection,
    )
    final = graph.invoke(initial_state("T-X", "???", "ada.mercer@example.com"))
    assert final["account"] is None
    assert final["knowledge"] is None
    assert final["resolution"].action.type == "escalate_to_human"


def test_both_branches_append_to_the_shared_trace(conn, collection):
    graph = build_graph(
        triage_llm=_triage_llm("billing"),
        resolver_llm=_resolver_llm(),
        critic_llm=_critic_llm(),
        conn=conn,
        collection=collection,
    )
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    # Concurrent writes to `trace` must merge, not overwrite: both branches survive.
    visited = [t.split(" ->")[0] for t in final["trace"]]
    assert visited[0] == "triage"
    assert {"account", "knowledge"} <= set(visited)
    assert visited.index("resolver") > visited.index("account")
    assert visited.index("resolver") > visited.index("knowledge")


def test_resolver_runs_once_after_both_branches_finish(conn, collection):
    resolver_llm = _resolver_llm()
    graph = build_graph(
        triage_llm=_triage_llm("billing"),
        resolver_llm=resolver_llm,
        conn=conn,
        collection=collection,
    )
    final = graph.invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    assert resolver_llm.calls == 1
    assert final["attempts"] == 1


def test_budget_is_metered_across_the_whole_graph(conn, collection):
    meter = BudgetMeter("T-01")
    graph = build_graph(
        triage_llm=_triage_llm("billing"),
        resolver_llm=_resolver_llm(),
        conn=conn,
        meter=meter,
        collection=collection,
    )
    graph.invoke(initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com"))
    assert {e.agent for e in meter.entries} == {"triage", "resolver"}


# --- Phase 4 acceptance: the two lookups really do overlap in time ------------


def test_account_and_knowledge_run_concurrently(conn, collection):
    """Proves overlap directly rather than timing the whole graph, which would be flaky."""
    spans: dict[str, tuple[float, float]] = {}

    def _timed(name: str, inner):
        def node(state):
            start = time.perf_counter()
            result = inner(state)
            time.sleep(0.25)
            spans[name] = (start, time.perf_counter())
            return result

        return node

    builder = StateGraph(TicketState)
    builder.add_node("triage", partial(graph_mod.triage_node, llm=_triage_llm("billing")))
    builder.add_node(
        "account", _timed("account", partial(graph_mod.account_node, llm=None, conn=conn))
    )
    builder.add_node(
        "knowledge",
        _timed("knowledge", partial(graph_mod.knowledge_node, collection=collection)),
    )
    builder.add_node("resolver", partial(graph_mod.resolver_node, llm=_resolver_llm()))
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage", graph_mod.route_after_triage, ["account", "knowledge", "resolver"]
    )
    builder.add_edge("account", "resolver")
    builder.add_edge("knowledge", "resolver")
    builder.add_edge("resolver", END)

    started = time.perf_counter()
    builder.compile().invoke(
        initial_state("T-01", "charged twice for L-10422", "jonas.wexler@example.com")
    )
    total = time.perf_counter() - started

    account_span, knowledge_span = spans["account"], spans["knowledge"]
    overlap = min(account_span[1], knowledge_span[1]) - max(account_span[0], knowledge_span[0])
    serial = (account_span[1] - account_span[0]) + (knowledge_span[1] - knowledge_span[0])

    assert overlap > 0, "account and knowledge did not overlap; they ran sequentially"
    assert total < serial, f"graph took {total:.2f}s, no faster than running both ({serial:.2f}s)"


# --- Phase 3 acceptance: all 30 tickets, no crashes ---------------------------


def test_every_golden_ticket_runs_end_to_end(conn, collection):
    failures = []
    for ticket in load_golden():
        graph = build_graph(
            triage_llm=_triage_llm(ticket.expected_category),
            resolver_llm=_resolver_llm(),
            conn=conn,
            collection=collection,
        )
        try:
            final = graph.invoke(initial_state(ticket.id, ticket.text, ticket.customer_email))
        except Exception as exc:
            failures.append((ticket.id, repr(exc)))
            continue
        if final["resolution"] is None:
            failures.append((ticket.id, "no resolution produced"))

    assert not failures, f"graph failed on {len(failures)} tickets: {failures}"
