"""Deliberate failure injection: every dependency fails in turn, one at a time.

The bar is not "nothing goes wrong" — it is that a single failing dependency degrades the
ticket instead of crashing the pipeline, and that the failure is visible in the trace rather
than silently swallowed.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from support_desk import config
from support_desk.data import db
from support_desk.graph import build_graph
from support_desk.observability.trace import TicketTrace
from support_desk.state import initial_state
from support_desk.tools import kb_search

TICKET = ("T-03", "I'd like my money back for order L-10401.", "ada.mercer@example.com")


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_resilience.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def collection():
    return kb_search.build_index(persist=False, reset=True)


class _Scripted:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def invoke(self, _messages: Any) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return SimpleNamespace(content=self.payload, usage_metadata={})


def _triage(category: str = "billing") -> _Scripted:
    return _Scripted(
        json.dumps({"category": category, "urgency": "high", "intent": "x", "confidence": 0.9})
    )


def _resolver(payload: Any = None) -> _Scripted:
    return _Scripted(
        payload
        if payload is not None
        else json.dumps(
            {"action": {"type": "reply_only"}, "reply": "Thanks for writing in.", "citations": []}
        )
    )


def _run(conn, default_collection, **overrides):
    tracer = TicketTrace("T-03")
    graph = build_graph(
        triage_llm=overrides.get("triage", _triage()),
        resolver_llm=overrides.get("resolver", _resolver()),
        critic_llm=overrides.get("critic", _Scripted(json.dumps({"verdict": "approve"}))),
        conn=overrides.get("conn", conn),
        collection=overrides.get("collection", default_collection),
        tracer=tracer,
    )
    final = graph.invoke(initial_state(*TICKET))
    return final, tracer


# --- One dependency fails at a time ------------------------------------------


def test_database_failure_degrades_rather_than_crashes(conn, collection, monkeypatch):
    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("support_desk.agents.account.accounts_db.get_customer", _boom)
    final, _ = _run(conn, collection)

    assert final["account"].status == "tool_error"
    assert final["resolution"] is not None


def test_retrieval_failure_still_produces_a_resolution(conn, collection):
    class _BrokenIndex:
        def query(self, **_kwargs):
            raise RuntimeError("index unavailable")

        def get(self, **_kwargs):
            raise RuntimeError("index unavailable")

    final, _ = _run(conn, collection, collection=_BrokenIndex())

    assert final["knowledge"].citations == []
    assert final["resolution"] is not None


def test_empty_retrieval_is_not_treated_as_policy(conn, collection):
    class _EmptyIndex:
        def query(self, **_kwargs):
            return {"documents": [[]], "metadatas": [[]]}

        def get(self, **_kwargs):
            return {"documents": [], "metadatas": []}

    final, _ = _run(conn, collection, collection=_EmptyIndex())
    assert final["knowledge"].snippets == []
    assert final["resolution"] is not None


def test_triage_outage_falls_back_to_unknown(conn, collection):
    final, _ = _run(conn, collection, triage=_Scripted(RuntimeError("provider down")))
    assert final["triage"].category == "unknown"
    assert final["resolution"] is not None


def test_resolver_outage_escalates(conn, collection):
    final, _ = _run(conn, collection, resolver=_Scripted(RuntimeError("provider down")))
    assert final["resolution"].action.type == "escalate_to_human"


def test_malformed_resolver_output_escalates(conn, collection):
    final, _ = _run(conn, collection, resolver=_Scripted("not json at all"))
    assert final["resolution"].action.type == "escalate_to_human"


def test_critic_outage_does_not_block_an_allowed_reply(conn, collection):
    final, _ = _run(conn, collection, critic=_Scripted(RuntimeError("provider down")))
    assert final["resolution"].action.type == "reply_only"


def test_every_single_point_of_failure_still_reaches_a_terminal_state(conn, collection):
    """No injected failure may leave a ticket without an answer."""
    injections = {
        "triage": {"triage": _Scripted(RuntimeError("down"))},
        "resolver": {"resolver": _Scripted(RuntimeError("down"))},
        "critic": {"critic": _Scripted(RuntimeError("down"))},
        "bad-json": {"resolver": _Scripted("garbage")},
    }
    stuck = []
    for name, override in injections.items():
        final, _ = _run(conn, collection, **override)
        if final.get("resolution") is None:
            stuck.append(name)
    assert not stuck, f"these failures left a ticket unresolved: {stuck}"


# --- Failures must be visible -------------------------------------------------


def test_trace_records_every_node_that_ran(conn, collection):
    _, tracer = _run(conn, collection)
    nodes = tracer.nodes()
    assert nodes[0] == "triage"
    assert {"account", "knowledge", "resolver", "guard", "critic"} <= set(nodes)
    assert tracer.total_ms > 0


def test_trace_is_written_as_json(conn, collection, tmp_path):
    from support_desk.observability.budget import BudgetMeter

    _, tracer = _run(conn, collection)
    meter = BudgetMeter("T-03")
    meter.record("triage", "openai/gpt-4o-mini", 100, 20)

    path = tracer.write(tmp_path, meter=meter)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["ticket_id"] == "T-03"
    assert payload["spans"]
    assert payload["cost"]["calls"] == 1
    assert "triage" in payload["cost"]["by_agent"]


def test_a_node_raising_is_recorded_as_a_failed_span():
    tracer = TicketTrace("T-X")
    with pytest.raises(ValueError), tracer.span("resolver"):
        raise ValueError("boom")

    failures = tracer.failures()
    assert len(failures) == 1
    assert failures[0].node == "resolver"
    assert "ValueError" in failures[0].detail


def test_concurrent_branches_both_appear_in_the_trace(conn, collection):
    """account and knowledge run in parallel; neither may be lost from the trace."""
    _, tracer = _run(conn, collection)
    nodes = tracer.nodes()
    assert nodes.count("account") == 1
    assert nodes.count("knowledge") == 1
