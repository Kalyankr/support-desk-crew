"""Phase 6: interrupts, durable resume, and the approval queue.

The headline test runs two real OS processes — the first submits a high-risk ticket and
exits, the second resumes it from the checkpoint file alone. Simulating that in-process
would prove nothing about durability.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from support_desk import approvals, config
from support_desk.data import db
from support_desk.data.golden import load_golden
from support_desk.graph import build_graph
from support_desk.guard import policy_guard
from support_desk.schemas import ProposedAction
from support_desk.state import initial_state
from support_desk.tools import kb_search

WORKER = Path(__file__).parent / "resume_worker.py"


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_phase6.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def collection():
    return kb_search.build_index(persist=False, reset=True)


class _Scripted:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, _messages: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(content=self.payload, usage_metadata={})


def _triage(category: str = "billing") -> _Scripted:
    return _Scripted(
        json.dumps({"category": category, "urgency": "high", "intent": "x", "confidence": 0.9})
    )


def _resolver(amount: float = 249.0, basis: str = "change_of_mind") -> _Scripted:
    return _Scripted(
        json.dumps(
            {
                "action": {"type": "refund", "amount": amount, "basis": basis},
                "reply": "We will refund your order.",
                "citations": ["refund-policy.md#standard-return-window"],
            }
        )
    )


def _critic() -> _Scripted:
    return _Scripted(json.dumps({"verdict": "approve"}))


def _graph(conn, collection, tmp_path, resolver=None):
    return build_graph(
        triage_llm=_triage(),
        resolver_llm=resolver or _resolver(),
        critic_llm=_critic(),
        conn=conn,
        collection=collection,
        checkpointer=approvals.open_checkpointer(tmp_path / "ckpt.db"),
    )


# --- Interrupt behaviour ------------------------------------------------------


def test_high_risk_refund_pauses_instead_of_executing(conn, collection, tmp_path):
    graph = _graph(conn, collection, tmp_path)
    pending = approvals.submit(
        graph, "T-03", "refund order L-10401 please", "ada.mercer@example.com", "t1"
    )
    assert pending is not None
    assert pending.payload["action"] == "refund(249.00)"


def test_low_risk_action_does_not_pause(conn, collection, tmp_path):
    graph = build_graph(
        triage_llm=_triage(),
        resolver_llm=_Scripted(
            json.dumps({"action": {"type": "reply_only"}, "reply": "Here you go.", "citations": []})
        ),
        critic_llm=_critic(),
        conn=conn,
        collection=collection,
        checkpointer=approvals.open_checkpointer(tmp_path / "ckpt.db"),
    )
    pending = approvals.submit(
        graph, "T-16", "how do I reset it?", "elena.marchetti@example.com", "t2"
    )
    assert pending is None


def test_approval_payload_explains_why_not_just_what(conn, collection, tmp_path):
    """A reviewer must be able to judge the decision, not just rubber-stamp it."""
    graph = _graph(conn, collection, tmp_path)
    pending = approvals.submit(
        graph, "T-03", "refund order L-10401", "ada.mercer@example.com", "t3"
    )
    payload = pending.payload

    assert payload["why_flagged"] == ["refund 249.00 exceeds $100"]
    assert payload["order"]["id"] == "L-10401"
    assert payload["order_total"] == 249.0
    assert payload["basis"] == "change_of_mind"
    assert payload["citations"]
    assert any(t.startswith("resolver ->") for t in payload["trace"])
    assert payload["reply"]


# --- Resolution of a decision -------------------------------------------------


def test_approval_lets_the_refund_proceed(conn, collection, tmp_path):
    graph = _graph(conn, collection, tmp_path)
    approvals.submit(graph, "T-03", "refund order L-10401", "ada.mercer@example.com", "t4")
    approvals.decide(graph, "t4", approved=True, reason="checked the statement")

    state = approvals.final_state(graph, "t4")
    assert state["resolution"].action.type == "refund"
    assert state["approval"] == {"approved": True, "reason": "checked the statement"}


def test_rejection_escalates_and_never_refunds(conn, collection, tmp_path):
    graph = _graph(conn, collection, tmp_path)
    approvals.submit(graph, "T-03", "refund order L-10401", "ada.mercer@example.com", "t5")
    approvals.decide(graph, "t5", approved=False, reason="customer already refunded by bank")

    state = approvals.final_state(graph, "t5")
    assert state["resolution"].action.type == "escalate_to_human"
    assert any("a human rejected" in f for f in state["feedback"])


def test_decided_ticket_leaves_the_queue(conn, collection, tmp_path):
    graph = _graph(conn, collection, tmp_path)
    approvals.submit(graph, "T-03", "refund order L-10401", "ada.mercer@example.com", "t6")
    assert approvals.pending_for(graph, "t6") is not None
    approvals.decide(graph, "t6", approved=True)
    assert approvals.pending_for(graph, "t6") is None


def test_queue_lists_only_waiting_tickets(conn, collection, tmp_path):
    checkpointer = approvals.open_checkpointer(tmp_path / "queue.db")
    graph = build_graph(
        triage_llm=_triage(),
        resolver_llm=_resolver(),
        critic_llm=_critic(),
        conn=conn,
        collection=collection,
        checkpointer=checkpointer,
    )
    approvals.submit(graph, "A", "refund order L-10401", "ada.mercer@example.com", "qa")
    approvals.submit(graph, "B", "refund order L-10401", "ada.mercer@example.com", "qb")
    approvals.decide(graph, "qa", approved=True)

    waiting = {p.thread_id for p in approvals.list_pending(graph, checkpointer)}
    assert waiting == {"qb"}


# --- Phase 6 acceptance: nothing high-risk slips through ----------------------


def test_every_high_risk_golden_refund_reaches_the_queue(conn, collection, tmp_path):
    """Must be all of them. One miss is an unreviewed payout."""
    expected = [t for t in load_golden() if t.expected_approval_required and t.order_id is not None]
    assert expected

    escaped = []
    for ticket in expected:
        graph = build_graph(
            triage_llm=_triage(ticket.expected_category),
            resolver_llm=_resolver(amount=ticket.expected_action.amount, basis="change_of_mind"),
            critic_llm=_critic(),
            conn=conn,
            collection=collection,
            checkpointer=approvals.open_checkpointer(tmp_path / f"hr_{ticket.id}.db"),
        )
        final = graph.invoke(
            initial_state(ticket.id, ticket.text, ticket.customer_email),
            config=approvals.thread(ticket.id),
        )
        paused = approvals.pending_for(graph, ticket.id) is not None
        acted = final.get("resolution") and final["resolution"].action.type == "refund"
        if acted and not paused:
            escaped.append(ticket.id)

    assert not escaped, f"high-risk refunds executed without approval: {escaped}"


def test_guard_flags_every_golden_high_risk_refund(conn):
    """The queue can only be complete if the guard flags these in the first place."""
    from support_desk.agents.account import account_lookup

    basis = {"T-01": "duplicate_charge", "T-02": "duplicate_charge", "T-17": "warranty"}
    missed = []
    for ticket in load_golden():
        if not ticket.expected_approval_required:
            continue
        account = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        action = ProposedAction(
            type="refund",
            amount=ticket.expected_action.amount,
            basis=basis.get(ticket.id, "change_of_mind"),
        )
        decision = policy_guard.evaluate(action, account, ["refund-policy.md#x"])
        if not decision.needs_approval:
            missed.append(ticket.id)

    assert not missed, f"guard failed to flag high-risk refunds: {missed}"


# --- Durability across a real process boundary --------------------------------


@pytest.mark.slow
def test_ticket_survives_a_real_process_death(tmp_path):
    """Two separate OS processes: one submits and exits, another resumes from disk."""
    acc = tmp_path / "acc.db"
    ckpt = tmp_path / "ckpt.db"

    submitted = subprocess.run(
        [sys.executable, str(WORKER), "submit", str(acc), str(ckpt), "T-KILL"],
        capture_output=True,
        text=True,
        check=True,
    )
    first = json.loads(submitted.stdout.strip().splitlines()[-1])
    assert first["paused"] is True
    assert first["payload"]["action"] == "refund(249.00)"

    resumed = subprocess.run(
        [sys.executable, str(WORKER), "resume", str(acc), str(ckpt), "T-KILL"],
        capture_output=True,
        text=True,
        check=True,
    )
    second = json.loads(resumed.stdout.strip().splitlines()[-1])

    assert second["action"] == "refund"
    assert second["approval"] == {"approved": True, "reason": "verified by agent K"}
    # The trace from before the crash is still there, continued by the second process.
    assert second["trace"][0].startswith("triage ->")
    assert "approval -> approved" in second["trace"]
