"""Golden-set accuracy check for the Account agent — Phase 2's acceptance bar.

Unlike Phase 1's triage score, this is fully deterministic (regex + SQL, no LLM), so it
runs as a normal pytest assertion rather than a separate manual script.
"""

from __future__ import annotations

import pytest

from support_desk import config
from support_desk.agents.account import account_lookup
from support_desk.data import db
from support_desk.data.golden import load_golden

ACCURACY_THRESHOLD = 0.90


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_account_eval.db")
    yield connection
    connection.close()


def test_account_agent_resolves_correct_order_for_most_tickets(conn):
    tickets = [t for t in load_golden() if t.order_id is not None]
    assert tickets, "golden set must contain tickets with an order id to score against"

    misses = []
    for ticket in tickets:
        result = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        hit = result.order is not None and result.order.id == ticket.order_id
        if not hit:
            misses.append((ticket.id, ticket.order_id, result.status))

    accuracy = (len(tickets) - len(misses)) / len(tickets)
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"account resolution accuracy {accuracy:.1%} below {ACCURACY_THRESHOLD:.0%}: {misses}"
    )
