"""Golden-set accuracy check for the Account agent — Phase 2's acceptance bar.

Deliberately scores ALL 30 tickets, not just the 27 that carry an order id.

The earlier version of this file filtered to `order_id is not None`, which made it
impossible to fail: every one of those 27 tickets embeds its order id verbatim in
`L-#####` form, so a regex for exactly that pattern is guaranteed to match. Worse, the
filter excluded T-30 — the one ticket written specifically to test the not-found path
("the number on my confirmation is L-99999"). Negative cases are where this agent can
actually be wrong, so they belong in the score.
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


def test_account_agent_matches_the_golden_set(conn):
    tickets = load_golden()
    misses = []

    for ticket in tickets:
        result = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        resolved = result.order.id if result.order else None
        if resolved != ticket.order_id:
            misses.append((ticket.id, ticket.order_id, resolved, result.status))

    accuracy = (len(tickets) - len(misses)) / len(tickets)
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"account resolution accuracy {accuracy:.1%} below {ACCURACY_THRESHOLD:.0%}: {misses}"
    )


def test_nonexistent_order_is_not_invented(conn):
    """T-30: the ticket names L-99999, which does not exist. The agent must not invent it."""
    ticket = next(t for t in load_golden() if t.id == "T-30")
    result = account_lookup(ticket.text, ticket.customer_email, conn=conn)
    assert result.order is None
    assert result.status == "not_found"


def test_tickets_without_an_order_id_resolve_to_no_order(conn):
    for ticket in load_golden():
        if ticket.order_id is not None:
            continue
        result = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        assert result.order is None, f"{ticket.id} should resolve no order"
        assert result.status in {"no_order_referenced", "not_found"}
