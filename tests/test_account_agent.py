"""Account agent tests — deterministic regex + SQL, no LLM, no network."""

from __future__ import annotations

import sqlite3

import pytest

from support_desk import config
from support_desk.agents.account import account_lookup
from support_desk.data import db


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_account_agent.db")
    yield connection
    connection.close()


def test_resolves_single_order_mentioned_in_text(conn):
    result = account_lookup(
        "I was charged twice for order L-10422, please help.",
        "jonas.wexler@example.com",
        conn=conn,
    )
    assert result.status == "found"
    assert result.order is not None
    assert result.order.id == "L-10422"
    assert result.payment is not None
    assert result.shipment is not None


def test_found_with_no_order_id_in_text(conn):
    result = account_lookup(
        "How do I factory reset my Lumen?", "elena.marchetti@example.com", conn=conn
    )
    assert result.status == "found"
    assert result.order is None
    assert result.customer is not None


def test_unknown_customer_is_not_found(conn):
    result = account_lookup("order L-10422 please", "nobody@example.com", conn=conn)
    assert result.status == "not_found"


def test_unknown_order_is_not_found(conn):
    result = account_lookup("order L-99999 please", "jonas.wexler@example.com", conn=conn)
    assert result.status == "not_found"


def test_multiple_order_ids_are_ambiguous(conn):
    result = account_lookup(
        "Was it L-10422 or L-10420 that got charged twice?",
        "jonas.wexler@example.com",
        conn=conn,
    )
    assert result.status == "ambiguous"


def test_order_belonging_to_a_different_customer_is_ambiguous(conn):
    # L-10422 belongs to jonas.wexler, not ada.mercer.
    result = account_lookup(
        "Please check order L-10422 for me.", "ada.mercer@example.com", conn=conn
    )
    assert result.status == "ambiguous"


def test_tool_error_is_reported_not_raised(conn, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("support_desk.agents.account.accounts_db.get_customer", _boom)
    result = account_lookup("order L-10422", "jonas.wexler@example.com", conn=conn)
    assert result.status == "tool_error"
    assert "locked" in result.detail


# --- Injection safety ----------------------------------------------------------


def test_injection_payload_in_ticket_text_is_harmless(conn):
    payload = "'; DROP TABLE orders; --"
    result = account_lookup(
        f"My order never arrived {payload}", "jonas.wexler@example.com", conn=conn
    )
    # The payload matches no order-id pattern, so it is simply never looked up.
    assert result.status == "found"
    assert result.order is None
    counts = db.table_counts(conn)
    assert counts["orders"] == 25
