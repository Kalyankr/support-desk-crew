"""Tests for the parameterised account-database tools — no agent, no LLM, pure SQL layer."""

from __future__ import annotations

import pytest

from support_desk import config
from support_desk.data import db
from support_desk.tools import accounts_db


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_accounts_db.db")
    yield connection
    connection.close()


def test_get_customer_found(conn):
    customer = accounts_db.get_customer("ada.mercer@example.com", conn=conn)
    assert customer is not None
    assert customer.id == "C-001"
    assert customer.tier == "standard"


def test_get_customer_not_found(conn):
    assert accounts_db.get_customer("nobody@example.com", conn=conn) is None


def test_get_order_found(conn):
    order = accounts_db.get_order("L-10422", conn=conn)
    assert order is not None
    assert order.total == pytest.approx(249.0)


def test_get_order_not_found(conn):
    assert accounts_db.get_order("L-99999", conn=conn) is None


def test_list_recent_orders_respects_limit_and_ordering(conn):
    customer = accounts_db.get_customer("ada.mercer@example.com", conn=conn)
    orders = accounts_db.list_recent_orders(customer.id, limit=2, conn=conn)
    assert len(orders) <= 2
    if len(orders) == 2:
        assert orders[0].placed_at >= orders[1].placed_at


def test_get_payment_and_shipment_for_known_order(conn):
    payment = accounts_db.get_payment("L-10422", conn=conn)
    shipment = accounts_db.get_shipment("L-10422", conn=conn)
    assert payment is not None
    assert shipment is not None
    assert payment.order_id == shipment.order_id == "L-10422"


def test_cancelled_order_has_no_payment(conn):
    assert accounts_db.get_payment("L-10415", conn=conn) is None


# --- Injection safety ---------------------------------------------------------


def test_sql_injection_payload_is_inert_and_database_survives(conn):
    payload = "x'); DROP TABLE orders; --"

    assert accounts_db.get_customer(payload, conn=conn) is None
    assert accounts_db.get_order(payload, conn=conn) is None
    assert accounts_db.get_payment(payload, conn=conn) is None

    # The payload must never have reached the SQL engine as anything but a literal string.
    counts = db.table_counts(conn)
    assert counts["orders"] == 25
    assert counts["customers"] == 12
