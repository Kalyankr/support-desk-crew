"""Phase 0 unit tests: the data and the cost meter, before any agent exists."""

from __future__ import annotations

from datetime import date

import pytest

from support_desk import config
from support_desk.data import db, verify
from support_desk.data.golden import GoldenTicket, load_golden
from support_desk.observability.budget import BudgetExceeded, BudgetMeter


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_support_desk.db")
    yield connection
    connection.close()


# --- Database ----------------------------------------------------------------


def test_seed_loads_expected_volumes(conn):
    counts = db.table_counts(conn)
    assert counts["customers"] == 12
    assert counts["orders"] == 25
    assert counts["payments"] == 26  # 24 orders charged + 2 duplicates
    assert counts["shipments"] == 22


def test_cancelled_order_has_no_payment(conn):
    row = conn.execute("SELECT COUNT(*) AS n FROM payments WHERE order_id = 'L-10415'").fetchone()
    assert row["n"] == 0


def test_duplicate_charges_exist_on_expected_orders(conn):
    rows = conn.execute(
        "SELECT order_id FROM payments GROUP BY order_id HAVING COUNT(*) > 1"
    ).fetchall()
    assert {r["order_id"] for r in rows} == {"L-10420", "L-10422"}


def test_exactly_one_payment_is_already_refunded(conn):
    rows = conn.execute("SELECT order_id FROM payments WHERE refunded = 1").fetchall()
    assert [r["order_id"] for r in rows] == ["L-10407"]


def test_unshipped_orders_have_no_shipment(conn):
    rows = conn.execute(
        """
        SELECT o.id FROM orders o
        LEFT JOIN shipments s ON s.order_id = o.id
        WHERE s.id IS NULL
        """
    ).fetchall()
    assert {r["id"] for r in rows} == {"L-10408", "L-10415", "L-10421"}


def test_refund_windows_are_represented(conn):
    """The seed must contain orders on both sides of both tier windows."""
    rows = conn.execute(
        """
        SELECT o.id, c.tier, julianday(:as_of) - julianday(o.placed_at) AS age_days
        FROM orders o JOIN customers c ON c.id = o.customer_id
        """,
        {"as_of": config.AS_OF.isoformat()},
    ).fetchall()
    ages = [(r["tier"], r["age_days"]) for r in rows]
    assert any(t == "standard" and a <= config.STANDARD_REFUND_WINDOW_DAYS for t, a in ages)
    assert any(t == "standard" and a > config.STANDARD_REFUND_WINDOW_DAYS for t, a in ages)
    assert any(
        t == "plus" and config.STANDARD_REFUND_WINDOW_DAYS < a <= config.PLUS_REFUND_WINDOW_DAYS
        for t, a in ages
    )
    assert any(t == "plus" and a > config.PLUS_REFUND_WINDOW_DAYS for t, a in ages)


def test_tracking_staleness_buckets_are_represented(conn):
    rows = conn.execute(
        """
        SELECT order_id, julianday(:as_of) - julianday(last_scan_at) AS stale_days
        FROM shipments WHERE last_scan_at IS NOT NULL
        """,
        {"as_of": config.AS_OF.isoformat()},
    ).fetchall()
    stale = [r["stale_days"] for r in rows]
    assert any(d < config.TRACKING_STALE_DAYS for d in stale)
    assert any(config.TRACKING_STALE_DAYS <= d < config.TRACKING_LOST_DAYS for d in stale)
    assert any(d >= config.TRACKING_LOST_DAYS for d in stale)


def test_sql_injection_in_ticket_text_is_inert(conn):
    hostile = "'; DROP TABLE orders; --"
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (hostile,)).fetchone()
    assert row is None
    assert db.table_counts(conn)["orders"] == 25


# --- Golden set --------------------------------------------------------------


def test_golden_set_loads_and_is_the_expected_size():
    assert len(load_golden()) == 30


def test_golden_ticket_ids_are_unique():
    ids = [t.id for t in load_golden()]
    assert len(set(ids)) == len(ids)


def test_every_refund_over_threshold_requires_approval():
    for t in load_golden():
        if t.expected_action.type == "refund":
            expected = t.expected_action.amount > config.APPROVAL_THRESHOLD_USD
            assert t.expected_approval_required is expected, t.id


def test_refund_never_exceeds_order_total(conn):
    for t in load_golden():
        if t.expected_action.type != "refund" or not t.order_id:
            continue
        row = conn.execute("SELECT total FROM orders WHERE id = ?", (t.order_id,)).fetchone()
        assert t.expected_action.amount <= row["total"], t.id


def test_golden_set_rejects_an_inconsistent_approval_flag():
    with pytest.raises(ValueError):
        GoldenTicket.model_validate(
            {
                "id": "T-BAD",
                "text": "refund me",
                "customer_email": "ada.mercer@example.com",
                "expected_category": "billing",
                "expected_action": {"type": "refund", "amount": 249.0},
                "expected_approval_required": False,
            }
        )


def test_golden_set_rejects_a_refund_without_an_amount():
    with pytest.raises(ValueError):
        GoldenTicket.model_validate(
            {
                "id": "T-BAD",
                "text": "refund me",
                "customer_email": "ada.mercer@example.com",
                "expected_category": "billing",
                "expected_action": {"type": "refund"},
                "expected_approval_required": False,
            }
        )


def test_verifier_reports_no_problems():
    tickets = load_golden()
    problems = (
        verify.check_policies()
        + verify.check_citations(tickets)
        + verify.check_referential_integrity(tickets)
        + verify.check_coverage(tickets)
    )
    assert problems == []


# --- Budget meter ------------------------------------------------------------


def test_meter_accumulates_cost_and_attributes_it_by_agent():
    meter = BudgetMeter("T-01", max_cost_usd=1.0, max_tokens=100_000)
    meter.record("triage", "openai/gpt-4o-mini", 1_000, 200)
    meter.record("resolver", "openai/gpt-4o", 2_000, 500)
    assert meter.total_tokens == 3_700
    assert meter.total_cost_usd == pytest.approx(0.010270, rel=1e-3)
    assert next(iter(meter.by_agent())) == "resolver"


def test_meter_raises_on_token_overrun():
    meter = BudgetMeter("T-01", max_cost_usd=10.0, max_tokens=1_000)
    with pytest.raises(BudgetExceeded, match="tokens"):
        meter.record("resolver", "openai/gpt-4o-mini", 900, 200)


def test_meter_raises_on_cost_overrun():
    meter = BudgetMeter("T-01", max_cost_usd=0.001, max_tokens=1_000_000)
    with pytest.raises(BudgetExceeded, match=r"\$"):
        meter.record("resolver", "openai/gpt-4o", 10_000, 10_000)


def test_unknown_model_falls_back_to_a_price_rather_than_free():
    meter = BudgetMeter("T-01")
    usage = meter.record("resolver", "some-new-model", 1_000, 1_000)
    assert usage.cost_usd > 0


# --- Config ------------------------------------------------------------------


def test_clock_is_a_fixed_constant():
    assert config.AS_OF == date(2026, 10, 1)
