"""Policy Guard tests. No model anywhere — these are the rules that must always hold.

The golden-set test here is the one that must be 100%, not 90%: a guard that is usually
right is not a control.
"""

from __future__ import annotations

from datetime import date

import pytest

from support_desk import config
from support_desk.data import db
from support_desk.data.golden import load_golden
from support_desk.guard import policy_guard
from support_desk.schemas import (
    AccountLookup,
    CustomerRecord,
    OrderRecord,
    PaymentRecord,
    ProposedAction,
)


@pytest.fixture(scope="module")
def conn():
    connection = db.build(config.LOCAL_DIR / "test_guard.db")
    yield connection
    connection.close()


def _account(
    *,
    total: float = 249.0,
    age_days: int = 5,
    tier: str = "standard",
    payments: int = 1,
    refunded: bool = False,
) -> AccountLookup:
    placed = date.fromordinal(config.AS_OF.toordinal() - age_days)
    return AccountLookup(
        status="found",
        customer=CustomerRecord(id="C-001", name="A", email="a@example.com", tier=tier),
        order=OrderRecord(
            id="L-10401",
            customer_id="C-001",
            placed_at=placed,
            item="Lumen desk lamp",
            total=total,
            status="delivered",
        ),
        payments=[
            PaymentRecord(
                id=f"P-{i}",
                order_id="L-10401",
                amount=total,
                charged_at=placed,
                refunded=refunded,
            )
            for i in range(payments)
        ],
    )


def _refund(amount: float, basis: str = "change_of_mind") -> ProposedAction:
    return ProposedAction(type="refund", amount=amount, basis=basis)


_CITE = ["refund-policy.md#standard-return-window"]

# What a correct Resolver would declare for each golden refund. Test-supplied on purpose:
# nothing in the system derives this yet, so it is a fixture, not a system output.
GOLDEN_BASIS = {
    "T-01": "duplicate_charge",
    "T-02": "duplicate_charge",
    "T-17": "warranty",
    "T-25": "expedited_fee",
}


# --- Rule: refund ceiling -----------------------------------------------------


def test_refund_may_not_exceed_order_total():
    decision = policy_guard.evaluate(_refund(300.0), _account(total=249.0), _CITE)
    assert decision.vetoed
    assert "exceeds order total" in decision.reasons[0]


def test_partial_refund_is_allowed():
    assert policy_guard.evaluate(_refund(50.0), _account(total=249.0), _CITE).allowed


# --- Rule: already refunded ---------------------------------------------------


def test_already_refunded_order_cannot_be_refunded_again():
    decision = policy_guard.evaluate(_refund(50.0), _account(refunded=True), _CITE)
    assert decision.vetoed
    assert "already refunded" in decision.reasons[0]


# --- Rule: return window, scoped to change of mind ----------------------------


def test_change_of_mind_outside_standard_window_is_refused():
    decision = policy_guard.evaluate(_refund(50.0), _account(age_days=45), _CITE)
    assert decision.vetoed
    assert "outside the 30d standard window" in decision.reasons[0]


def test_plus_tier_gets_the_longer_window():
    assert policy_guard.evaluate(_refund(50.0), _account(age_days=45, tier="plus"), _CITE).allowed


def test_plus_tier_still_bounded_at_sixty_days():
    assert policy_guard.evaluate(_refund(50.0), _account(age_days=70, tier="plus"), _CITE).vetoed


def test_warranty_refund_is_not_bound_by_the_return_window():
    """A fault is not a change of mind; golden T-17 is 134 days old and must still refund."""
    decision = policy_guard.evaluate(
        _refund(50.0, basis="warranty"), _account(age_days=134, tier="plus"), _CITE
    )
    assert decision.allowed


def test_warranty_refund_is_bounded_by_the_warranty_period():
    assert policy_guard.evaluate(
        _refund(50.0, basis="warranty"), _account(age_days=900), _CITE
    ).vetoed


def test_duplicate_charge_is_not_bound_by_the_return_window():
    decision = policy_guard.evaluate(
        _refund(50.0, basis="duplicate_charge"), _account(age_days=200, payments=2), _CITE
    )
    assert decision.allowed


def test_duplicate_claim_is_checked_against_the_record():
    """The model may claim a duplicate; the guard verifies there really are two payments."""
    decision = policy_guard.evaluate(
        _refund(50.0, basis="duplicate_charge"), _account(payments=1), _CITE
    )
    assert decision.vetoed
    assert "one payment" in decision.reasons[0]


def test_refund_without_a_basis_is_refused():
    action = ProposedAction(type="refund", amount=50.0)
    assert policy_guard.evaluate(action, _account(), _CITE).vetoed


# --- Rule: approval threshold -------------------------------------------------


def test_refund_over_threshold_needs_approval():
    decision = policy_guard.evaluate(_refund(150.0), _account(), _CITE)
    assert decision.allowed
    assert decision.needs_approval


def test_refund_at_threshold_does_not_need_approval():
    decision = policy_guard.evaluate(_refund(100.0), _account(), _CITE)
    assert decision.allowed
    assert not decision.needs_approval


# --- Rule: citations ----------------------------------------------------------


def test_refund_without_a_citation_is_refused():
    decision = policy_guard.evaluate(_refund(50.0), _account(), citations=[])
    assert decision.vetoed
    assert "cites none" in decision.reasons[0]


def test_reply_only_needs_no_citation():
    assert policy_guard.evaluate(ProposedAction(type="reply_only"), _account(), []).allowed


# --- Missing facts ------------------------------------------------------------


def test_refund_without_a_resolved_order_is_refused():
    decision = policy_guard.evaluate(_refund(50.0), AccountLookup(status="not_found"), _CITE)
    assert decision.vetoed


def test_refund_on_an_order_with_no_payment_is_refused():
    account = _account(payments=0)
    decision = policy_guard.evaluate(_refund(50.0), account, _CITE)
    assert decision.vetoed
    assert "no payment" in decision.reasons[0]


# --- Phase 5 acceptance -------------------------------------------------------


def test_guard_allows_every_golden_refund_when_the_basis_is_correct(conn):
    """No false vetoes: the guard must never refuse a refund the policy owes.

    Scoped honestly — the basis is supplied by the test, so this proves the guard does not
    block correct behaviour. It does NOT prove the system derives the basis correctly; that
    needs a live Resolver. The opposite direction (a wrong basis is caught) is covered by
    the relabelling tests below.
    """
    from support_desk.agents.account import account_lookup

    wrongly_refused = []
    for ticket in load_golden():
        if ticket.expected_action.type != "refund":
            continue
        account = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        action = ProposedAction(
            type="refund",
            amount=ticket.expected_action.amount,
            basis=GOLDEN_BASIS.get(ticket.id, "change_of_mind"),
        )
        decision = policy_guard.evaluate(action, account, ["refund-policy.md#x"])
        if decision.vetoed:
            wrongly_refused.append((ticket.id, decision.reasons))

    assert not wrongly_refused, f"guard refused refunds the policy owes: {wrongly_refused}"


# --- Basis relabelling: the window must not be escapable by renaming the reason ---


@pytest.mark.parametrize("basis", ["change_of_mind", "warranty", "expedited_fee"])
def test_stale_order_is_never_auto_refunded_whatever_basis_is_claimed(basis):
    """A model picking a different label must not turn a stale order into an automatic refund.

    Before this was enforced, claiming `warranty` or `expedited_fee` on a 200-day-old order
    passed straight through, silently bypassing the return window.
    """
    decision = policy_guard.evaluate(_refund(50.0, basis=basis), _account(age_days=200), _CITE)
    assert decision.vetoed or decision.needs_approval, (
        f"basis={basis} auto-approved a 200d order, bypassing the return window"
    )


def test_expedited_fee_refund_is_capped_at_the_actual_fee():
    decision = policy_guard.evaluate(
        _refund(249.0, basis="expedited_fee"), _account(total=274.0), _CITE
    )
    assert decision.vetoed
    assert "exceeds the $25 fee" in decision.reasons[0]


def test_legitimate_expedited_fee_refund_is_allowed():
    decision = policy_guard.evaluate(
        _refund(25.0, basis="expedited_fee"), _account(total=274.0), _CITE
    )
    assert decision.allowed
    assert not decision.needs_approval


def test_warranty_outside_the_return_window_requires_a_human():
    """Code cannot see whether a unit is faulty, so it does not decide alone."""
    decision = policy_guard.evaluate(_refund(50.0, basis="warranty"), _account(age_days=200), _CITE)
    assert decision.allowed
    assert decision.needs_approval


def test_warranty_inside_the_window_needs_no_extra_approval():
    decision = policy_guard.evaluate(_refund(50.0, basis="warranty"), _account(age_days=10), _CITE)
    assert decision.allowed
    assert not decision.needs_approval


def test_approval_flag_matches_the_golden_set(conn):
    from support_desk.agents.account import account_lookup

    mismatches = []
    for ticket in load_golden():
        if ticket.expected_action.type != "refund":
            continue
        account = account_lookup(ticket.text, ticket.customer_email, conn=conn)
        action = ProposedAction(
            type="refund",
            amount=ticket.expected_action.amount,
            basis=GOLDEN_BASIS.get(ticket.id, "change_of_mind"),
        )
        decision = policy_guard.evaluate(action, account, ["refund-policy.md#x"])
        if decision.needs_approval != ticket.expected_approval_required:
            mismatches.append((ticket.id, decision.needs_approval))

    assert not mismatches, f"approval routing disagrees with the golden set: {mismatches}"
