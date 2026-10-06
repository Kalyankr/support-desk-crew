"""The hard rules. Plain Python, no model, no prompt, no exceptions.

Everything the business cannot afford to get wrong lives here rather than in an instruction
a model is asked to follow. A prompt is a request; this is a control. The Resolver proposes
an action and declares *why*; this module checks that claim against the account record and
either allows it, sends it to the approval queue, or vetoes it.

The refund window is deliberately scoped to change-of-mind returns. A duplicate charge is our
billing error and a warranty failure is a fault, so neither is bounded by the return window —
applying one blanket age rule would wrongly refuse refunds the policy owes (golden ticket
T-17 is a 134-day-old order that must still be refunded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from support_desk import config
from support_desk.schemas import AccountLookup, ProposedAction

_MONTH = 30  # warranty is expressed in months; days-per-month need not be exact for a cap


@dataclass(frozen=True)
class Decision:
    allowed: bool
    needs_approval: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def vetoed(self) -> bool:
        return not self.allowed

    def __str__(self) -> str:
        if self.vetoed:
            return f"veto: {'; '.join(self.reasons)}"
        return "approval required" if self.needs_approval else "allow"


ALLOW = Decision(allowed=True)


def _veto(*reasons: str) -> Decision:
    return Decision(allowed=False, reasons=tuple(reasons))


def _window_days(tier: str) -> int:
    return config.PLUS_REFUND_WINDOW_DAYS if tier == "plus" else config.STANDARD_REFUND_WINDOW_DAYS


def evaluate(
    action: ProposedAction,
    account: AccountLookup | None,
    citations: list[str] | None = None,
    as_of: date | None = None,
) -> Decision:
    """Check a proposed action against every hard rule."""
    today = as_of or config.AS_OF
    citations = citations or []

    if action.type != "refund":
        return _check_citations(action, citations)

    order = account.order if account else None
    if order is None:
        return _veto("refund proposed but no order was resolved")

    payments = account.payments if account else []
    if not payments:
        return _veto(f"order {order.id} has no payment to refund")

    # Rule: never refund an order whose payment is already refunded.
    if all(p.refunded for p in payments):
        return _veto(f"order {order.id} is already refunded")

    amount = action.amount or 0.0

    # Rule: a refund may never exceed the order total.
    if amount > order.total:
        return _veto(f"refund {amount:.2f} exceeds order total {order.total:.2f}")

    if amount <= 0:
        return _veto(f"refund amount must be positive, got {amount:.2f}")

    basis_veto = _check_basis(action, account, order, payments, today)
    if basis_veto is not None:
        return basis_veto

    citation_check = _check_citations(action, citations)
    if citation_check.vetoed:
        return citation_check

    # Rule: any refund over the threshold goes to a human. No tier, tone or instruction
    # in the customer's message can bypass this.
    if amount > config.APPROVAL_THRESHOLD_USD:
        return Decision(
            allowed=True,
            needs_approval=True,
            reasons=(f"refund {amount:.2f} exceeds ${config.APPROVAL_THRESHOLD_USD:.0f}",),
        )

    return ALLOW


def _check_basis(action, account, order, payments, today) -> Decision | None:
    tier = account.customer.tier if account and account.customer else "standard"
    age = (today - order.placed_at).days
    basis = action.basis

    if basis is None:
        return _veto("refund proposed without a stated basis")

    if basis == "duplicate_charge":
        if len(payments) < 2:
            return _veto(f"duplicate charge claimed but order {order.id} has one payment")
        if (action.amount or 0) > max(p.amount for p in payments):
            return _veto("duplicate refund exceeds the duplicated payment")
        return None

    if basis == "warranty":
        if age > config.WARRANTY_MONTHS * _MONTH:
            return _veto(f"order {order.id} is {age}d old, outside the warranty period")
        return None

    if basis == "change_of_mind":
        window = _window_days(tier)
        if age > window:
            return _veto(f"order {order.id} is {age}d old, outside the {window}d {tier} window")
        return None

    if basis == "expedited_fee" and (action.amount or 0) > order.total:
        return _veto("expedited fee refund exceeds the order total")

    return None


def _check_citations(action: ProposedAction, citations: list[str]) -> Decision:
    """A reply that leans on policy must say which policy."""
    if action.type in {"refund", "replace_unit"} and not citations:
        return _veto(f"{action.type} invokes policy but cites none")
    return ALLOW
