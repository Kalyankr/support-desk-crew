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

from datetime import date

from pydantic import BaseModel, Field

from support_desk import config
from support_desk.schemas import AccountLookup, OrderRecord, PaymentRecord, ProposedAction

_MONTH = 30  # warranty is expressed in months; days-per-month need not be exact for a cap


class Decision(BaseModel):
    """Pydantic rather than a dataclass so it survives checkpointing unchanged.

    A frozen dataclass round-trips through the checkpoint serializer with `reasons` turned
    from a tuple into a list, which breaks equality after a resume.
    """

    model_config = {"frozen": True}

    allowed: bool
    needs_approval: bool = False
    reasons: tuple[str, ...] = Field(default_factory=tuple)

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

    basis_decision = _check_basis(action, account, order, payments, today)
    if basis_decision is not None and basis_decision.vetoed:
        return basis_decision

    citation_check = _check_citations(action, citations)
    if citation_check.vetoed:
        return citation_check

    approval_reasons: list[str] = []
    if basis_decision is not None and basis_decision.needs_approval:
        approval_reasons.extend(basis_decision.reasons)

    # Rule: any refund over the threshold goes to a human. No tier, tone or instruction
    # in the customer's message can bypass this.
    if amount > config.APPROVAL_THRESHOLD_USD:
        approval_reasons.append(f"refund {amount:.2f} exceeds ${config.APPROVAL_THRESHOLD_USD:.0f}")

    if approval_reasons:
        return Decision(allowed=True, needs_approval=True, reasons=tuple(approval_reasons))

    return ALLOW


def _check_basis(
    action: ProposedAction,
    account: AccountLookup | None,
    order: OrderRecord,
    payments: list[PaymentRecord],
    today: date,
) -> Decision | None:
    """Verify the Resolver's stated reason against the record.

    A basis is a claim, not a fact. Without this, the return window would be trivially
    escapable: relabel a stale change-of-mind return as a warranty claim and it sails through.
    Where code cannot verify the claim at all (a fault is not visible in the database), the
    answer is a human, not a guess.
    """
    tier = account.customer.tier if account and account.customer else "standard"
    age = (today - order.placed_at).days
    amount = action.amount or 0.0
    basis = action.basis

    if basis is None:
        return _veto("refund proposed without a stated basis")

    if basis == "duplicate_charge":
        if len(payments) < 2:
            return _veto(f"duplicate charge claimed but order {order.id} has one payment")
        if amount > max(p.amount for p in payments):
            return _veto("duplicate refund exceeds the duplicated payment")
        return None

    if basis == "warranty":
        if age > config.WARRANTY_MONTHS * _MONTH:
            return _veto(f"order {order.id} is {age}d old, outside the warranty period")
        if age > _window_days(tier):
            # The database cannot show whether the unit is actually faulty, so a warranty
            # claim that also escapes the return window is verified by a person.
            return Decision(
                allowed=True,
                needs_approval=True,
                reasons=(f"warranty claim on a {age}d order needs a human to confirm the fault",),
            )
        return None

    if basis == "change_of_mind":
        window = _window_days(tier)
        if age > window:
            return _veto(f"order {order.id} is {age}d old, outside the {window}d {tier} window")
        return None

    if basis == "expedited_fee" and amount > config.EXPEDITED_FEE_USD:
        return _veto(
            f"expedited fee refund {amount:.2f} exceeds the ${config.EXPEDITED_FEE_USD:.0f} fee"
        )

    return None


def _check_citations(action: ProposedAction, citations: list[str]) -> Decision:
    """A reply that leans on policy must say which policy."""
    if action.type in {"refund", "replace_unit"} and not citations:
        return _veto(f"{action.type} invokes policy but cites none")
    return ALLOW
