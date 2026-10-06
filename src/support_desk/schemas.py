"""Structured output contracts shared across agents.

Each agent's output is a Pydantic model, not free text. The schema is the handoff contract
between agents — Phase 3 wires these together without any agent needing to parse prose.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["billing", "product", "shipping", "unknown"]
Urgency = Literal["low", "medium", "high"]


class TriageResult(BaseModel):
    """The Triage agent's judgment on a raw ticket. Downstream agents treat this as fact."""

    category: Category
    urgency: Urgency
    intent: str = Field(
        min_length=1, max_length=80, description="short snake_case phrase, e.g. duplicate_charge"
    )
    confidence: float = Field(ge=0.0, le=1.0)


# --- Phase 2: account records ------------------------------------------------

OrderStatus = Literal["placed", "shipped", "delivered", "cancelled"]
Tier = Literal["standard", "plus"]


class CustomerRecord(BaseModel):
    id: str
    name: str
    email: str
    tier: Tier


class OrderRecord(BaseModel):
    id: str
    customer_id: str
    placed_at: date
    item: str
    total: float
    status: OrderStatus


class PaymentRecord(BaseModel):
    id: str
    order_id: str
    amount: float
    charged_at: date
    refunded: bool


class ShipmentRecord(BaseModel):
    id: str
    order_id: str
    carrier: str
    tracking: str
    last_scan_at: date | None


AccountStatus = Literal[
    "found",
    "no_order_referenced",
    "not_found",
    "ambiguous",
    "tool_error",
]


class AccountLookup(BaseModel):
    """What the Account agent resolved for a ticket.

    `found` always means an order was resolved; a ticket that legitimately references no
    order returns `no_order_referenced` so downstream agents never have to null-check to
    tell "nothing to look up" apart from "looked up successfully".
    """

    customer: CustomerRecord | None = None
    order: OrderRecord | None = None
    payments: list[PaymentRecord] = Field(default_factory=list)
    shipment: ShipmentRecord | None = None
    status: AccountStatus
    detail: str = ""

    @property
    def payment(self) -> PaymentRecord | None:
        return self.payments[0] if self.payments else None

    @property
    def is_duplicate_charge(self) -> bool:
        return len(self.payments) > 1


# --- Phase 3: knowledge and resolution ---------------------------------------


class KnowledgeResult(BaseModel):
    """Policy material relevant to a ticket.

    Phase 3 fills this from a hard-coded category->document map (see agents/knowledge.py);
    Phase 4 replaces that with real retrieval. The contract stays the same either way.
    """

    snippets: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    stub: bool = True


ActionType = Literal[
    "reply_only",
    "resend_tracking",
    "replace_unit",
    "refund",
    "escalate_to_human",
]

# Why a refund is owed. The Resolver declares it; the Policy Guard checks the claim against
# the account record, because the refund window applies to a change of mind but not to a
# duplicate charge or a warranty failure.
RefundBasis = Literal[
    "change_of_mind",
    "duplicate_charge",
    "warranty",
    "expedited_fee",
]


class ProposedAction(BaseModel):
    """What the Resolver wants to do. Proposed only — Phase 5's Policy Guard may veto it."""

    type: ActionType
    amount: float | None = None
    basis: RefundBasis | None = None

    @model_validator(mode="after")
    def _amount_matches_type(self) -> ProposedAction:
        if self.type == "refund" and self.amount is None:
            raise ValueError("refund actions must state an amount")
        if self.type != "refund" and self.amount is not None:
            raise ValueError(f"{self.type} actions must not carry an amount")
        if self.type != "refund" and self.basis is not None:
            raise ValueError(f"{self.type} actions must not carry a refund basis")
        return self

    def __str__(self) -> str:
        return f"refund({self.amount:.2f})" if self.type == "refund" else self.type


class Resolution(BaseModel):
    """The Resolver agent's output: an action plus the customer-facing reply."""

    action: ProposedAction
    reply: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
