"""Structured output contracts shared across agents.

Each agent's output is a Pydantic model, not free text. The schema is the handoff contract
between agents — Phase 3 wires these together without any agent needing to parse prose.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

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
    payment: PaymentRecord | None = None
    shipment: ShipmentRecord | None = None
    status: AccountStatus
    detail: str = ""
