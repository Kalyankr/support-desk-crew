"""The golden ticket set: fixed inputs with known-correct outcomes.

Written in Phase 0, before any agent exists. Every later phase is scored against it.
The Pydantic models below make the golden set self-checking — a ticket that claims a $249
refund needs no approval will fail to load.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from support_desk import config

Category = Literal["billing", "product", "shipping"]
ActionType = Literal["reply_only", "resend_tracking", "replace_unit", "refund", "escalate_to_human"]


class ExpectedAction(BaseModel):
    type: ActionType
    amount: float | None = None

    @model_validator(mode="after")
    def _amount_matches_type(self) -> ExpectedAction:
        if self.type == "refund" and self.amount is None:
            raise ValueError("refund actions must state an amount")
        if self.type != "refund" and self.amount is not None:
            raise ValueError(f"{self.type} actions must not carry an amount")
        return self

    def __str__(self) -> str:
        return f"refund({self.amount:.2f})" if self.type == "refund" else self.type


class GoldenTicket(BaseModel):
    id: str
    text: str
    customer_email: str
    order_id: str | None = None
    expected_category: Category
    expected_action: ExpectedAction
    expected_approval_required: bool
    expected_citations: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _approval_follows_hard_rule(self) -> GoldenTicket:
        action = self.expected_action
        required = action.type == "refund" and (action.amount or 0) > config.APPROVAL_THRESHOLD_USD
        if required != self.expected_approval_required:
            raise ValueError(
                f"{self.id}: expected_approval_required={self.expected_approval_required} "
                f"contradicts the >${config.APPROVAL_THRESHOLD_USD:.0f} rule for {action}"
            )
        return self


@lru_cache(maxsize=1)
def load_golden() -> tuple[GoldenTicket, ...]:
    raw = json.loads(config.GOLDEN_TICKETS.read_text(encoding="utf-8"))
    return tuple(GoldenTicket.model_validate(item) for item in raw)
