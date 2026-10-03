"""Structured output contracts shared across agents.

Each agent's output is a Pydantic model, not free text. The schema is the handoff contract
between agents — Phase 3 wires these together without any agent needing to parse prose.
"""

from __future__ import annotations

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
