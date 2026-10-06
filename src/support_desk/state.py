"""The single object that flows through the graph.

Agents never call each other. Each node reads the fields it needs, writes its own field,
and returns — the graph decides what runs next. That indirection is what makes the system
testable (inspect state at any point), resumable (Phase 6 serialises this), and parallelisable
(Phase 4 runs two nodes that write different fields).

`trace` uses an additive reducer so concurrent branches append instead of overwriting each
other. Every other field is written by exactly one node, which is why plain overwrite is safe.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from support_desk.schemas import AccountLookup, KnowledgeResult, Resolution, TriageResult


class TicketState(TypedDict):
    ticket_id: str
    ticket_text: str
    customer_email: str
    triage: TriageResult | None
    account: AccountLookup | None
    knowledge: KnowledgeResult | None
    resolution: Resolution | None
    guard: Any | None
    verdict: Any | None
    feedback: Annotated[list[str], operator.add]
    needs_approval: bool
    attempts: int
    trace: Annotated[list[str], operator.add]


def initial_state(ticket_id: str, ticket_text: str, customer_email: str) -> TicketState:
    return TicketState(
        ticket_id=ticket_id,
        ticket_text=ticket_text,
        customer_email=customer_email,
        triage=None,
        account=None,
        knowledge=None,
        resolution=None,
        guard=None,
        verdict=None,
        feedback=[],
        needs_approval=False,
        attempts=0,
        trace=[],
    )
