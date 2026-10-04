"""The graph: who runs, in what order, and on what condition.

The whole topology lives in this one file on purpose. Agents stay ignorant of each other —
routing is a property of the system, not of any agent, and keeping it in one readable place
is what lets you answer "why did this ticket do that?" without reading five prompts.

Phase 3 topology:

    START -> triage -> (account | knowledge | resolver) -> resolver -> END

Routing is deterministic code reading `triage.category`. The model decides the category;
it does not decide the route. Phase 4 makes account and knowledge run in parallel.
"""

from __future__ import annotations

import sqlite3
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from support_desk.agents.account import account_lookup, account_lookup_llm
from support_desk.agents.knowledge import knowledge_lookup
from support_desk.agents.resolver import resolve
from support_desk.agents.triage import triage
from support_desk.observability.budget import BudgetMeter
from support_desk.state import TicketState

_ACCOUNT_CATEGORIES = {"billing", "shipping"}


def triage_node(state: TicketState, llm: Any, meter: BudgetMeter | None = None) -> dict[str, Any]:
    result = triage(state["ticket_text"], meter=meter, llm=llm)
    return {"triage": result, "trace": [f"triage -> {result.category} ({result.confidence:.2f})"]}


def account_node(
    state: TicketState,
    llm: Any | None = None,
    conn: sqlite3.Connection | None = None,
    meter: BudgetMeter | None = None,
) -> dict[str, Any]:
    if llm is None:
        result = account_lookup(state["ticket_text"], state["customer_email"], conn=conn)
    else:
        result = account_lookup_llm(
            state["ticket_text"], state["customer_email"], llm=llm, conn=conn, meter=meter
        )
    return {"account": result, "trace": [f"account -> {result.status}"]}


def knowledge_node(state: TicketState) -> dict[str, Any]:
    category = state["triage"].category if state["triage"] else "unknown"
    result = knowledge_lookup(category)
    return {"knowledge": result, "trace": [f"knowledge -> {result.citations or 'none'}"]}


def resolver_node(state: TicketState, llm: Any, meter: BudgetMeter | None = None) -> dict[str, Any]:
    result = resolve(
        ticket_text=state["ticket_text"],
        triage=state["triage"],
        account=state["account"],
        knowledge=state["knowledge"],
        llm=llm,
        meter=meter,
    )
    return {
        "resolution": result,
        "attempts": state["attempts"] + 1,
        "trace": [f"resolver -> {result.action}"],
    }


def route_after_triage(state: TicketState) -> str:
    """Deterministic: the model chose the category, this code chooses the path."""
    category = state["triage"].category if state["triage"] else "unknown"
    if category in _ACCOUNT_CATEGORIES:
        return "account"
    if category == "product":
        return "knowledge"
    return "resolver"


def build_graph(
    triage_llm: Any,
    resolver_llm: Any,
    account_llm: Any | None = None,
    conn: sqlite3.Connection | None = None,
    meter: BudgetMeter | None = None,
) -> Any:
    """Build the Phase 3 graph. Every model is injected so the whole thing runs offline."""
    builder = StateGraph(TicketState)

    builder.add_node("triage", partial(triage_node, llm=triage_llm, meter=meter))
    builder.add_node("account", partial(account_node, llm=account_llm, conn=conn, meter=meter))
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("resolver", partial(resolver_node, llm=resolver_llm, meter=meter))

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {"account": "account", "knowledge": "knowledge", "resolver": "resolver"},
    )
    builder.add_edge("account", "resolver")
    builder.add_edge("knowledge", "resolver")
    builder.add_edge("resolver", END)

    return builder.compile()
