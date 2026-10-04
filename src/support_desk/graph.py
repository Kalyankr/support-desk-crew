"""The graph: who runs, in what order, and on what condition.

The whole topology lives in this one file on purpose. Agents stay ignorant of each other —
routing is a property of the system, not of any agent, and keeping it in one readable place
is what lets you answer "why did this ticket do that?" without reading five prompts.

Phase 4 topology:

    START -> triage -> { account, knowledge }  (concurrent) -> resolver -> END

Routing is deterministic code reading `triage.category`. The model decides the category;
it does not decide the route. Account and knowledge run in the same superstep and write
disjoint fields, so the fan-in needs no coordination between them.
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


def knowledge_node(state: TicketState, collection: Any | None = None) -> dict[str, Any]:
    category = state["triage"].category if state["triage"] else "unknown"
    result = knowledge_lookup(state["ticket_text"], category, collection=collection)
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


def route_after_triage(state: TicketState) -> list[str]:
    """Deterministic fan-out: the model chose the category, this code chooses the paths.

    Returning a list makes LangGraph run those nodes concurrently. It is safe here because
    account and knowledge write disjoint fields; only `trace` is shared, and it has an
    additive reducer.
    """
    category = state["triage"].category if state["triage"] else "unknown"
    if category == "unknown":
        return ["resolver"]
    return ["account", "knowledge"]


def build_graph(
    triage_llm: Any,
    resolver_llm: Any,
    account_llm: Any | None = None,
    conn: sqlite3.Connection | None = None,
    meter: BudgetMeter | None = None,
    collection: Any | None = None,
) -> Any:
    """Build the graph. Every dependency is injected so the whole thing runs offline."""
    builder = StateGraph(TicketState)

    builder.add_node("triage", partial(triage_node, llm=triage_llm, meter=meter))
    builder.add_node("account", partial(account_node, llm=account_llm, conn=conn, meter=meter))
    builder.add_node("knowledge", partial(knowledge_node, collection=collection))
    builder.add_node("resolver", partial(resolver_node, llm=resolver_llm, meter=meter))

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        ["account", "knowledge", "resolver"],
    )
    builder.add_edge("account", "resolver")
    builder.add_edge("knowledge", "resolver")
    builder.add_edge("resolver", END)

    return builder.compile()
