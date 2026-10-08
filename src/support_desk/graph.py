"""The graph: who runs, in what order, and on what condition.

The whole topology lives in this one file on purpose. Agents stay ignorant of each other —
routing is a property of the system, not of any agent, and keeping it in one readable place
is what lets you answer "why did this ticket do that?" without reading five prompts.

Phase 6 topology:

    START -> triage -> { account, knowledge }  (concurrent) -> resolver -> guard
    guard    -> critic | approval (high risk) | resolver (vetoed) | escalate (out of revisions)
    approval -> critic (approved) | escalate (rejected)      <- interrupts, waits for a human
    critic   -> END    | resolver (revise)    | escalate (out of revisions)

Routing is deterministic code reading `triage.category`. The model decides the category;
it does not decide the route. Account and knowledge run in the same superstep and write
disjoint fields, so the fan-in needs no coordination between them.

Every loop back to the resolver is bounded by `config.MAX_REVISIONS`, and both loops share
the same `attempts` counter — a draft cannot bounce indefinitely between guard and critic.
"""

from __future__ import annotations

import sqlite3
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from support_desk import config
from support_desk.agents.account import account_lookup, account_lookup_llm
from support_desk.agents.critic import critique
from support_desk.agents.knowledge import knowledge_lookup
from support_desk.agents.resolver import resolve
from support_desk.agents.triage import triage
from support_desk.guard import policy_guard
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import ProposedAction, Resolution
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
        feedback=state["feedback"],
    )
    return {
        "resolution": result,
        "attempts": state["attempts"] + 1,
        "trace": [f"resolver -> {result.action}"],
    }


def guard_node(state: TicketState) -> dict[str, Any]:
    resolution = state["resolution"]
    decision = policy_guard.evaluate(
        resolution.action, state["account"], citations=resolution.citations
    )
    update: dict[str, Any] = {
        "guard": decision,
        "needs_approval": decision.needs_approval,
        "trace": [f"guard -> {decision}"],
    }
    if decision.vetoed:
        reasons = "; ".join(decision.reasons)
        update["feedback"] = [f"policy guard refused {resolution.action}: {reasons}"]
    return update


def critic_node(state: TicketState, llm: Any, meter: BudgetMeter | None = None) -> dict[str, Any]:
    verdict = critique(state["resolution"], state["knowledge"], llm=llm, meter=meter)
    update: dict[str, Any] = {"verdict": verdict, "trace": [f"critic -> {verdict.verdict}"]}
    if verdict.verdict == "revise":
        update["feedback"] = [f"critic asked for a revision: {verdict.feedback}"]
    return update


def escalate_node(state: TicketState) -> dict[str, Any]:
    """Terminal safety net: the loop gave up, so a human takes it."""
    return {
        "resolution": Resolution(
            action=ProposedAction(type="escalate_to_human"),
            reply="This ticket needs a human agent to review it.",
            citations=[],
        ),
        "needs_approval": False,
        "trace": ["escalate -> revision limit reached"],
    }


def approval_node(state: TicketState) -> dict[str, Any]:
    """Stop and wait for a person. Execution resumes here, possibly days later.

    The payload is everything a reviewer needs to judge the decision rather than rubber-stamp
    it: the action, why the guard flagged it, the facts it rests on, and the path that got here.
    """
    resolution = state["resolution"]
    decision = state["guard"]
    account = state["account"]
    order = account.order if account else None

    answer = interrupt(
        {
            "ticket_id": state["ticket_id"],
            "ticket_text": state["ticket_text"],
            "customer_email": state["customer_email"],
            "action": str(resolution.action),
            "amount": resolution.action.amount,
            "basis": resolution.action.basis,
            "reply": resolution.reply,
            "citations": resolution.citations,
            "why_flagged": list(decision.reasons) if decision else [],
            "order": order.model_dump(mode="json") if order else None,
            "order_total": order.total if order else None,
            "trace": list(state["trace"]),
        }
    )

    approved = bool(answer.get("approved", False))
    reason = str(answer.get("reason", ""))
    update: dict[str, Any] = {
        "approval": {"approved": approved, "reason": reason},
        "trace": [f"approval -> {'approved' if approved else 'rejected'}"],
    }
    if not approved:
        update["feedback"] = [f"a human rejected {resolution.action}: {reason}"]
    return update


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


def route_after_guard(state: TicketState) -> str:
    """A veto sends the draft back, but only while revisions remain."""
    decision = state["guard"]
    if decision is not None and decision.vetoed:
        return "resolver" if state["attempts"] <= config.MAX_REVISIONS else "escalate"
    if decision is not None and decision.needs_approval:
        return "approval"
    return "critic"


def route_after_approval(state: TicketState) -> str:
    """A rejected action never reaches the customer; a human takes the ticket instead."""
    approval = state["approval"]
    return "critic" if approval and approval.get("approved") else "escalate"


def route_after_critic(state: TicketState) -> str:
    verdict = state["verdict"]
    if verdict is not None and verdict.verdict == "revise":
        return "resolver" if state["attempts"] <= config.MAX_REVISIONS else "escalate"
    return END


def build_graph(
    triage_llm: Any,
    resolver_llm: Any,
    critic_llm: Any | None = None,
    account_llm: Any | None = None,
    conn: sqlite3.Connection | None = None,
    meter: BudgetMeter | None = None,
    collection: Any | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build the graph. Every dependency is injected so the whole thing runs offline."""
    builder = StateGraph(TicketState)

    builder.add_node("triage", partial(triage_node, llm=triage_llm, meter=meter))
    builder.add_node("account", partial(account_node, llm=account_llm, conn=conn, meter=meter))
    builder.add_node("knowledge", partial(knowledge_node, collection=collection))
    builder.add_node("resolver", partial(resolver_node, llm=resolver_llm, meter=meter))
    builder.add_node("guard", guard_node)
    builder.add_node("approval", approval_node)
    builder.add_node("critic", partial(critic_node, llm=critic_llm, meter=meter))
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        ["account", "knowledge", "resolver"],
    )
    builder.add_edge("account", "resolver")
    builder.add_edge("knowledge", "resolver")
    builder.add_edge("resolver", "guard")
    builder.add_conditional_edges(
        "guard", route_after_guard, ["resolver", "approval", "critic", "escalate"]
    )
    builder.add_conditional_edges("approval", route_after_approval, ["critic", "escalate"])
    builder.add_conditional_edges("critic", route_after_critic, ["resolver", "escalate", END])
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=checkpointer)
