"""The approval queue: run tickets, pause on high risk, resume after a human decides.

The checkpointer is the queue. There is no second store to drift out of sync — a pending
approval *is* a thread whose graph stopped at an interrupt, so a crash cannot lose one or
invent one. `decide()` may run in a different process, days later; that is the point.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from support_desk import config
from support_desk.state import initial_state


@dataclass(frozen=True)
class Pending:
    thread_id: str
    payload: dict[str, Any]

    @property
    def ticket_id(self) -> str:
        return str(self.payload.get("ticket_id", self.thread_id))


# Every custom type that can appear inside a checkpointed TicketState.
ALLOWED_MODULES = (
    ("support_desk.schemas", "TriageResult"),
    ("support_desk.schemas", "AccountLookup"),
    ("support_desk.schemas", "CustomerRecord"),
    ("support_desk.schemas", "OrderRecord"),
    ("support_desk.schemas", "PaymentRecord"),
    ("support_desk.schemas", "ShipmentRecord"),
    ("support_desk.schemas", "KnowledgeResult"),
    ("support_desk.schemas", "ProposedAction"),
    ("support_desk.schemas", "Resolution"),
    ("support_desk.guard.policy_guard", "Decision"),
    ("support_desk.agents.critic", "CriticVerdict"),
)


def open_checkpointer(path: Any | None = None) -> Any:
    """A SqliteSaver on a real file, usable across processes.

    The project's own state types are allow-listed explicitly; without this LangGraph warns
    now and will refuse to deserialize them in a future version, which would silently break
    every pending approval on upgrade.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    target = path or config.CHECKPOINT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MODULES))


def thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_payload(snapshot: Any) -> dict[str, Any] | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            return item.value
    return None


def submit(graph: Any, ticket_id: str, text: str, email: str, thread_id: str) -> Pending | None:
    """Run a ticket. Returns a Pending if it stopped for approval, else None."""
    graph.invoke(initial_state(ticket_id, text, email), config=thread(thread_id))
    return pending_for(graph, thread_id)


def pending_for(graph: Any, thread_id: str) -> Pending | None:
    payload = _interrupt_payload(graph.get_state(thread(thread_id)))
    return Pending(thread_id=thread_id, payload=payload) if payload else None


def list_pending(graph: Any, checkpointer: Any) -> list[Pending]:
    """Every thread currently waiting on a human.

    The thread ids are collected before any state is read: `checkpointer.list()` holds the
    saver's lock while it yields, and `get_state()` wants that same lock, so querying inside
    the loop deadlocks.
    """
    thread_ids: list[str] = []
    seen: set[str] = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.add(thread_id)
            thread_ids.append(thread_id)

    found = (pending_for(graph, thread_id) for thread_id in thread_ids)
    return [item for item in found if item is not None]


def decide(graph: Any, thread_id: str, approved: bool, reason: str = "") -> dict[str, Any]:
    """Resume a paused ticket with a human's answer. Safe to call from another process."""
    return graph.invoke(
        Command(resume={"approved": approved, "reason": reason}), config=thread(thread_id)
    )


def final_state(graph: Any, thread_id: str) -> dict[str, Any]:
    return graph.get_state(thread(thread_id)).values
