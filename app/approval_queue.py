"""Approval queue UI.

Run with:  uv run --extra ui streamlit run app/approval_queue.py

Review-only by design. The reviewer approves or rejects; the graph resumes from its
checkpoint and decides what happens next. Nothing here can execute a refund directly.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from support_desk import approvals, config
from support_desk.graph import build_graph
from support_desk.tools import kb_search


@st.cache_resource
def _resources():
    """Resume needs no model: approval routes to the critic, which degrades to approve."""
    checkpointer = approvals.open_checkpointer()
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    graph = build_graph(
        triage_llm=None,
        resolver_llm=None,
        critic_llm=None,
        conn=conn,
        collection=kb_search.build_index(persist=True),
        checkpointer=checkpointer,
    )
    return graph, checkpointer


def _render(pending: approvals.Pending, graph) -> None:
    payload = pending.payload
    amount = payload.get("amount")
    header = f"{payload.get('ticket_id')} — {payload.get('action')}"

    with st.expander(header, expanded=True):
        st.caption(payload.get("customer_email", ""))

        # Why, before what: a reviewer who only sees the action will rubber-stamp it.
        st.subheader("Why this needs you")
        for reason in payload.get("why_flagged", []):
            st.warning(reason)

        order = payload.get("order")
        if order:
            left, right = st.columns(2)
            left.metric("Refund requested", f"${amount:,.2f}" if amount else "—")
            right.metric("Order total", f"${payload.get('order_total', 0):,.2f}")
            st.caption(
                f"Order {order['id']} · placed {order['placed_at']} · {order['status']}"
                f" · basis: {payload.get('basis') or 'n/a'}"
            )

        st.subheader("Customer wrote")
        st.text(payload.get("ticket_text", ""))

        st.subheader("Draft reply")
        st.text(payload.get("reply", ""))

        if payload.get("citations"):
            st.caption("Cites: " + ", ".join(payload["citations"]))

        with st.popover("How the agent got here"):
            for step in payload.get("trace", []):
                st.text(step)

        reason = st.text_input("Reason (required to reject)", key=f"reason-{pending.thread_id}")
        approve_col, reject_col = st.columns(2)

        if approve_col.button("Approve", key=f"ok-{pending.thread_id}", type="primary"):
            approvals.decide(graph, pending.thread_id, approved=True, reason=reason)
            st.rerun()

        if reject_col.button("Reject", key=f"no-{pending.thread_id}"):
            if not reason.strip():
                st.error("A rejection needs a reason — it is fed back to the agent.")
            else:
                approvals.decide(graph, pending.thread_id, approved=False, reason=reason)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Approval queue", layout="centered")
    st.title("Approval queue")

    graph, checkpointer = _resources()
    pending = approvals.list_pending(graph, checkpointer)

    if not pending:
        st.success("Nothing waiting for review.")
        return

    st.caption(f"{len(pending)} ticket(s) paused, waiting on a human.")
    for item in pending:
        _render(item, graph)


if __name__ == "__main__":
    main()
