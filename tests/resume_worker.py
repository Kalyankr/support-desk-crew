"""Helper for the cross-process durability test.

Run as a separate OS process so "resume after a crash" is proven, not simulated:
  stage 1 -> submit a high-risk ticket, print the interrupt, exit (process dies mid-ticket)
  stage 2 -> fresh process, same checkpoint file, resume with a human decision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _llm(payload: str):
    return SimpleNamespace(invoke=lambda _m: SimpleNamespace(content=payload, usage_metadata={}))


def _build(db_path: Path, checkpoint_path: Path):
    from support_desk.approvals import open_checkpointer
    from support_desk.data import db
    from support_desk.graph import build_graph
    from support_desk.tools import kb_search

    conn = db.build(db_path)
    collection = kb_search.build_index(persist=False, reset=True)
    triage = _llm(
        json.dumps({"category": "billing", "urgency": "high", "intent": "x", "confidence": 0.9})
    )
    resolver = _llm(
        json.dumps(
            {
                "action": {"type": "refund", "amount": 249.0, "basis": "change_of_mind"},
                "reply": "We will refund your order.",
                "citations": ["refund-policy.md#standard-return-window"],
            }
        )
    )
    critic = _llm(json.dumps({"verdict": "approve"}))
    graph = build_graph(
        triage_llm=triage,
        resolver_llm=resolver,
        critic_llm=critic,
        conn=conn,
        collection=collection,
        checkpointer=open_checkpointer(checkpoint_path),
    )
    return graph


def main() -> None:
    stage, db_path, checkpoint_path, thread_id = (
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        sys.argv[4],
    )
    graph = _build(db_path, checkpoint_path)

    if stage == "submit":
        from support_desk.approvals import submit

        pending = submit(
            graph,
            "T-03",
            "I'd like my money back for order L-10401.",
            "ada.mercer@example.com",
            thread_id,
        )
        print(json.dumps({"paused": pending is not None, "payload": pending.payload}))
        return

    if stage == "resume":
        from support_desk.approvals import decide, final_state

        decide(graph, thread_id, approved=True, reason="verified by agent K")
        state = final_state(graph, thread_id)
        print(
            json.dumps(
                {
                    "action": state["resolution"].action.type,
                    "amount": state["resolution"].action.amount,
                    "approval": state["approval"],
                    "trace": state["trace"],
                }
            )
        )
        return

    raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
