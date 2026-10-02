"""Phase 0 gate: rebuild the database and prove the golden set is coherent.

Run with `python -m support_desk.data.verify` or `support-desk-verify`.
"""

from __future__ import annotations

import sys
from collections import Counter

from support_desk import config
from support_desk.data import db
from support_desk.data.golden import GoldenTicket, load_golden

REQUIRED_POLICIES = {
    "refund-policy.md",
    "warranty-policy.md",
    "shipping-sla.md",
    "tone-of-voice.md",
    "escalation-matrix.md",
}


def check_policies() -> list[str]:
    present = {p.name for p in config.POLICY_DIR.glob("*.md")}
    return [f"missing policy doc: {name}" for name in sorted(REQUIRED_POLICIES - present)]


def check_citations(tickets: tuple[GoldenTicket, ...]) -> list[str]:
    present = {p.name for p in config.POLICY_DIR.glob("*.md")}
    return [
        f"{t.id} cites unknown doc: {c}"
        for t in tickets
        for c in t.expected_citations
        if c not in present
    ]


def check_referential_integrity(tickets: tuple[GoldenTicket, ...]) -> list[str]:
    problems: list[str] = []
    with db.build() as conn:
        emails = {r["email"] for r in conn.execute("SELECT email FROM customers")}
        order_ids = {r["id"] for r in conn.execute("SELECT id FROM orders")}
        orphans = conn.execute(
            "SELECT id FROM orders WHERE customer_id NOT IN (SELECT id FROM customers)"
        ).fetchall()
        problems += [f"orphan order: {r['id']}" for r in orphans]

        over_refund = conn.execute(
            """
            SELECT o.id FROM orders o
            JOIN payments p ON p.order_id = o.id
            WHERE p.amount > o.total
            """
        ).fetchall()
        problems += [f"payment exceeds order total: {r['id']}" for r in over_refund]

    for t in tickets:
        if t.customer_email not in emails:
            problems.append(f"{t.id} references unknown customer: {t.customer_email}")
        if t.order_id and t.order_id not in order_ids:
            problems.append(f"{t.id} references unknown order: {t.order_id}")
    return problems


def check_coverage(tickets: tuple[GoldenTicket, ...]) -> list[str]:
    """A golden set that never exercises a branch cannot detect a regression in it."""
    actions = Counter(t.expected_action.type for t in tickets)
    categories = Counter(t.expected_category for t in tickets)
    problems = [f"no ticket exercises action: {a}" for a in config.ACTIONS if not actions[a]]
    problems += [
        f"category {c} has only {categories[c]} tickets (want >= 8)"
        for c in ("billing", "product", "shipping")
        if categories[c] < 8
    ]
    if not any(t.expected_approval_required for t in tickets):
        problems.append("no ticket exercises the approval queue")
    return problems


def main() -> int:
    tickets = load_golden()

    with db.build() as conn:
        counts = db.table_counts(conn)

    print(f"as of          {config.AS_OF}")
    print("database       " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"policy docs    {len(list(config.POLICY_DIR.glob('*.md')))}")
    print(f"golden tickets {len(tickets)}")
    print()
    print("  by category   " + dict(Counter(t.expected_category for t in tickets)).__str__())
    print(
        "  by action     " + dict(Counter(str(t.expected_action.type) for t in tickets)).__str__()
    )
    print(f"  need approval {sum(t.expected_approval_required for t in tickets)}")
    print()

    problems = (
        check_policies()
        + check_citations(tickets)
        + check_referential_integrity(tickets)
        + check_coverage(tickets)
    )
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("OK — Phase 0 foundations are coherent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
