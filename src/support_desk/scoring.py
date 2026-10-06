"""Score the Triage agent against the golden set — Phase 1's acceptance check.

Makes real model calls (uses OPENROUTER_API_KEY). Not part of `pytest`: it costs tokens and
depends on an external API, so it stays a manual, explicit step.
"""

from __future__ import annotations

from support_desk import config
from support_desk.agents.triage import triage
from support_desk.data.golden import load_golden
from support_desk.observability.budget import BudgetMeter

ACCURACY_THRESHOLD = 0.85


def main() -> None:
    tickets = load_golden()
    correct = 0
    total_cost = 0.0

    for ticket in tickets:
        meter = BudgetMeter(ticket.id)
        result = triage(ticket.text, meter=meter)
        total_cost += meter.total_cost_usd
        hit = result.category == ticket.expected_category
        correct += int(hit)
        marker = "OK  " if hit else "MISS"
        print(
            f"[{marker}] {ticket.id:<6} expected={ticket.expected_category:<9} "
            f"got={result.category:<9} conf={result.confidence:.2f} "
            f"cost=${meter.total_cost_usd:.5f}"
        )

    accuracy = correct / len(tickets)
    mean_cost = total_cost / len(tickets)
    print()
    print(
        f"accuracy: {correct}/{len(tickets)} = {accuracy:.1%}  (threshold {ACCURACY_THRESHOLD:.0%})"
    )
    print(f"mean cost/ticket: ${mean_cost:.5f}  (cap ${config.MAX_COST_PER_TICKET_USD:.5f})")

    if accuracy < ACCURACY_THRESHOLD:
        raise SystemExit(f"FAIL: accuracy {accuracy:.1%} below {ACCURACY_THRESHOLD:.0%} threshold")
    print("PASS — Phase 1 acceptance met.")


if __name__ == "__main__":
    main()
