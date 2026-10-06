"""Resolver agent: decides what to do about a ticket and drafts the reply.

The first agent whose job genuinely needs judgment. Triage classifies, Account looks things
up — both have a defensible deterministic implementation. Choosing between a refund, a
replacement and an escalation, then writing to a customer, does not.

It proposes only. Phase 5's Policy Guard can veto any action it returns, and the approval
threshold is enforced there in code, never here in a prompt.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from support_desk import config
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import AccountLookup, KnowledgeResult, Resolution, TriageResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You resolve customer support tickets for "Lumen", a $249 smart desk lamp.

Choose exactly one action:
- reply_only          answer a question; no account change
- resend_tracking     shipping visibility problem
- replace_unit        a faulty unit under warranty
- refund              money back; you must state the amount
- escalate_to_human   you cannot resolve it safely, or facts are missing/contradictory

Rules:
- Use ONLY the account facts and policy text given to you. Never invent an order, amount or
  policy. If a fact you need is missing, escalate_to_human.
- Never refund more than the order total.
- Cite the policy document filename whenever a policy decides your action.
- Write the reply to the customer directly, plainly, and without promising timelines.

Respond with ONLY this JSON, no prose and no markdown fences:
{"action": {"type": "<action>", "amount": <number or omit>, "basis": "<basis or omit>"},
 "reply": "<text to the customer>",
 "citations": ["<policy-file.md#section>"]}

For a refund you must also give `basis`, one of:
  change_of_mind | duplicate_charge | warranty | expedited_fee
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class _ChatModel(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


def _facts(account: AccountLookup | None) -> str:
    if account is None:
        return "No account lookup was performed."
    lines = [f"lookup_status: {account.status}"]
    if account.detail:
        lines.append(f"lookup_detail: {account.detail}")
    lines.append(f"duplicate_charge: {account.is_duplicate_charge}")
    for label, record in (
        ("customer", account.customer),
        ("order", account.order),
        ("shipment", account.shipment),
    ):
        lines.append(f"{label}: {record.model_dump_json() if record else 'none'}")
    lines.append(f"payments: {[json.loads(p.model_dump_json()) for p in account.payments]}")
    return "\n".join(lines)


def _policy(knowledge: KnowledgeResult | None) -> str:
    if knowledge is None or not knowledge.snippets:
        return "No policy text retrieved."
    return "\n\n".join(
        f"--- {cite} ---\n{snippet}"
        for cite, snippet in zip(knowledge.citations, knowledge.snippets, strict=False)
    )


def _extract_json(text: str) -> dict[str, Any]:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def resolve(
    ticket_text: str,
    triage: TriageResult | None,
    account: AccountLookup | None,
    knowledge: KnowledgeResult | None,
    llm: _ChatModel,
    meter: BudgetMeter | None = None,
    max_retries: int = 2,
    feedback: list[str] | None = None,
) -> Resolution:
    """Propose an action and draft a reply, falling back to escalation if unparsable."""
    context = (
        f"category: {triage.category if triage else 'unknown'}\n"
        f"urgency: {triage.urgency if triage else 'unknown'}\n"
        f"intent: {triage.intent if triage else 'unknown'}\n\n"
        f"ticket:\n{ticket_text}\n\n"
        f"account facts:\n{_facts(account)}\n\n"
        f"policy:\n{_policy(knowledge)}"
    )
    if feedback:
        rejections = "\n".join(f"- {item}" for item in feedback)
        context += f"\n\nA previous attempt was rejected. Fix these and try again:\n{rejections}"

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([("system", _SYSTEM_PROMPT), ("human", context)])
        except Exception as exc:
            logger.warning("resolver call failed: %s", exc)
            break

        content = response.content if isinstance(response.content, str) else str(response.content)
        if meter is not None:
            usage = getattr(response, "usage_metadata", None) or {}
            meter.record(
                agent="resolver",
                model=config.RESOLVER_MODEL,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            )

        try:
            return Resolution.model_validate(_extract_json(content))
        except Exception as exc:
            logger.warning("resolver attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc)

    return Resolution(
        action={"type": "escalate_to_human"},
        reply="This ticket needs a human agent to review it.",
        citations=[],
    )
