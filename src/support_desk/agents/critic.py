"""Critic agent: reviews the Resolver's draft before it reaches the customer.

Judgment, not rules — "is this grounded, complete, and on tone?" is exactly the sort of
question that has no deterministic answer, which is why it is a model and the Policy Guard
is not. The Critic cannot approve anything the Guard vetoed; it runs after the Guard, and
its only power is to send work back.

It returns a verdict, never a rewrite: the Resolver owns the draft, so a rejection must come
with feedback specific enough to act on.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from support_desk import config
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import KnowledgeResult, Resolution

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class CriticVerdict(BaseModel):
    verdict: Literal["approve", "revise"]
    grounded: bool = True
    complete: bool = True
    on_tone: bool = True
    feedback: str = Field(default="", max_length=500)


_SYSTEM_PROMPT = """\
You review a draft reply to a customer before it is sent. You do not rewrite it.

Judge three things:
- grounded: every factual or policy claim is supported by the account facts or policy text
  provided. Invented order numbers, amounts, dates or policies are not grounded.
- complete: the reply actually addresses what the customer asked.
- on_tone: plain, direct, no false promises, no blame.

Respond with ONLY this JSON, no prose and no markdown fences:
{"verdict": "approve" | "revise",
 "grounded": true|false, "complete": true|false, "on_tone": true|false,
 "feedback": "<what to fix, one or two sentences; empty if approving>"}

Approve only if all three are true.
"""


class _ChatModel(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


def critique(
    resolution: Resolution,
    knowledge: KnowledgeResult | None,
    llm: _ChatModel | None,
    meter: BudgetMeter | None = None,
    max_retries: int = 1,
) -> CriticVerdict:
    """Approve the draft or send it back. Unparsable output approves, deliberately."""
    if llm is None:
        # Distinct from a failed call: no critic was configured, so there is nothing to report.
        return CriticVerdict(verdict="approve", feedback="no critic configured")

    policy = "\n\n".join(knowledge.snippets) if knowledge and knowledge.snippets else "none"
    context = (
        f"proposed action: {resolution.action}\n"
        f"citations: {resolution.citations}\n\n"
        f"draft reply:\n{resolution.reply}\n\n"
        f"policy available to the resolver:\n{policy}"
    )

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([("system", _SYSTEM_PROMPT), ("human", context)])
        except Exception as exc:
            logger.warning("critic call failed: %s", exc)
            break

        content = response.content if isinstance(response.content, str) else str(response.content)
        if meter is not None:
            usage = getattr(response, "usage_metadata", None) or {}
            meter.record(
                agent="critic",
                model=config.CRITIC_MODEL,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            )

        match = _JSON_OBJECT.search(content)
        if match:
            try:
                return CriticVerdict.model_validate(json.loads(match.group(0)))
            except Exception as exc:
                logger.warning("critic attempt %d failed: %s", attempt + 1, exc)

    # A broken critic must not block a reply the Policy Guard already allowed, or every
    # ticket would escalate on an unrelated outage.
    logger.warning("critic unavailable; approving by default")
    return CriticVerdict(verdict="approve", feedback="critic unavailable")
