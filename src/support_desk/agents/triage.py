"""Triage agent: one prompt, one model call, Pydantic-validated output.

No tools, no other agents — the smallest unit that still teaches the full loop every later
agent reuses. Invalid or unparsable output is retried up to `max_retries` times, then falls
back to category="unknown" rather than raising: a bad triage must never crash the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from support_desk import config
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import TriageResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the triage agent for a customer support desk that sells "Lumen", a smart desk lamp.
Read the ticket and classify it. Respond with ONLY a single JSON object, no prose, no
markdown fences, matching exactly this shape:

{"category": "billing" | "product" | "shipping" | "unknown",
 "urgency": "low" | "medium" | "high",
 "intent": "<short snake_case phrase, e.g. duplicate_charge>",
 "confidence": <float 0.0-1.0>}

Category guide:
- billing: charges, refunds, payments, invoices.
- product: the lamp itself — defects, functionality, quality.
- shipping: delivery, tracking, carrier issues.
- unknown: genuinely unclear from the text alone.

Base confidence on how unambiguous the ticket is, not on how it should be resolved.
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class _ChatModel(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


def _client() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.TRIAGE_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0,
    )


def _extract_json(text: str) -> dict[str, Any]:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def triage(
    ticket_text: str,
    meter: BudgetMeter | None = None,
    max_retries: int = 2,
    llm: _ChatModel | None = None,
) -> TriageResult:
    """Classify a ticket, retrying on malformed output before falling back to "unknown"."""
    model = llm if llm is not None else _client()

    for _attempt in range(max_retries + 1):
        try:
            response = model.invoke([("system", _SYSTEM_PROMPT), ("human", ticket_text)])
        except Exception as exc:
            # A provider outage must degrade this ticket, not take down the graph.
            logger.warning("triage call failed: %s", exc)
            break

        content = response.content if isinstance(response.content, str) else str(response.content)

        if meter is not None:
            usage = getattr(response, "usage_metadata", None) or {}
            meter.record(
                agent="triage",
                model=config.TRIAGE_MODEL,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            )

        try:
            payload = _extract_json(content)
            return TriageResult.model_validate(payload)
        except Exception as exc:
            logger.warning("triage attempt %d/%d failed: %s", _attempt + 1, max_retries + 1, exc)
            continue

    return TriageResult(category="unknown", urgency="medium", intent="unparsed", confidence=0.0)
