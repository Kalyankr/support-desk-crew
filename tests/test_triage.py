"""Triage agent tests — a fake chat model, no network, no tokens spent.

Proves the retry-then-fallback contract: malformed output is retried, and only truly
exhausted retries fall back to category="unknown".
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from support_desk.agents.triage import triage
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import TriageResult


class _FakeLLM:
    """Replays canned responses in order, one per `invoke` call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def invoke(self, _messages: list[tuple[str, str]]) -> Any:
        content = next(self._responses)
        return SimpleNamespace(
            content=content,
            usage_metadata={"input_tokens": 50, "output_tokens": 20},
        )


def test_triage_parses_valid_json():
    llm = _FakeLLM(
        [
            '{"category": "billing", "urgency": "high", '
            '"intent": "duplicate_charge", "confidence": 0.92}'
        ]
    )
    result = triage("I was charged twice", llm=llm)
    assert result == TriageResult(
        category="billing", urgency="high", intent="duplicate_charge", confidence=0.92
    )


def test_triage_strips_markdown_fences():
    llm = _FakeLLM(
        [
            '```json\n{"category": "shipping", "urgency": "low", '
            '"intent": "late_delivery", "confidence": 0.7}\n```'
        ]
    )
    result = triage("where is my package", llm=llm)
    assert result.category == "shipping"


def test_triage_retries_then_recovers():
    llm = _FakeLLM(
        [
            "not json at all",
            '{"category": "product", "urgency": "medium", "intent": "flicker", "confidence": 0.8}',
        ]
    )
    result = triage("the lamp flickers", llm=llm, max_retries=2)
    assert result.category == "product"


def test_triage_falls_back_to_unknown_after_exhausting_retries():
    llm = _FakeLLM(["garbage", "still garbage", "more garbage"])
    result = triage("???", llm=llm, max_retries=2)
    assert result.category == "unknown"
    assert result.confidence == 0.0


def test_triage_records_cost_on_each_attempt():
    meter = BudgetMeter("T-TEST")
    llm = _FakeLLM(
        [
            '{"category": "billing", "urgency": "low", '
            '"intent": "invoice_question", "confidence": 0.6}'
        ]
    )
    triage("question about my invoice", meter=meter, llm=llm)
    assert meter.total_tokens == 70
    assert len(meter.entries) == 1


def test_triage_rejects_out_of_range_confidence_and_retries():
    llm = _FakeLLM(
        [
            '{"category": "billing", "urgency": "low", "intent": "x", "confidence": 1.5}',
            '{"category": "billing", "urgency": "low", "intent": "x", "confidence": 0.5}',
        ]
    )
    result = triage("ticket", llm=llm, max_retries=2)
    assert result.confidence == 0.5
