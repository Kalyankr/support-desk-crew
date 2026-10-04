"""Resolver agent tests — fake model, no network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from support_desk.agents.resolver import resolve
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import AccountLookup, KnowledgeResult, TriageResult


class _FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        self.prompts.append(messages[-1][1])
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            content=response, usage_metadata={"input_tokens": 200, "output_tokens": 50}
        )


_TRIAGE = TriageResult(
    category="billing", urgency="high", intent="duplicate_charge", confidence=0.9
)


def test_parses_a_refund_resolution():
    payload = json.dumps(
        {
            "action": {"type": "refund", "amount": 249.0},
            "reply": "We have refunded the duplicate charge.",
            "citations": ["refund-policy.md"],
        }
    )
    result = resolve("charged twice", _TRIAGE, None, None, llm=_FakeLLM([payload]))
    assert result.action.type == "refund"
    assert result.action.amount == 249.0
    assert result.citations == ["refund-policy.md"]


def test_refund_without_amount_is_rejected_then_retried():
    bad = json.dumps({"action": {"type": "refund"}, "reply": "ok", "citations": []})
    good = json.dumps(
        {"action": {"type": "reply_only"}, "reply": "Here is how it works.", "citations": []}
    )
    result = resolve("question", _TRIAGE, None, None, llm=_FakeLLM([bad, good]))
    assert result.action.type == "reply_only"


def test_non_refund_action_carrying_an_amount_is_rejected():
    bad = json.dumps(
        {"action": {"type": "reply_only", "amount": 50.0}, "reply": "x", "citations": []}
    )
    good = json.dumps({"action": {"type": "reply_only"}, "reply": "x", "citations": []})
    result = resolve("question", _TRIAGE, None, None, llm=_FakeLLM([bad, good]))
    assert result.action.amount is None


def test_falls_back_to_escalation_when_output_never_parses():
    llm = _FakeLLM(["garbage", "still garbage", "more garbage"])
    result = resolve("???", _TRIAGE, None, None, llm=llm, max_retries=2)
    assert result.action.type == "escalate_to_human"


def test_falls_back_to_escalation_when_the_model_errors():
    result = resolve("x", _TRIAGE, None, None, llm=_FakeLLM([RuntimeError("provider down")]))
    assert result.action.type == "escalate_to_human"


def test_account_facts_and_policy_reach_the_prompt():
    account = AccountLookup(status="not_found", detail="no order L-99999")
    knowledge = KnowledgeResult(
        snippets=["Refunds within 30 days."], citations=["refund-policy.md"], stub=True
    )
    llm = _FakeLLM([json.dumps({"action": {"type": "reply_only"}, "reply": "x", "citations": []})])
    resolve("where is L-99999", _TRIAGE, account, knowledge, llm=llm)

    prompt = llm.prompts[0]
    assert "not_found" in prompt
    assert "no order L-99999" in prompt
    assert "Refunds within 30 days." in prompt
    assert "refund-policy.md" in prompt


def test_records_cost():
    meter = BudgetMeter("T-RES")
    payload = json.dumps({"action": {"type": "reply_only"}, "reply": "x", "citations": []})
    resolve("x", _TRIAGE, None, None, llm=_FakeLLM([payload]), meter=meter)
    assert meter.total_tokens == 250
