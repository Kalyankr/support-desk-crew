"""Critic tests — fake model, no network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from support_desk.agents.critic import critique
from support_desk.observability.budget import BudgetMeter
from support_desk.schemas import KnowledgeResult, ProposedAction, Resolution


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
            content=response, usage_metadata={"input_tokens": 300, "output_tokens": 40}
        )


_DRAFT = Resolution(
    action=ProposedAction(type="reply_only"),
    reply="We have looked into this for you.",
    citations=["refund-policy.md#standard-return-window"],
)


def test_approves_a_good_draft():
    payload = json.dumps(
        {"verdict": "approve", "grounded": True, "complete": True, "on_tone": True, "feedback": ""}
    )
    verdict = critique(_DRAFT, None, llm=_FakeLLM([payload]))
    assert verdict.verdict == "approve"


def test_requests_revision_with_actionable_feedback():
    payload = json.dumps(
        {
            "verdict": "revise",
            "grounded": False,
            "complete": True,
            "on_tone": True,
            "feedback": "The reply cites an order number that does not appear in the facts.",
        }
    )
    verdict = critique(_DRAFT, None, llm=_FakeLLM([payload]))
    assert verdict.verdict == "revise"
    assert verdict.grounded is False
    assert "order number" in verdict.feedback


def test_draft_and_policy_reach_the_critic():
    knowledge = KnowledgeResult(snippets=["Refunds within 30 days."], citations=["refund.md#w"])
    payload = json.dumps({"verdict": "approve"})
    llm = _FakeLLM([payload])
    critique(_DRAFT, knowledge, llm=llm)

    prompt = llm.prompts[0]
    assert "We have looked into this for you." in prompt
    assert "Refunds within 30 days." in prompt


def test_unparsable_critic_approves_rather_than_blocking():
    """A broken critic must not escalate every ticket; the guard already vetted the action."""
    verdict = critique(_DRAFT, None, llm=_FakeLLM(["garbage", "still garbage"]), max_retries=1)
    assert verdict.verdict == "approve"
    assert "unavailable" in verdict.feedback


def test_critic_outage_approves_rather_than_blocking():
    verdict = critique(_DRAFT, None, llm=_FakeLLM([RuntimeError("down")]))
    assert verdict.verdict == "approve"


def test_records_cost():
    meter = BudgetMeter("T-CRIT")
    critique(_DRAFT, None, llm=_FakeLLM([json.dumps({"verdict": "approve"})]), meter=meter)
    assert meter.total_tokens == 340
