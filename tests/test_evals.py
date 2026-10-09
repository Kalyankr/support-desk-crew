"""The scorecard as a regression suite.

Thresholds that are met are asserted. Rows that cannot be measured without a live model are
asserted to be *honestly marked* blocked — so nobody can quietly fill them with a scripted
model's output and call it a pass.
"""

from __future__ import annotations

import pytest

from support_desk import config, evals
from support_desk.data import db
from support_desk.tools import kb_search


@pytest.fixture(scope="module")
def scorecard():
    conn = db.build(config.LOCAL_DIR / "test_evals.db")
    collection = kb_search.build_index(persist=False, reset=True)
    try:
        yield evals.build_scorecard(conn, collection)
    finally:
        conn.close()


def test_scorecard_fits_on_one_screen(scorecard, capsys):
    rendered = scorecard.render()
    print(rendered)
    assert len(rendered.splitlines()) <= 24
    assert "metric" in rendered


def test_no_hard_rule_violations(scorecard):
    assert scorecard.get("hard-rule violations").value == 0


def test_no_high_risk_action_skips_review(scorecard):
    assert scorecard.get("high risk unflagged").value == 0


def test_account_resolution_meets_target(scorecard):
    assert scorecard.get("account resolution").value == "100%"


def test_retrieval_recall_meets_target(scorecard):
    assert scorecard.get("retrieval recall").value == "100%"


def test_orchestration_latency_is_well_inside_budget(scorecard):
    assert float(scorecard.get("p95 latency (ms)").value) < 20_000


def test_unmeasurable_rows_are_marked_blocked_not_guessed(scorecard):
    """Guards against a future change that fakes these with scripted-model output."""
    for metric in ("triage accuracy", "action accuracy", "draft groundedness"):
        assert scorecard.get(metric).blocked, f"{metric} must stay honest until a model is wired"


def test_cost_is_attributed_per_agent(scorecard):
    """You cannot decide which agent to cut without knowing what each one costs."""
    assert set(scorecard.calls_by_agent) == {"triage", "resolver", "critic"}
    assert all(v > 0 for v in scorecard.calls_by_agent.values())


def test_account_agent_costs_no_model_calls(scorecard):
    """Phase 2's finding, now a regression test: resolving an order needs no LLM."""
    assert "account" not in scorecard.calls_by_agent
