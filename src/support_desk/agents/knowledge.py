"""Knowledge agent — Phase 3 stub.

Deliberately not retrieval: a fixed category -> policy document map. It exists so the graph
has a real second branch to route into and a real citations contract for the Resolver to
consume. Phase 4 swaps the body for Chroma-backed search; `KnowledgeResult` does not change.
"""

from __future__ import annotations

from support_desk import config
from support_desk.schemas import Category, KnowledgeResult

_CATEGORY_DOCS: dict[str, str] = {
    "billing": "refund-policy.md",
    "product": "warranty-policy.md",
    "shipping": "shipping-sla.md",
}


def knowledge_lookup(category: Category) -> KnowledgeResult:
    doc = _CATEGORY_DOCS.get(category)
    if doc is None:
        return KnowledgeResult()

    path = config.POLICY_DIR / doc
    if not path.exists():
        return KnowledgeResult()

    return KnowledgeResult(snippets=[path.read_text(encoding="utf-8")], citations=[doc])
