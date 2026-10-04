"""Knowledge agent: retrieves the policy text relevant to a ticket.

Phase 3 mapped category -> whole document. This retrieves the specific `##` sections that
answer the ticket, which matters because the Resolver is told to cite what it relies on: a
citation to `refund-policy.md#approval-threshold` is checkable, a citation to the whole file
is not.

Citations are derived from retrieved chunks, never written by a model, so a claim can always
be traced back to source text. Phase 5's groundedness check depends on that being true.
"""

from __future__ import annotations

import logging
from typing import Any

from support_desk.schemas import Category, KnowledgeResult
from support_desk.tools import kb_search

logger = logging.getLogger(__name__)

_CATEGORY_HINT: dict[str, str] = {
    "billing": "refund payment charge invoice",
    "product": "warranty fault defect replacement",
    "shipping": "delivery tracking carrier late",
}


def knowledge_lookup(
    ticket_text: str,
    category: Category = "unknown",
    k: int = 6,
    collection: Any | None = None,
) -> KnowledgeResult:
    """Retrieve policy relevant to this ticket, plus standing procedural policy."""
    hint = _CATEGORY_HINT.get(category, "")
    query = f"{hint} {ticket_text}".strip()

    try:
        chunks = kb_search.search(query, k=k, collection=collection)
        chunks = chunks + kb_search.always_on_chunks(collection=collection)
    except Exception as exc:
        logger.warning("policy retrieval failed, resolver will see no policy text: %s", exc)
        return KnowledgeResult(stub=False)

    return KnowledgeResult(
        snippets=[c.text for c in chunks],
        citations=[c.chunk_id for c in chunks],
        stub=False,
    )
