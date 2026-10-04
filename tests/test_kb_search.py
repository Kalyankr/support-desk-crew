"""Retrieval tests: chunking, citation precision, and recall against the golden set.

Embeddings run locally through Chroma's bundled ONNX model, so these are offline — but they
are not deterministic in the way SQL is. Assertions target behaviour that should hold for any
reasonable embedding (the right document is retrieved), not exact scores or orderings.
"""

from __future__ import annotations

import pytest

from support_desk.agents.knowledge import knowledge_lookup
from support_desk.data.golden import load_golden
from support_desk.tools import kb_search


@pytest.fixture(scope="module")
def collection():
    return kb_search.build_index(persist=False, reset=True)


# --- Chunking -----------------------------------------------------------------


def test_chunks_split_on_markdown_headings():
    text = "# Title\n\n## First rule\n\nBody one.\n\n## Second rule\n\nBody two.\n"
    chunks = kb_search.chunk_markdown(text, "demo.md")
    assert [c.heading for c in chunks] == ["First rule", "Second rule"]
    assert "Body one." in chunks[0].text
    assert "Body two." not in chunks[0].text


def test_chunk_keeps_its_heading_in_the_text():
    text = "## Approval threshold\n\nAny refund over $100 requires approval.\n"
    chunk = kb_search.chunk_markdown(text, "refund-policy.md")[0]
    assert chunk.text.startswith("Approval threshold")


def test_chunk_id_is_a_precise_citation():
    text = "## Approval threshold\n\nBody.\n"
    chunk = kb_search.chunk_markdown(text, "refund-policy.md")[0]
    assert chunk.chunk_id == "refund-policy.md#approval-threshold"


def test_document_without_headings_becomes_one_chunk():
    chunks = kb_search.chunk_markdown("Just prose, no headings.", "flat.md")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "flat.md"


def test_corpus_loads_every_policy_document():
    docs = {c.doc_id for c in kb_search.load_corpus()}
    assert docs == {
        "refund-policy.md",
        "warranty-policy.md",
        "shipping-sla.md",
        "tone-of-voice.md",
        "escalation-matrix.md",
    }


# --- Retrieval ----------------------------------------------------------------


def test_index_contains_every_chunk(collection):
    assert collection.count() == len(kb_search.load_corpus())


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("can I get my money back after 45 days", "refund-policy.md"),
        ("the lamp flickers and is still under guarantee", "warranty-policy.md"),
        ("my parcel has not moved for two weeks", "shipping-sla.md"),
    ],
)
def test_search_finds_the_right_document(collection, query, expected_doc):
    hits = kb_search.search(query, k=6, collection=collection)
    assert expected_doc in {h.doc_id for h in hits}


def test_similarity_search_never_returns_always_on_docs(collection):
    hits = kb_search.search("I want to escalate this complaint", k=10, collection=collection)
    assert not ({h.doc_id for h in hits} & set(kb_search.ALWAYS_ON_DOCS))


def test_search_returns_doc_and_heading_for_every_hit(collection):
    hits = kb_search.search("refund over one hundred dollars", k=3, collection=collection)
    for hit in hits:
        assert hit.doc_id.endswith(".md")
        assert hit.heading
        assert "#" in hit.chunk_id


# --- Knowledge agent ----------------------------------------------------------


def test_knowledge_agent_always_returns_citations(collection):
    result = knowledge_lookup("I want a refund", "billing", collection=collection)
    assert result.stub is False
    assert len(result.citations) == len(result.snippets) > 0


def test_knowledge_agent_degrades_quietly_when_retrieval_fails():
    class _Broken:
        def query(self, **_kwargs):
            raise RuntimeError("index unavailable")

    result = knowledge_lookup("anything", "billing", collection=_Broken())
    assert result.citations == []
    assert result.snippets == []


# --- Phase 4 acceptance: retrieval recall against the golden set --------------


def test_expected_policy_document_is_retrieved_for_most_tickets(collection):
    """Precondition for groundedness: the topical policy a ticket needs must be retrieved.

    Scores topical documents only. `escalation-matrix.md` is excluded because it is standing
    policy supplied to every ticket rather than something similarity search can find — see
    ALWAYS_ON_DOCS. Counting it here would measure a hardcoded inclusion, not retrieval.

    End-to-end groundedness (does the *draft* only claim what was retrieved?) needs a live
    model and is deferred; this measures the half that does not.
    """
    tickets = [
        t for t in load_golden() if set(t.expected_citations) - set(kb_search.ALWAYS_ON_DOCS)
    ]
    assert tickets

    misses = []
    for ticket in tickets:
        result = knowledge_lookup(ticket.text, ticket.expected_category, collection=collection)
        retrieved = {c.split("#")[0] for c in result.citations}
        wanted = set(ticket.expected_citations) - set(kb_search.ALWAYS_ON_DOCS)
        if not wanted.issubset(retrieved):
            misses.append((ticket.id, sorted(wanted), sorted(retrieved)))

    recall = (len(tickets) - len(misses)) / len(tickets)
    assert recall >= 0.90, f"topical retrieval recall {recall:.1%} below 90%: {misses}"


def test_standing_policy_is_supplied_even_when_similarity_would_miss_it(collection):
    """T-19 is a legal threat; it shares no vocabulary with the escalation matrix."""
    ticket = next(t for t in load_golden() if t.id == "T-19")

    retrieved_only = kb_search.search(ticket.text, k=6, collection=collection)
    assert "escalation-matrix.md" not in {c.doc_id for c in retrieved_only}

    result = knowledge_lookup(ticket.text, ticket.expected_category, collection=collection)
    assert "escalation-matrix.md" in {c.split("#")[0] for c in result.citations}
