"""Policy retrieval over the markdown corpus, backed by Chroma.

Chunked by markdown heading, not by character count. A `##` section in these documents is
already a self-contained rule ("Approval threshold", "Duplicate charges"), so heading
boundaries keep each chunk semantically whole and make citations precise — a hit points at
a named section a human can go read, not at characters 400-800 of a file.

Embeddings are computed locally by Chroma's bundled ONNX MiniLM. No API traffic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from support_desk import config

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
COLLECTION = "policies"

# Procedural policy that applies to every ticket regardless of topic. It is deliberately
# excluded from similarity search: these documents describe what *we* do ("escalate
# immediately", "what the handover must contain"), while tickets describe what happened to
# the customer ("burning smell", "my solicitor says"). The vocabularies never overlap, so
# embedding similarity cannot retrieve them — measured recall stayed at 88.9% even when
# retrieving a third of the whole corpus. Always-on context is the correct mechanism.
ALWAYS_ON_DOCS = frozenset({"escalation-matrix.md"})


@dataclass(frozen=True)
class Chunk:
    text: str
    doc_id: str
    heading: str

    @property
    def chunk_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.heading.lower()).strip("-")
        return f"{self.doc_id}#{slug}"


def chunk_markdown(text: str, doc_id: str) -> list[Chunk]:
    """Split a policy document into one chunk per `##` section."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [Chunk(text=text.strip(), doc_id=doc_id, heading=doc_id)]

    chunks = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            heading = match.group(1)
            chunks.append(Chunk(text=f"{heading}\n\n{body}", doc_id=doc_id, heading=heading))
    return chunks


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(config.POLICY_DIR.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def _client(persist: bool) -> Any:
    import chromadb

    if not persist:
        return chromadb.EphemeralClient()
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH))


def build_index(persist: bool = True, reset: bool = False) -> Any:
    """Embed the policy corpus into Chroma. Idempotent unless `reset`."""
    client = _client(persist)

    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception as exc:  # noqa: BLE001 - chroma raises a bare error for absent names
            logger.debug("no existing collection to reset: %s", exc)

    collection = client.get_or_create_collection(COLLECTION)
    chunks = load_corpus()
    if collection.count() != len(chunks):
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "heading": c.heading} for c in chunks],
        )
    return collection


@lru_cache(maxsize=1)
def _default_collection() -> Any:
    return build_index(persist=True)


def search(
    query: str,
    k: int = 6,
    collection: Any | None = None,
    exclude_docs: frozenset[str] = ALWAYS_ON_DOCS,
) -> list[Chunk]:
    """Return the k most relevant topical chunks, each carrying its document and heading."""
    col = collection if collection is not None else _default_collection()
    where = {"doc_id": {"$nin": list(exclude_docs)}} if exclude_docs else None
    result = col.query(query_texts=[query], n_results=k, where=where)

    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    return [
        Chunk(text=doc, doc_id=meta["doc_id"], heading=meta["heading"])
        for doc, meta in zip(documents[0], metadatas[0], strict=False)
    ]


def always_on_chunks(collection: Any | None = None) -> list[Chunk]:
    """Procedural policy handed to the Resolver on every ticket, never retrieved by similarity."""
    col = collection if collection is not None else _default_collection()
    result = col.get(where={"doc_id": {"$in": list(ALWAYS_ON_DOCS)}})

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        Chunk(text=doc, doc_id=meta["doc_id"], heading=meta["heading"])
        for doc, meta in zip(documents, metadatas, strict=False)
    ]
