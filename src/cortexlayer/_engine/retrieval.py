"""Retrieval pipeline: vector search + link-expansion.

Searches the whole collection (no vault/cluster routing — deliberately excluded).
Links only *add* candidates, never restrict. MVP expansion heuristic: pull the top 1
linked page per seed, no relevance judgment.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from . import storage

DEFAULT_K = 4


def retrieve(collection: Collection, query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return combined seed + link-expanded passages.

    Each passage is ``{page_id, text, score, via, [linked_from]}``: seeds are
    ``via="direct"`` (score = Chroma distance, lower = closer), expanded pages
    are ``via="link"`` with the seed id in ``linked_from``. Seeds come first,
    then expanded pages, deduplicated by page ID (a page that is both seed and
    expansion keeps ``direct``).
    """
    return expand_links(collection, storage.query(collection, query, n_results=k))


def expand_links(collection: Collection, seeds: list[dict]) -> list[dict]:
    """Seeds (storage-shaped page dicts with ``score``) -> passages, adding the
    top linked page of each seed. Split from :func:`retrieve` so other seed
    sources (fact memory's entity-boosted search) reuse the same expansion."""
    seen: set[str] = set()
    passages: list[dict] = []

    def _add(page_id: str, text: str, score: float, via: str,
             linked_from: str | None = None) -> None:
        if page_id not in seen:
            seen.add(page_id)
            passage: dict = {"page_id": page_id, "text": text,
                             "score": score, "via": via}
            if linked_from is not None:
                passage["linked_from"] = linked_from
            passages.append(passage)

    for seed in seeds:
        _add(seed["id"], seed["text"], seed.get("score", 0.0), "direct")
    for seed in seeds:
        # MVP heuristic: top 1 linked page per seed (links are stored sorted).
        if seed["links"]:
            target = storage.get_page(collection, seed["links"][0])
            if target is not None:
                _add(target["id"], target["text"], target.get("score", 0.0),
                     "link", linked_from=seed["id"])
    return passages
