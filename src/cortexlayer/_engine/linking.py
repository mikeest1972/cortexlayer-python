"""Linking pass: connect pages that share entities.

Deterministic entity-overlap linking, no LLM judgment. Runs as a batch over the
full collection — after bulk ingestion or on-demand via `memory_relink` — never
per-insert. Re-running recomputes links from scratch, so it is idempotent.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from . import storage

PAGE_BATCH = 500


def _normalize(entity: str) -> str:
    return entity.strip().lower()


def run_linking_pass(
    collection: Collection,
    common_entity_fraction: float = 0.3,
    common_entity_min_pages: int = 5,
) -> dict:
    """Populate every page's `links` with IDs of pages sharing ≥1 entity.

    Matching is exact on normalized (case-insensitive) entity strings.
    Entities appearing in more than max(common_entity_min_pages,
    common_entity_fraction * N) pages are ignored for linking — in dialogue,
    speaker names show up nearly everywhere and would otherwise link each page
    to nearly every other page (found in benchmark diagnosis).
    Ignored entities stay stored on the page; they just don't create links.
    Returns summary stats: {"pages": n, "links_written": m}.
    """
    pages: list[dict] = []
    offset = 0
    while True:
        batch = storage.list_pages(collection, limit=PAGE_BATCH, offset=offset)
        if not batch:
            break
        pages.extend(batch)
        offset += len(batch)

    # Inverted index: normalized entity -> page IDs containing it.
    index: dict[str, list[str]] = {}
    for page in pages:
        for entity in page["entities"]:
            index.setdefault(_normalize(entity), []).append(page["id"])

    threshold = max(common_entity_min_pages, common_entity_fraction * len(pages))
    linkable = {entity for entity, ids in index.items() if len(ids) <= threshold}

    links_written = 0
    for page in pages:
        neighbors: list[str] = []
        for entity in page["entities"]:
            if _normalize(entity) not in linkable:
                continue
            for other_id in index.get(_normalize(entity), []):
                if other_id != page["id"] and other_id not in neighbors:
                    neighbors.append(other_id)
        neighbors.sort()
        if neighbors != sorted(page["links"]):
            storage.update_links(collection, page["id"], neighbors)
            links_written += 1

    return {"pages": len(pages), "links_written": links_written}
