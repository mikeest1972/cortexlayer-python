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
    min_link_weight: float = 0.3,
    max_links_per_page: int = 12,
) -> dict:
    """Populate every page's `links` with IDs of the pages it's most related to.

    Matching is exact on normalized (case-insensitive) entity strings. Two
    pages link when the entities they share are, together, rare enough to be
    meaningful: each shared entity contributes `1 / df(entity)` to a pairwise
    score, where df is how many (non-excluded) pages contain it. The default
    (0.3) means an entity confined to 2-3 pages (weight 0.5-0.33) still links
    them on its own — a normal small, specific cluster — but an entity spread
    across 4+ pages (weight ≤0.25) needs company: several such entities have
    to line up before a link forms. This replaces plain "shares ≥1 entity"
    linking, which let a handful of frequently-recurring but unexcluded
    entities (e.g. project/person names below the common-entity cutoff) turn
    an entire collection into one dense cluster (observed on Miguel's own
    project-notes memories, 2026-09-26).

    Entities appearing in more than max(common_entity_min_pages,
    common_entity_fraction * N) pages are excluded entirely — in dialogue,
    speaker names show up nearly everywhere and would otherwise link each page
    to nearly every other page (found in benchmark diagnosis). Excluded
    entities stay stored on the page; they just never contribute to a score.

    Each page keeps at most `max_links_per_page` links, and only to a page
    that *mutually* ranks among each other's top candidates by score (a
    "mutual top-K" graph): this is what actually bounds a hub's degree — if
    one side ranked the other outside its own top-K, the edge is dropped on
    both sides, since a page with many strong matches will always be some
    weaker match's top pick otherwise, which would defeat the cap. Without
    this, a large-enough collection can still re-accumulate a dense cluster
    as entities cross the weight threshold via many small contributions.

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
    linkable = {entity: ids for entity, ids in index.items() if len(ids) <= threshold}

    # Pairwise score: sum of 1/df(entity) over entities the two pages share.
    scores: dict[tuple[str, str], float] = {}
    for ids in linkable.values():
        weight = 1.0 / len(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                scores[key] = scores.get(key, 0.0) + weight

    # Candidates per page: neighbors whose combined score clears the bar.
    candidates: dict[str, list[tuple[str, float]]] = {page["id"]: [] for page in pages}
    for (a, b), score in scores.items():
        if score < min_link_weight:
            continue
        candidates[a].append((b, score))
        candidates[b].append((a, score))

    # Each page's own top-K, by score.
    top_ids: dict[str, set[str]] = {}
    for page_id, neighbors in candidates.items():
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        top_ids[page_id] = {other for other, _ in neighbors[:max_links_per_page]}

    links_written = 0
    for page in pages:
        page_id = page["id"]
        kept = {
            other
            for other, _ in candidates[page_id]
            if other in top_ids[page_id] and page_id in top_ids.get(other, ())
        }
        neighbors = sorted(kept)
        if neighbors != sorted(page["links"]):
            storage.update_links(collection, page_id, neighbors)
            links_written += 1

    return {"pages": len(pages), "links_written": links_written}
