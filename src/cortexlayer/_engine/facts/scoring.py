"""Search scoring for fact memory.

Adapted from Mem0 (Apache License 2.0; mem0ai 2.1.0, ``mem0/utils/scoring.py``
and ``Memory._compute_entity_boosts``). See the repository NOTICE file.

Changes: Mem0 only computes BM25 when the vector store implements keyword
search, and its Chroma connector does not, so in a Chroma deployment the BM25
branch never runs. The default here is identical (no keyword term). Task 0079
adds it as an opt-in: a dependency-free BM25 (no lemmatizer) whose score is
normalised to [0, 1] by the IDF mass of the query terms the store knows, then
fused as Mem0 does: ``combined = (semantic + keyword + entity) / max_possible``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

ENTITY_BOOST_WEIGHT = 0.5
ENTITY_MATCH_MIN_SIMILARITY = 0.5   # entity-store hits below this are ignored
MAX_QUERY_ENTITIES = 8


def distance_to_score(distance: float) -> float:
    """Chroma L2 distance -> similarity in (0, 1], higher is closer (as Mem0)."""
    return 1.0 / (1.0 + distance)


def entity_boost(similarity: float, num_linked: int) -> float:
    """Boost for the memories linked to one matched entity.

    Scaled down for entities linked to very many memories (they carry less
    information): ``sim * 0.5 / (1 + 0.001 * (n - 1)^2)``.
    """
    n = max(num_linked, 1)
    weight = 1.0 / (1.0 + 0.001 * ((n - 1) ** 2))
    return similarity * ENTITY_BOOST_WEIGHT * weight


BM25_K1 = 1.2
BM25_B = 0.75
KEYWORD_MAX_EXTRA_CANDIDATES = 50   # keyword hits the semantic over-fetch missed

_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have he her his how i "
    "if in is it its me my of on or our she so than that the their them then there these they this "
    "those to was we were what when where which who whom why will with would you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    """Lower-cased content words: possessive/plural endings folded, stopwords dropped.

    Deliberately tiny (no stemmer/lemmatizer dependency); applied identically to
    facts and queries, so consistency is what matters.
    """
    out: List[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok.endswith("'s"):
            tok = tok[:-2]
        tok = tok.replace("'", "")
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if tok and tok not in _STOPWORDS:
            out.append(tok)
    return out


def keyword_scores(query: str, documents: Dict[str, str]) -> Dict[str, float]:
    """BM25 of ``query`` over ``documents`` (id -> text), normalised to (0, 1].

    Raw BM25 is divided by the summed IDF of the query terms that occur in at
    least one document, so a fact containing every (known) query term scores
    ~1 and one containing only a very common term scores near 0 — comparable
    with the semantic and entity terms whatever the store size. Query terms no
    fact contains do not count against the others. Only facts with a nonzero
    score are returned.
    """
    q_terms = list(dict.fromkeys(tokenize(query)))
    if not q_terms or not documents:
        return {}
    tokens = {i: tokenize(t) for i, t in documents.items()}
    n = len(tokens)
    avg_len = (sum(len(t) for t in tokens.values()) / n) or 1.0
    counts = {i: {} for i in tokens}
    df = {t: 0 for t in q_terms}
    wanted = set(q_terms)
    for i, toks in tokens.items():
        c = counts[i]
        for t in toks:
            if t in wanted:
                c[t] = c.get(t, 0) + 1
        for t in c:
            df[t] += 1
    idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items() if d}
    total_idf = sum(idf.values())
    if total_idf <= 0:
        return {}
    out: Dict[str, float] = {}
    for i, c in counts.items():
        if not c:
            continue
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(tokens[i]) / avg_len)
        raw = sum(idf[t] * c[t] * (BM25_K1 + 1.0) / (c[t] + norm) for t in c)
        score = min(raw / total_idf, 1.0)   # tf=1 at average length gives idf, so full coverage ~ 1
        if score > 0:
            out[i] = score
    return out


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    entity_boosts: Dict[str, float],
    threshold: float,
    top_k: int,
    keyword: Optional[Dict[str, float]] = None,
    keyword_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """``combined = (semantic + w*keyword + entity_boost) / max_possible``, top-k descending.

    The threshold gates the *semantic* score before combining. ``max_possible``
    is 1.0, +``keyword_weight`` when any keyword score is active, +0.5 when any
    entity boost is. With no ``keyword`` this is exactly Mem0-with-Chroma.
    """
    keyword = keyword or {}
    max_possible = 1.0 + (ENTITY_BOOST_WEIGHT if entity_boosts else 0.0)
    if keyword:
        max_possible += keyword_weight
    scored: List[Dict[str, Any]] = []
    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue
        semantic = result.get("score") or 0.0
        if semantic < threshold:
            continue
        boost = entity_boosts.get(str(mem_id), 0.0)
        kw = keyword_weight * keyword.get(str(mem_id), 0.0)
        combined = min((semantic + kw + boost) / max_possible, 1.0)
        scored.append({**result, "id": str(mem_id), "score": combined})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]
