"""Keyword (BM25) scoring for the fact engine (task 0079). Fakes only — no models."""

import json

import pytest

from cortexlayer import Memory
from cortexlayer._engine import storage
from cortexlayer._engine.facts import backends, scoring
from cortexlayer._engine.facts.engine import FactEngine

from test_facts import FakeEmbedder, ScriptedLLM  # noqa: E402  (same doubles as the 0078 tests)

QUERY = "What does Caroline's necklace symbolize?"
NECKLACE = "Caroline's necklace is a gift from her Swedish grandmother."


class BandEmbedder:
    """Reproduces the real-model failure: every fact sits in a narrow similarity band
    and the fact that literally says "necklace" is the *farthest* of them."""

    name = "fake:band"

    def __init__(self, texts):
        n = len(texts)
        self.vecs = {}
        for i, t in enumerate(texts):                    # unit vectors at angle 0.30 + i*0.01 from the query
            import math
            a = 0.30 + 0.01 * i
            self.vecs[t] = [math.cos(a), math.sin(a), 0.0, 0.0]
        self.query = [1.0, 0.0, 0.0, 0.0]

    def embed_batch(self, texts, action="add"):
        return [self.query if t == QUERY else self.vecs.get(t, [0.0, 0.0, 1.0, 0.0]) for t in texts]


class NoEntities:
    """Isolates the keyword term: no entity index, so ranking is semantic (+ keyword) only."""

    def entities(self, text):
        return []


def _corpus():
    fillers = [f"Caroline attended the event number {i} in Boston." for i in range(11)]
    return fillers + [NECKLACE]                           # necklace = last = least similar = 12th


def _engine(tmp_path, spacy_nlp, texts, *, weight):
    emb = BandEmbedder(texts)
    llm = ScriptedLLM()
    llm.queue.append(json.dumps({"memory": [{"text": t} for t in texts]}))
    client = storage.get_client(str(tmp_path / f"s{weight}"))
    eng = FactEngine(client, str(tmp_path / f"s{weight}"), backends.resolve_llm(llm), emb, NoEntities(),
                     keyword_weight=weight)
    eng.add("u", "Caroline talked about many things.")
    return eng


def test_reproduces_the_failure_and_keyword_scoring_fixes_it(tmp_path, spacy_nlp):
    texts = _corpus()
    off = _engine(tmp_path, spacy_nlp, texts, weight=0.0)
    on = _engine(tmp_path, spacy_nlp, texts, weight=1.0)
    ranked_off = [p["text"] for p in off.seeds("u", QUERY, 12)]
    assert ranked_off.index(NECKLACE) == 11                                # 12th of 12 on semantic score alone
    assert NECKLACE not in [p["text"] for p in off.seeds("u", QUERY, 4)]  # never reaches a k=4 answerer
    assert on.seeds("u", QUERY, 4)[0]["text"] == NECKLACE                  # exact-term fact wins with BM25


def test_keyword_off_is_identical_to_the_mem0_scoring(tmp_path, spacy_nlp):
    texts = _corpus()
    a = _engine(tmp_path, spacy_nlp, texts, weight=0.0)
    ranked = a.seeds("u", QUERY, 12)
    cands = [{"id": p["id"], "score": p["score"]} for p in ranked]
    # feeding score_and_rank no keyword input gives exactly the old formula
    assert scoring.score_and_rank(cands, {}, 0.0, 12) == scoring.score_and_rank(cands, {}, 0.0, 12, None, 1.0)
    assert scoring.score_and_rank(cands, {}, 0.0, 12) == scoring.score_and_rank(cands, {}, 0.0, 12, {}, 5.0)


def test_keyword_hit_outside_the_semantic_overfetch_is_pulled_in_with_its_true_score(tmp_path, spacy_nlp):
    filler = [f"Caroline visited place number {i}." for i in range(75)]
    texts = filler + [NECKLACE]                                            # 76 facts; limit=1 -> over-fetch is 60
    on = _engine(tmp_path, spacy_nlp, texts, weight=1.0)
    off = _engine(tmp_path, spacy_nlp, texts, weight=0.0)
    assert NECKLACE not in [p["text"] for p in off.seeds("u", QUERY, 1)]
    top = on.seeds("u", QUERY, 1)[0]
    assert top["text"] == NECKLACE
    # its semantic component is what Chroma would report: 1/(1+squared L2); no entities => divisor 2.0
    facts = on.collection("u")
    res = facts.query(query_embeddings=[on._embedder.query], n_results=76, include=["distances"])
    dist = dict(zip(res["ids"][0], res["distances"][0]))[top["id"]]
    got = facts.get(include=["documents"])
    kw = scoring.keyword_scores(QUERY, dict(zip(got["ids"], got["documents"])))[top["id"]]
    assert top["score"] == pytest.approx((scoring.distance_to_score(dist) + kw) / 2.0)


def test_keyword_failure_degrades_to_semantic_search(tmp_path, spacy_nlp, monkeypatch):
    on = _engine(tmp_path, spacy_nlp, _corpus(), weight=1.0)
    monkeypatch.setattr(scoring, "keyword_scores", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert len(on.seeds("u", QUERY, 4)) == 4                               # still answers, semantic only


# --- scoring math ---------------------------------------------------------------------


def test_tokenize_folds_possessives_and_plurals_and_drops_stopwords():
    assert scoring.tokenize("What does Caroline's necklaces symbolize?") == ["caroline", "necklace", "symbolize"]
    assert scoring.tokenize("The parties and the boss") == ["party", "boss"]
    assert scoring.tokenize("") == [] and scoring.tokenize("the of and") == []


def test_keyword_scores_weights_rare_terms_and_ignores_unknown_ones():
    docs = {"a": "Caroline went shopping", "b": "Caroline adopted a dog", "c": "Caroline lost her necklace",
            "d": "Melanie likes pottery"}
    s = scoring.keyword_scores("Caroline necklace zzzunknown", docs)
    assert set(s) == {"a", "b", "c"} and 0 < s["a"] < s["c"] <= 1.0
    assert s["c"] > 0.9                                                    # covers every *known* query term
    assert s["a"] == pytest.approx(s["b"], rel=0.2)                        # only the common term
    assert scoring.keyword_scores("zzz", docs) == {} and scoring.keyword_scores("necklace", {}) == {}
    assert scoring.keyword_scores("the of", docs) == {}


def test_score_and_rank_fuses_keyword_with_mem0_divisor():
    cands = [{"id": "a", "score": 0.6}, {"id": "b", "score": 0.5}]
    out = scoring.score_and_rank(cands, {}, 0.1, 5, {"b": 1.0}, 1.0)
    assert [r["id"] for r in out] == ["b", "a"]
    assert out[0]["score"] == pytest.approx((0.5 + 1.0) / 2.0)             # divisor 2.0 with keyword
    both = scoring.score_and_rank(cands, {"a": 0.5}, 0.1, 5, {"b": 1.0}, 1.0)
    assert next(r for r in both if r["id"] == "a")["score"] == pytest.approx((0.6 + 0.5) / 2.5)
    half = scoring.score_and_rank(cands, {}, 0.1, 5, {"b": 1.0}, 0.5)
    assert half[0]["score"] == pytest.approx((0.5 + 0.5) / 1.5)            # weight scales term and divisor
    assert scoring.score_and_rank([{"id": "c", "score": 0.05}], {}, 0.1, 5, {"c": 1.0}) == []   # never rescues sub-threshold


# --- through Memory ---------------------------------------------------------------------


def test_memory_keyword_scoring_option(tmp_path, spacy_nlp):
    def make(kw, name):
        llm = ScriptedLLM()
        return Memory(str(tmp_path / name), backend="facts", llm=llm, embedder=FakeEmbedder(),
                      entity_extractor=spacy_nlp, keyword_scoring=kw)
    m = make(True, "on")
    m.add("Caroline adopted a dog named Max. Melanie repaired an old bicycle.", user_id="u")
    assert m.search("bicycle", user_id="u", limit=1)[0].text.startswith("Melanie repaired")
    assert make(False, "off")._facts._kw_weight == 0.0 and m._facts._kw_weight == 1.0
    assert make(0.5, "half")._facts._kw_weight == 0.5
    cfg = Memory.from_config({"data_dir": str(tmp_path / "cfg"), "backend": "facts", "entity_extractor": spacy_nlp,
                              "embedder": FakeEmbedder(), "llm": ScriptedLLM(), "keyword_scoring": True})
    assert cfg._facts._kw_weight == 1.0
