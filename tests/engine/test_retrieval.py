"""Ported verbatim from the Cortex backend suite (test_retrieval.py, mem0 test omitted) — proves the copied engine
behaves identically. Only the imports changed."""

from cortexlayer._engine import ingestion as _ingestion
from cortexlayer._engine import nlp as _nlp


class ingestion:  # noqa: N801 — shim: keeps the ported test bodies byte-identical
    _nlp = _nlp.SpacyNLP()

    @staticmethod
    def ingest_text(collection, text):
        return _ingestion.add_text(collection, text, ingestion._nlp)

from cortexlayer._engine import linking
from cortexlayer._engine import retrieval
from cortexlayer._engine import storage


def _fresh_collection(tmp_path):
    client = storage.get_client(persist_dir=str(tmp_path / "chroma"))
    return storage.get_collection(client)


def _seed_two_hop(col):
    """S (about the film/director) <-> A (about the director's age)."""
    ingestion.ingest_text(
        col,
        "The director of Inception is Christopher Nolan.\n"
        "Christopher Nolan was born on July 30, 1970.\n",
    )
    linking.run_linking_pass(col)


def test_expansion_pulls_linked_page(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Who directed Inception?", k=1)
    texts = [p["text"] for p in passages]
    # Seed answers who; the linked page (birth date) rides along via expansion.
    assert any("Christopher Nolan" in t and "director" in t for t in texts)
    assert any("1970" in t for t in texts)


def test_expansion_is_robust_to_which_seed_wins(tmp_path):
    """Whatever the top seed is, its top-1 link must be in the results."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    for query in ["Who directed Inception?", "When was Nolan born?", "Inception film"]:
        passages = retrieval.retrieve(col, query, k=1)
        ids = [p["page_id"] for p in passages]
        assert len(ids) == len(set(ids)), "duplicates leaked through"
        seed = storage.query(col, query, n_results=1)[0]
        if seed["links"]:
            assert seed["links"][0] in ids


def test_dedupe_with_mutual_links(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Inception Nolan", k=5)
    ids = [p["page_id"] for p in passages]
    assert len(ids) == len(set(ids))
    assert len(passages) == 2  # both pages, no duplicates despite mutual links


def test_empty_collection(tmp_path):
    col = _fresh_collection(tmp_path)
    assert retrieval.retrieve(col, "anything", k=4) == []


def test_provenance_marks_direct_vs_link(tmp_path):
    """0044: seeds are via=direct (with a score), expansions via=link + linked_from."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Who directed Inception?", k=1)
    by_id = {p["page_id"]: p for p in passages}
    assert len(passages) == 2
    seed = storage.query(col, "Who directed Inception?", n_results=1)[0]
    assert "score" in seed
    direct = by_id[seed["id"]]
    assert direct["via"] == "direct"
    assert "linked_from" not in direct
    assert direct["score"] == seed["score"]
    linked = [p for p in passages if p["page_id"] != seed["id"]][0]
    assert linked["via"] == "link"
    assert linked["linked_from"] == seed["id"]


def test_provenance_dedupe_keeps_direct(tmp_path):
    """A page that is both seed and expansion stays via=direct."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Inception Nolan", k=5)
    by_id = {p["page_id"]: p for p in passages}
    assert all(p["via"] == "direct" for p in passages if "linked_from" not in p)
    # Mutual links: each page links the other, both are seeds — no link rows survive.
    assert all("linked_from" not in p for p in passages), by_id
