"""Ported verbatim from the Cortex backend suite (test_linking.py) — proves the copied engine
behaves identically. Only the imports changed."""

from cortexlayer._engine import linking
from cortexlayer._engine import storage


def _fresh_collection(tmp_path):
    client = storage.get_client(persist_dir=str(tmp_path / "chroma"))
    return storage.get_collection(client)


def _seed(col):
    """Three pages: A↔B share 'Inception', C is isolated."""
    a = storage.insert_page(col, "Nolan directed Inception.", ["Christopher Nolan", "Inception"])
    b = storage.insert_page(col, "Inception came out in 2010.", ["Inception", "2010"])
    c = storage.insert_page(col, "Apples are a fruit.", ["Apples"])
    return a, b, c


def test_shared_entity_creates_mutual_links(tmp_path):
    col = _fresh_collection(tmp_path)
    a, b, _ = _seed(col)
    stats = linking.run_linking_pass(col)
    assert stats == {"pages": 3, "links_written": 2}
    assert storage.get_page(col, a)["links"] == [b]
    assert storage.get_page(col, b)["links"] == [a]


def test_isolated_page_keeps_empty_links(tmp_path):
    col = _fresh_collection(tmp_path)
    _, _, c = _seed(col)
    linking.run_linking_pass(col)
    assert storage.get_page(col, c)["links"] == []


def test_rerun_is_idempotent(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed(col)
    linking.run_linking_pass(col)
    stats = linking.run_linking_pass(col)
    assert stats["links_written"] == 0


def test_no_self_links_and_case_insensitive(tmp_path):
    col = _fresh_collection(tmp_path)
    # Same page's entity must not link to itself; matching ignores case.
    a = storage.insert_page(col, "Nolan film.", ["nolan"])
    b = storage.insert_page(col, "Nolan interview.", ["Nolan"])
    linking.run_linking_pass(col)
    assert storage.get_page(col, a)["links"] == [b]
    assert storage.get_page(col, b)["links"] == [a]


def test_ubiquitous_entity_creates_no_links(tmp_path):
    col = _fresh_collection(tmp_path)
    # "Common" on all 6 pages exceeds the min-pages floor → ignored for links.
    # Rare pair-entities still link their pairs.
    pairs = []
    for topic in ("apples", "orbits"):
        a = storage.insert_page(col, f"First {topic} page.", ["Common", topic])
        b = storage.insert_page(col, f"Second {topic} page.", ["Common", topic])
        pairs.append((a, b))
    storage.insert_page(col, "Filler one.", ["Common"])
    storage.insert_page(col, "Filler two.", ["Common"])
    linking.run_linking_pass(col)
    for a, b in pairs:
        assert storage.get_page(col, a)["links"] == [b]
        assert storage.get_page(col, b)["links"] == [a]
