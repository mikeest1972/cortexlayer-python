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


def test_moderately_common_entity_alone_does_not_link(tmp_path):
    col = _fresh_collection(tmp_path)
    # "Project" sits on 4 of 5 pages (df=4, weight=0.25) — under the
    # common-entity cutoff (threshold=5) so it's still "linkable", but too
    # weak on its own (0.25 < default min_link_weight=0.5) to link any pair.
    # Old plain "shares >=1 entity" linking would have linked all 4 into a
    # clique; weighted scoring should leave them unlinked.
    ids = [storage.insert_page(col, f"Page {i}.", ["Project", f"topic{i}"]) for i in range(4)]
    storage.insert_page(col, "Unrelated filler.", ["Filler"])
    linking.run_linking_pass(col)
    for page_id in ids:
        assert storage.get_page(col, page_id)["links"] == []


def test_two_shared_entities_combine_to_link(tmp_path):
    col = _fresh_collection(tmp_path)
    # "Project" (df=4, weight 0.25) and "Widget" (df=4, weight 0.25) are each
    # too weak alone (< default min_link_weight=0.3). A and B share both, so
    # their combined score (0.5) clears the bar; pages sharing only one of
    # the two (C/D via Project, E/F via Widget) stay unlinked.
    a = storage.insert_page(col, "Page A.", ["Project", "Widget"])
    b = storage.insert_page(col, "Page B.", ["Project", "Widget"])
    c = storage.insert_page(col, "Page C.", ["Project"])
    d = storage.insert_page(col, "Page D.", ["Project"])
    e = storage.insert_page(col, "Page E.", ["Widget"])
    f = storage.insert_page(col, "Page F.", ["Widget"])
    linking.run_linking_pass(col)
    assert storage.get_page(col, a)["links"] == [b]
    assert storage.get_page(col, b)["links"] == [a]
    for page_id in (c, d, e, f):
        assert storage.get_page(col, page_id)["links"] == []


def test_max_links_per_page_caps_a_hub(tmp_path):
    col = _fresh_collection(tmp_path / "hub")
    # A hub page shares a distinct rare entity (df=2, weight=0.5) with each
    # of 20 other pages — every pair clears min_link_weight on its own, so
    # without a cap the hub would collect 20 links. It should keep only its
    # top max_links_per_page.
    hub = storage.insert_page(col, "Hub page.", [f"shared{i}" for i in range(20)])
    leaves = [storage.insert_page(col, f"Leaf {i}.", [f"shared{i}"]) for i in range(20)]
    linking.run_linking_pass(col)
    hub_links = storage.get_page(col, hub)["links"]
    assert len(hub_links) == 12  # default max_links_per_page
    assert set(hub_links) <= set(leaves)
    # Mutual top-K: a leaf the hub didn't keep loses the edge on its side
    # too (it's not "hub isn't in my top-K", every leaf's only candidate IS
    # the hub — it's that the hub didn't reciprocate).
    dropped = [leaf for leaf in leaves if leaf not in hub_links]
    assert dropped and all(storage.get_page(col, leaf)["links"] == [] for leaf in dropped)
    assert all(storage.get_page(col, leaf)["links"] == [hub] for leaf in hub_links)


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
