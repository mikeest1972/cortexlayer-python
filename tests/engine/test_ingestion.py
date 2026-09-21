"""Ingestion — adapted from the backend's test_ingestion.py (the functions now
take an ``nlp`` object instead of a module-global spaCy)."""

from cortexlayer._engine import ingestion, storage


def _fresh_collection(tmp_path):
    client = storage.get_client(persist_dir=str(tmp_path / "chroma"))
    return storage.get_collection(client)


def test_extract_entities_finds_names(spacy_nlp):
    entities = spacy_nlp.entities("Christopher Nolan directed Inception in 2010.")
    assert "Christopher Nolan" in entities
    assert "Inception" in entities
    assert len(entities) == len(set(entities))          # ordered-unique


def test_chunk_text_splits_turns(spacy_nlp):
    chunks = ingestion.chunk_text("Alice said hello.\nBob replied hi.\n", spacy_nlp)
    assert chunks == ["Alice said hello.", "Bob replied hi."]


def test_chunk_text_respects_max_chars(spacy_nlp):
    long_turn = " ".join(f"Sentence number {i} states a fact." for i in range(20))
    chunks = ingestion.chunk_text(long_turn, spacy_nlp, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == long_turn                # no content lost


def test_add_text_end_to_end(tmp_path, spacy_nlp):
    col = _fresh_collection(tmp_path)
    ids = ingestion.add_text(
        col, "Christopher Nolan directed Inception.\nAlice met Bob in Paris.", spacy_nlp
    )
    assert len(ids) == 2
    page = storage.get_page(col, ids[0])
    assert page["text"] == "Christopher Nolan directed Inception."
    assert "Christopher Nolan" in page["entities"] and page["links"] == []


def test_add_text_with_timestamp_prefixes_undated_lines_only(tmp_path, spacy_nlp):
    col = _fresh_collection(tmp_path)
    ids = ingestion.add_text(
        col, "Alice moved to Lisbon.\n[1 Jan, 2020] Bob stayed in Rome.",
        spacy_nlp, timestamp="8 May, 2023",
    )
    pages = [storage.get_page(col, i) for i in ids]
    assert pages[0]["text"] == "[8 May, 2023] Alice moved to Lisbon."
    assert pages[1]["text"] == "[1 Jan, 2020] Bob stayed in Rome."   # keeps its own date
    assert all(p["created_at"] == "8 May, 2023" for p in pages)


def test_apply_timestamp_leaves_blank_lines_alone():
    assert ingestion.apply_timestamp("a\n\nb", "T") == "[T] a\n\n[T] b"


def test_add_text_ignores_blank_input_lines(tmp_path, spacy_nlp):
    col = _fresh_collection(tmp_path)
    assert ingestion.add_text(col, "\n\n  \n", spacy_nlp) == []
