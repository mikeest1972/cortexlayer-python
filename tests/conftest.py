"""Shared fixtures. spaCy loads once per session (it costs ~1s), and every
``Memory`` gets its own tmp store."""

import pytest


@pytest.fixture(scope="session")
def spacy_nlp():
    pytest.importorskip("chromadb")
    pytest.importorskip("spacy")
    from cortexlayer._engine.nlp import SpacyNLP

    nlp = SpacyNLP()
    nlp._load()
    return nlp


@pytest.fixture
def mem(tmp_path, spacy_nlp):
    from cortexlayer import Memory

    return Memory(str(tmp_path / "store"), entity_extractor=spacy_nlp)
