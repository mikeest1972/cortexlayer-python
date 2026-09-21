"""The NLP layer: spaCy parity, the regex fallback, and how 'auto' chooses."""

import warnings

import pytest

from cortexlayer import LocalDependencyError, MissingModelError
from cortexlayer._engine import nlp as nlp_mod
from cortexlayer._engine.nlp import RegexNLP, SpacyNLP, resolve_nlp


# --- regex fallback -----------------------------------------------------------


def test_regex_finds_capitalised_entities_and_keeps_order():
    ents = RegexNLP().entities("Christopher Nolan directed Inception in London.")
    assert ents == ["Christopher Nolan", "Inception", "London"]


def test_regex_drops_function_words_and_leading_articles():
    ents = RegexNLP().entities("The Godfather is great. I think It works. We met Bob.")
    assert "Godfather" in ents and "Bob" in ents
    assert not {"The", "I", "It", "We"} & set(ents)


def test_regex_entities_are_unique_and_do_not_span_lines():
    ents = RegexNLP().entities("Alice met Bob.\nAlice left.\nCarol stayed")
    assert ents == ["Alice", "Bob", "Carol"]


def test_regex_sentence_split():
    s = RegexNLP().sentences('It was late. "Go home," said Bob! Was it? Yes. 3 more.')
    assert s == ['It was late.', '"Go home," said Bob!', "Was it?", "Yes.", "3 more."]


def test_regex_extractor_still_links_pages_that_share_a_name(tmp_path):
    from cortexlayer import Memory

    pytest.importorskip("chromadb")
    m = Memory(str(tmp_path), entity_extractor="regex")
    m.add("Nolan directed Inception.")
    m.add("Nolan was born in London.")
    m.relink()
    assert all(p.degree == 1 for p in m.get_all().pages)


# --- resolve_nlp ------------------------------------------------------------------


def test_explicit_choices_and_custom_objects():
    assert isinstance(resolve_nlp("regex"), RegexNLP)

    class Mine:
        def entities(self, text):
            return ["x"]

        def sentences(self, text):
            return [text]

    mine = Mine()
    assert resolve_nlp(mine) is mine
    with pytest.raises(ValueError, match="entity_extractor"):
        resolve_nlp("nonsense")
    with pytest.raises(ValueError):
        resolve_nlp(42)


def test_spacy_choice_fails_loudly_without_the_model():
    with pytest.raises(MissingModelError, match="spacy download"):
        resolve_nlp("spacy", spacy_model="no_such_model_xyz")


def test_auto_uses_spacy_when_available(spacy_nlp):
    assert isinstance(resolve_nlp("auto"), SpacyNLP)


def test_auto_falls_back_to_regex_with_one_warning(monkeypatch):
    def boom(self):
        raise MissingModelError("model missing")

    monkeypatch.setattr(SpacyNLP, "_load", boom)
    monkeypatch.setattr(nlp_mod, "_warned_fallback", False)
    with pytest.warns(RuntimeWarning, match="regex entity"):
        first = resolve_nlp("auto")
    assert isinstance(first, RegexNLP)
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # a second warning would raise
        assert isinstance(resolve_nlp("auto"), RegexNLP)


def test_auto_fallback_also_covers_spacy_not_installed(monkeypatch):
    def boom(self):
        raise LocalDependencyError("spaCy is not installed")

    monkeypatch.setattr(SpacyNLP, "_load", boom)
    monkeypatch.setattr(nlp_mod, "_warned_fallback", True)   # already warned
    assert isinstance(resolve_nlp("auto"), RegexNLP)
