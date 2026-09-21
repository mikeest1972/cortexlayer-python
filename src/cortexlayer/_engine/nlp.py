"""Language processing behind ingestion: entity extraction + sentence splitting.

Two implementations behind one tiny interface (``entities`` / ``sentences``):

- :class:`SpacyNLP` — spaCy NER plus a proper-noun pass (the quality path; this
  is exactly what the Cortex server uses).
- :class:`RegexNLP` — dependency-free capitalised-phrase heuristics. Good
  enough to try the library without downloading a model, but it finds fewer
  entities, so the linking pass links fewer pages. ``"auto"`` falls back to it
  (with a one-time warning) only when spaCy or its model is missing.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, List, Protocol

from ..errors import LocalDependencyError, MissingModelError

SPACY_MODEL = "en_core_web_sm"


class NLP(Protocol):
    def entities(self, text: str) -> List[str]: ...
    def sentences(self, text: str) -> List[str]: ...


def _ordered_unique_add(seen: set, out: List[str], name: str) -> None:
    name = name.strip()
    if name and name not in seen:
        seen.add(name)
        out.append(name)


class SpacyNLP:
    """spaCy-backed extraction. Loading is lazy and raises clear errors."""

    def __init__(self, model: str = SPACY_MODEL) -> None:
        self.model = model
        self._nlp: Any = None

    def _load(self) -> Any:
        if self._nlp is None:
            try:
                import spacy
            except ImportError as e:  # pragma: no cover — exercised via monkeypatch
                raise LocalDependencyError(
                    "spaCy is not installed. Run: pip install \"cortexlayer[local]\" "
                    "(or use Memory(entity_extractor=\"regex\"))."
                ) from e
            try:
                self._nlp = spacy.load(self.model)
            except OSError as e:
                raise MissingModelError(
                    f"spaCy model '{self.model}' is not installed. Run: "
                    f"python -m spacy download {self.model}  "
                    "(or use Memory(entity_extractor=\"regex\") to run without one)."
                ) from e
        return self._nlp

    def entities(self, text: str) -> List[str]:
        """Ordered-unique entities/key terms.

        spaCy NER first, plus a proper-noun fallback: the small model reliably
        tags people/places but often misses titles and works of art (e.g.
        "Inception"), which POS tagging still catches as PROPN. Both feed the
        linking pass, where a missed shared term means a missed link.
        """
        doc = self._load()(text)
        seen: set = set()
        entities: List[str] = []
        for ent in doc.ents:
            _ordered_unique_add(seen, entities, ent.text)
        lowered = [e.lower() for e in entities]
        for token in doc:
            if token.pos_ != "PROPN":
                continue
            word = token.text.strip()
            if not word:
                continue
            # Skip tokens already covered by a collected entity ("Nolan" in
            # "Christopher Nolan"), keep genuinely new key terms ("Inception").
            if any(word.lower() in e or e in word.lower() for e in lowered):
                continue
            _ordered_unique_add(seen, entities, word)
            lowered.append(word.lower())
        return entities

    def sentences(self, text: str) -> List[str]:
        return [s.text.strip() for s in self._load()(text).sents if s.text.strip()]


_CAP_RUN = re.compile(r"[A-Z][\w'’-]*(?:[ \t]+[A-Z][\w'’-]*)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")
_STOP = frozenset(
    "I A An The This That These Those There Then Here He She It We They You Me "
    "My Our His Her Their Its Your Mine Ours And But Or Nor So Yet If In On At "
    "For To Of With By From As Is Are Was Were Be Been Am Do Does Did Have Has "
    "Had Will Would Can Could Should May Might Must What When Where Who Whom "
    "Which Why How Yes No Not Also Just Well Oh Hi Hello Thanks Thank Please "
    "Okay Ok Maybe Sure Some Any All Each Every Both Either Neither One".split()
)


class RegexNLP:
    """Dependency-free heuristics: capitalised runs minus function words."""

    def entities(self, text: str) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for match in _CAP_RUN.finditer(text):
            words = match.group(0).split()
            while words and words[0] in _STOP:   # "The Godfather" -> "Godfather"
                words.pop(0)
            if words:
                _ordered_unique_add(seen, out, " ".join(words))
        return out

    def sentences(self, text: str) -> List[str]:
        return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


_warned_fallback = False


def resolve_nlp(spec: Any = "auto", *, spacy_model: str = SPACY_MODEL) -> NLP:
    """``"auto"`` | ``"spacy"`` | ``"regex"`` | any object with ``entities`` and
    ``sentences`` methods.

    ``"auto"`` prefers spaCy and falls back to the regex heuristics with a
    one-time warning if spaCy or its model is unavailable — so a fresh
    ``pip install "cortexlayer[local]"`` works even before the model download.
    """
    global _warned_fallback
    if hasattr(spec, "entities") and hasattr(spec, "sentences"):
        return spec
    if spec == "regex":
        return RegexNLP()
    if spec == "spacy":
        nlp = SpacyNLP(spacy_model)
        nlp._load()          # fail now, with the helpful message
        return nlp
    if spec == "auto":
        nlp = SpacyNLP(spacy_model)
        try:
            nlp._load()
            return nlp
        except LocalDependencyError as e:
            if not _warned_fallback:
                _warned_fallback = True
                warnings.warn(
                    f"cortexlayer: {e} Falling back to the simpler regex entity "
                    "extractor — fewer entities means fewer links between pages.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return RegexNLP()
    raise ValueError(
        "entity_extractor must be 'auto', 'spacy', 'regex', or an object with "
        f"entities() and sentences() methods, got {spec!r}"
    )
