"""Pluggable LLM and embedder for fact memory.

Defaults mirror the Cortex server's Mem0 configuration (Ollama, ``think`` off,
temperature/top_p/num_predict as Mem0's Ollama client: 0.1 / 0.1 / 2000) so the
extraction call is the same call. Anything with the same method works, so tests
and other providers need no Ollama.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Protocol, Sequence

import httpx

from ...errors import CortexConfigError, LLMError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding:8b"


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str:
        """Return the model's raw reply (expected to be a JSON object string)."""
        ...


class Embedder(Protocol):
    name: str  # recorded on the store: vectors from different embedders can't be mixed

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        """One vector per text. ``action`` is ``add`` | ``search`` | ``update``."""
        ...


class OllamaLLM:
    """Chat model over Ollama's ``/api/chat`` with native JSON output."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        host: Optional[str] = None,
        *,
        temperature: float = 0.1,
        top_p: float = 0.1,
        max_tokens: int = 2000,
        think: bool = False,
        timeout: float = 300.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model
        self.host = (host or ollama_host()).rstrip("/")
        self._opts = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
        self._think = think
        self._timeout = timeout
        self._http = http_client

    def generate(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                # Same nudge Mem0's Ollama client appends for JSON mode.
                {"role": "user", "content": user + "\n\nPlease respond with valid JSON only."},
            ],
            "format": "json",
            "stream": False,
            # Thinking models otherwise return a trace and JSON extraction
            # silently yields nothing (verified with qwen3.5:9b).
            "think": self._think,
            "options": self._opts,
        }
        try:
            client = self._http or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(f"{self.host}/api/chat", json=body)
            finally:
                if self._http is None:
                    client.close()
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"LLM call to {self.host} ({self.model}) failed: {e}") from e


class OllamaEmbedder:
    """Embeddings over Ollama's ``/api/embed``."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        host: Optional[str] = None,
        *,
        timeout: float = 120.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model
        self.host = (host or ollama_host()).rstrip("/")
        self.name = f"ollama:{model}"
        self._timeout = timeout
        self._http = http_client

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        if not texts:
            return []
        try:
            client = self._http or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(
                    f"{self.host}/api/embed", json={"model": self.model, "input": list(texts)}
                )
            finally:
                if self._http is None:
                    client.close()
            resp.raise_for_status()
            vectors = resp.json().get("embeddings") or []
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"Embedding call to {self.host} ({self.model}) failed: {e}") from e
        if len(vectors) != len(texts):
            raise LLMError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts ({self.model})"
            )
        return vectors


class ChromaEmbedder:
    """Chroma's built-in ONNX MiniLM-L6-v2 — no Ollama or network needed after
    the one-time ~80 MB model download. The library's default embedder."""

    name = "chroma:onnx-minilm-l6-v2"

    def __init__(self) -> None:
        self._fn: Any = None

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        if not texts:
            return []
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
        return [[float(x) for x in v] for v in self._fn(list(texts))]


class _CallableLLM:
    def __init__(self, fn: Callable[[str, str], str]) -> None:
        self._fn = fn

    def generate(self, system: str, user: str) -> str:
        return self._fn(system, user)


def resolve_llm(spec: Any) -> LLM:
    """``None`` → default Ollama; dict → ``OllamaLLM(**dict)``; an object with
    ``generate`` → itself; a callable ``fn(system, user) -> str`` → wrapped."""
    if spec is None:
        return OllamaLLM()
    if isinstance(spec, dict):
        cfg = dict(spec)
        provider = cfg.pop("provider", "ollama")
        if provider != "ollama":
            raise CortexConfigError(f"unknown llm provider {provider!r} (only 'ollama' is built in)")
        return OllamaLLM(**cfg)
    if hasattr(spec, "generate"):
        return spec
    if callable(spec):
        return _CallableLLM(spec)
    raise CortexConfigError(f"llm must be None, a dict, a callable or an object with generate(); got {spec!r}")


def resolve_embedder(spec: Any) -> Embedder:
    """``None``/``"chroma"`` → Chroma ONNX; ``"ollama"`` → Ollama default model;
    dict → ``{"provider": "ollama"|"chroma", ...}``; an object with
    ``embed_batch`` and ``name`` → itself."""
    if spec is None or spec == "chroma":
        return ChromaEmbedder()
    if spec == "ollama":
        return OllamaEmbedder()
    if isinstance(spec, dict):
        cfg = dict(spec)
        provider = cfg.pop("provider", "ollama")
        if provider == "ollama":
            return OllamaEmbedder(**cfg)
        if provider == "chroma":
            return ChromaEmbedder()
        raise CortexConfigError(f"unknown embedder provider {provider!r}")
    if hasattr(spec, "embed_batch") and hasattr(spec, "name"):
        return spec
    raise CortexConfigError(
        f"embedder must be None, 'chroma', 'ollama', a dict, or an object with embed_batch() and name; got {spec!r}"
    )
