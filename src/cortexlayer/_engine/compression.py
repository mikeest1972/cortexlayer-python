"""Compression pass ("librarian" step): shrink retrieved passages to an answer.

A cheap/small LLM — qwen3.5:9b via local Ollama (reader consolidated onto the
extractor 2026-09-20, Miguel: keeps two models resident instead of three) — reads the raw passages and extracts a short direct answer plus source
page IDs. Pure summarization: no reasoning, link-following, or relevance judgment.

Transport is plain HTTP to Ollama's /api/chat via httpx (already in the dep tree
through chromadb). Host override: OLLAMA_HOST env var, default localhost:11434.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import httpx

DEFAULT_MODEL = "qwen3.5:9b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def build_prompt(query: str, passages: list[dict]) -> str:
    numbered = "\n\n".join(
        f"[Passage {i + 1} | page_id={p['page_id']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    return (
        "Answer the question using ONLY the passages below. Be concise and direct.\n"
        "Reply with a single JSON object: "
        '{"answer": "<short direct answer>", '
        '"source_page_ids": ["<page_id that supports the answer>", ...]}.\n'
        "If the passages do not contain the answer, reply with an empty answer "
        'and an empty source_page_ids list.\n\n'
        f"Question: {query}\n\nPassages:\n{numbered}"
    )


def ollama_chat(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single non-streaming chat call. Returns the raw response content."""
    with httpx.Client(base_url=OLLAMA_HOST, timeout=120.0) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                # Thinking models (e.g. qwen3.5) otherwise spend hundreds of hidden
                # reasoning tokens on a one-line answer (~11 s vs ~0.6 s measured).
                "think": False,
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def parse_response(content: str, fallback_ids: list[str]) -> dict:
    """Parse the model's JSON; fall back to raw text + all IDs on failure."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        parsed = json.loads(content[start:end])
        answer = str(parsed.get("answer", "")).strip()
        ids = [str(i) for i in parsed.get("source_page_ids", []) or []]
        return {"answer": answer, "source_page_ids": ids or fallback_ids}
    except (ValueError, AttributeError, TypeError):
        return {"answer": content.strip(), "source_page_ids": fallback_ids}


def compress(
    query: str,
    passages: list[dict],
    model: str = DEFAULT_MODEL,
    include_passage: bool = False,
    _chat: Callable[[str, str], str] | None = None,
) -> dict:
    """Compress passages into {answer, source_page_ids, [source_passage]}.

    ``_chat`` is injectable for tests (defaults to a live Ollama call).
    """
    chat = _chat or ollama_chat
    page_ids = [p["page_id"] for p in passages]
    result = parse_response(chat(build_prompt(query, passages), model), page_ids)
    if include_passage:
        result["source_passage"] = "\n\n".join(p["text"] for p in passages)
    return result
