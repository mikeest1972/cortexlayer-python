"""Fact-extraction prompts.

The system prompt (``additive_extraction_prompt.txt``) and the user-prompt
builder below are taken from Mem0 (https://github.com/mem0ai/mem0, Apache
License 2.0; ``mem0/configs/prompts.py``, mem0ai 2.1.0), copied unchanged so
Cortex's fact-memory engine starts from identical extraction behavior. See the
repository ``NOTICE`` file. Changes: none to the prompt text; the builder is
re-typed with type hints and its dates default to UTC today.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Dict, List, Optional, Union

PAST_MESSAGE_TRUNCATION_LIMIT = 300

_prompt_cache: Optional[str] = None


def extraction_system_prompt() -> str:
    """The additive (ADD-only) extraction system prompt, from package data."""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = (
            resources.files(__package__)
            .joinpath("additive_extraction_prompt.txt")
            .read_text(encoding="utf-8")
        )
    return _prompt_cache


def _truncate(text: str, limit: int = PAST_MESSAGE_TRUNCATION_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _format_summary(summary: Union[None, str, Dict[str, Any]]) -> str:
    if isinstance(summary, dict):
        return summary.get("summary", "")
    return summary or ""


def _format_conversation_history(messages: Optional[List[Dict[str, Any]]]) -> str:
    if not messages:
        return ""
    out = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("message") or msg.get("content", "")
        if role and content:
            out += f"{role}: {_truncate(content)}\n"
    return out


def _serialize_memories(memories: Optional[List[Dict[str, Any]]]) -> str:
    return json.dumps(memories or [], ensure_ascii=False)


def _format_new_messages(new_messages: Union[str, List[Dict[str, Any]], None]) -> str:
    if isinstance(new_messages, str):
        return new_messages
    return json.dumps(new_messages or [], ensure_ascii=False)


def _resolve_dates(current_date: Optional[str] = None, observation_date: Optional[str] = None):
    if current_date is None:
        current_date = datetime.now(timezone.utc).date().isoformat()
    if observation_date is None:
        observation_date = current_date
    return current_date, observation_date


def build_extraction_prompt(
    *,
    summary: Union[None, str, Dict[str, Any]] = None,
    recently_extracted_memories: Optional[List[Dict[str, Any]]] = None,
    existing_memories: Optional[List[Dict[str, Any]]] = None,
    new_messages: Union[str, List[Dict[str, Any]], None] = None,
    last_k_messages: Optional[List[Dict[str, Any]]] = None,
    current_date: Optional[str] = None,
    timestamp: Optional[str] = None,
    custom_instructions: Optional[str] = None,
) -> str:
    """Build the user-side prompt (pairs with :func:`extraction_system_prompt`)."""
    current_date, observation_date = _resolve_dates(current_date, timestamp)
    sections = [
        f"## Summary\n{_format_summary(summary)}",
        f"## Last k Messages\n{_format_conversation_history(last_k_messages)}",
        f"## Recently Extracted Memories\n{_serialize_memories(recently_extracted_memories)}",
        f"## Existing Memories\n{_serialize_memories(existing_memories)}",
        f"## New Messages\n{_format_new_messages(new_messages)}",
        f"## Observation Date\n{observation_date}",
        f"## Current Date\n{current_date}",
    ]
    if custom_instructions:
        sections.append(f"## Custom Instructions\n{custom_instructions}")
    sections.append("# Output:")
    return "\n\n".join(sections)
