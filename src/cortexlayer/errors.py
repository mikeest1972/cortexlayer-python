"""Typed errors for the Cortex client.

The server answers errors as ``{"error": "<message or code>"}`` with an HTTP
status; each status maps to one exception class, all subclasses of
:class:`CortexError`.
"""

from __future__ import annotations

from typing import Optional


class CortexError(Exception):
    """Base class for every error this package raises."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return self.message


class CortexConfigError(CortexError, ValueError):
    """The client was constructed with missing/invalid configuration."""


class InvalidRequestError(CortexError, ValueError):
    """Bad arguments — caught client-side, or a server 400."""


class AuthenticationError(CortexError):
    """401: the API key is missing, invalid, or revoked."""


class PermissionDeniedError(CortexError):
    """403. ``code`` is the server's machine-readable reason when it sent one
    (e.g. ``session_required``, ``not_invited``, ``email_not_verified``)."""

    def __init__(self, message: str, *, status: Optional[int] = 403) -> None:
        super().__init__(message, status=status)
        self.code = message


class NotFoundError(CortexError):
    """404: no such page (unknown ids and other users' ids look identical)."""


class ConflictError(CortexError):
    """409: the request conflicts with current state (e.g. ``key_limit``)."""

    def __init__(self, message: str, *, status: Optional[int] = 409) -> None:
        super().__init__(message, status=status)
        self.code = message


class RateLimitError(CortexError):
    """429. ``retry_after`` is the ``Retry-After`` header in seconds, if sent."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = 429,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, status=status)
        self.retry_after = retry_after


class ServerError(CortexError):
    """5xx (or an unparseable server reply)."""


class ConnectionError(CortexError):  # noqa: A001 — deliberate: mirrors httpx naming
    """The request never got a response (DNS, refused, timeout, TLS...)."""


class WritesNotSupportedError(CortexError):
    """The server is too old to have REST write endpoints (added in Cortex
    server task 0073).

    Raised when ``add`` / ``update`` / ``delete`` / ``relink`` get a 404/405
    on the write route itself (as opposed to a 404 for an unknown page id).
    """


class LocalDependencyError(CortexError, ImportError):
    """The embedded engine needs packages that are not installed.

    Install them with ``pip install "cortexlayer[local]"`` (Chroma + spaCy).
    The hosted client (:class:`CortexClient`) never needs them.
    """


class MissingModelError(LocalDependencyError):
    """spaCy is installed but its language model is not.

    Fix: ``python -m spacy download en_core_web_sm`` — or construct
    ``Memory(entity_extractor="regex")`` to run without a model.
    """


class LLMError(CortexError):
    """The language model needed for fact extraction / answering failed
    (unreachable, timed out, or returned an error). Distinct from "the model
    found nothing worth remembering", which is a normal empty result."""
