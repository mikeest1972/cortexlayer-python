"""cortexlayer — Python client for Cortex, a memory layer for AI agents.

>>> from cortexlayer import CortexClient        # doctest: +SKIP
>>> c = CortexClient(api_key="...")             # or set CORTEX_API_KEY
>>> c.search("where do I live?")                # doctest: +SKIP

Async: ``AsyncCortexClient`` (same methods, awaited). Every failure is a
subclass of :class:`CortexError`.
"""

from ._version import __version__
from .client import DEFAULT_BASE_URL, AsyncCortexClient, CortexClient
from .errors import (
    AuthenticationError,
    ConflictError,
    ConnectionError,
    CortexConfigError,
    CortexError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    WritesNotSupportedError,
    LocalDependencyError,
    MissingModelError,
    LLMError,
)
from .types import (
    Account,
    AddResult,
    Answer,
    Graph,
    Page,
    PageList,
    SearchResult,
    Usage,
    UsageBucket,
)

__all__ = [
    "__version__", "DEFAULT_BASE_URL",
    "CortexClient", "AsyncCortexClient",
    "SearchResult", "Page", "PageList", "AddResult", "Account",
    "Usage", "UsageBucket", "Graph",
    "CortexError", "CortexConfigError", "InvalidRequestError",
    "AuthenticationError", "PermissionDeniedError", "NotFoundError",
    "ConflictError", "RateLimitError", "ServerError", "ConnectionError",
    "WritesNotSupportedError", "LocalDependencyError", "MissingModelError", "LLMError",
    "Memory", "Answer",
]


def __getattr__(name):
    """``Memory`` is loaded on first use so client-only installs stay light
    (the embedded engine's heavy imports happen when a ``Memory`` is built)."""
    if name == "Memory":
        from .memory import Memory

        return Memory
    raise AttributeError(f"module 'cortexlayer' has no attribute {name!r}")
