# cortexlayer

Python library for **Cortex**, a memory layer for AI agents. Cortex stores memories as linked
pages and retrieves with vector search **plus link-expansion**, so multi-hop facts come back
without over-fetching a large top-k.

```bash
# Step 1 — install (run in your terminal, not in Python):
pip install "cortexlayer[local]"
python -m spacy download en_core_web_sm   # optional, better entity extraction
```

```python
# Step 2 — save this as hello_cortex.py and run with: python hello_cortex.py
from cortexlayer import Memory

m = Memory("./cortex-data")   # local store; omit the path for ~/.cortexlayer

m.add("Christopher Nolan directed Inception.", user_id="alice")
m.add("Christopher Nolan was born in London in 1970.", user_id="alice")
m.relink(user_id="alice")     # batch linking pass, run after adding several

for r in m.search("Who directed Inception?", user_id="alice", limit=1):
    print(r.title, "-", r.via)

# Christopher Nolan directed Inception. - direct
# Christopher Nolan was born in London in 1970. - link   <- pulled in through the shared entity
```

No server, no API key, nothing to sign up for — it runs fully on your machine. See below for the
hosted `CortexClient` and the `facts` engine (Mem0-style LLM fact extraction).

Two ways to use it, with the same method names and result types:

| | What it is | Install |
|---|---|---|
| **`Memory`** | The engine, embedded in your process. No server; data stays on your machine. | `pip install "cortexlayer[local]"` |
| **`CortexClient`** | A client for a running Cortex server (hosted or self-hosted). | `pip install cortexlayer` |

Python 3.10+. The client needs only `httpx`; the embedded engine adds Chroma and spaCy.

> **Status: 0.1 (alpha), on PyPI as [`cortexlayer`](https://pypi.org/project/cortexlayer/).** Local
> `Memory` has two engines: `raw` (default, no LLM) and `facts` (Mem0-style LLM fact extraction; see
> below). The Cortex server runs on this same library: its `facts` backend is `Memory(backend="facts")`,
> and it no longer depends on the `mem0ai` package. Source: [GitHub](https://github.com/mikeest1972/cortexlayer-python).

## Benchmarks

Measured on [LOCOMO](https://github.com/snap-research/locomo) — all 1,540 non-adversarial questions,
`Memory(backend="raw")`, `qwen3.5:9b` as answerer, k=4 — against Mem0's own published numbers (arXiv
2504.19413, Tables 1–2, GPT-4o-mini pipeline):

| Category | cortexlayer F1 | Mem0 F1 | cortexlayer tokens | Mem0 tokens |
|---|---|---|---|---|
| single-hop | 0.357 | 0.387 | 719 | 1,764 |
| multi-hop | 0.273 | 0.286 | 796 | 1,764 |
| temporal | 0.480 | 0.489 | 740 | 1,764 |
| open-domain | 0.170 | 0.476 | 751 | 1,764 |

**Near-parity with Mem0 on 3 of 4 categories**, at ~55&ndash;60% fewer retrieval tokens — plus a
compression pass that shrinks context further to ~7 tokens/answer, a lever Mem0 has no equivalent of.
Open-domain (inference/judgment questions, not fact lookup) is the one real gap left: GPT-4o-mini has
a genuine reasoning-capacity edge there that retrieval quality alone doesn't close.

Different answerer models on each side (local `qwen3.5:9b` vs GPT-4o-mini), so token counts and
retrieval architecture are the fair comparison, not a fully controlled one. Deterministic token-F1,
no LLM judge.

**What about no memory layer — just raw notes (e.g. an Obsidian vault) dumped straight into the
prompt?** Mem0's own paper runs that baseline too ("Full-Context Processing": no retrieval, the
whole ~26k-token conversation pasted into context every query). It actually beats Mem0 on their
LLM-judge score (72.9% vs 66.9%) — but at ~15x the tokens and ~12x the latency (p95 17.1s vs 1.44s),
and their own paper notes it doesn't scale: cost grows with how much you've ever written down, not
with what's relevant to the question. Fine for a small note collection that still fits in one
context window; not for memory that keeps growing.

## Quickstart: embedded (`Memory`)

```bash
pip install "cortexlayer[local]"
python -m spacy download en_core_web_sm     # recommended: better entity extraction
```

```python
from cortexlayer import Memory

m = Memory()                                 # ~/.cortexlayer, or Memory("path/to/dir")

m.add("Christopher Nolan directed Inception.", user_id="alice")
m.add("Christopher Nolan was born in London in 1970.", user_id="alice")
m.relink(user_id="alice")                    # batch linking pass: run after adding several

m.search("Who directed Inception?", user_id="alice", limit=1)
# [SearchResult(title='Christopher Nolan directed Inception.', via='direct', …),
#  SearchResult(title='Christopher Nolan was born in London in 1970.',
#               via='link', linked_from='…')]   <- pulled in through the shared entity
```

- **One store, many users:** every call takes an optional `user_id`; each user gets an isolated
  collection, so users can never see each other's pages. Omit it and the default user is used.
- **Linking is a batch pass**, never per insert: call `relink()` after adding memories (or pass
  `auto_relink=True` to relink after every `add`, which costs a scan of all pages).
- **Local and private:** memories live in an embedded Chroma store under `data_dir`. Nothing leaves
  your machine (Chroma's anonymous telemetry is switched off). Two one-time downloads: Chroma's small
  ONNX embedding model on first use (~80 MB), and the spaCy model if you install it.
- **No model? It still runs.** With `entity_extractor="auto"` (the default) `Memory` falls back to a
  simpler regex extractor, with a one-time warning, if spaCy or its model is missing. Entities drive
  linking, so the fallback finds fewer links. Force a choice with `"spacy"` or `"regex"`, or pass your
  own object with `entities(text)` and `sentences(text)` methods.
- **Short answers (optional):** `m.answer("Where did Alice move?")` retrieves and has an LLM distil a
  direct answer plus the supporting page ids. It uses a local Ollama by default
  (`$OLLAMA_HOST`); pass `chat=fn(prompt, model) -> str` to use any model.

| `Memory` method | Returns |
|---|---|
| `add(text, user_id=, timestamp=)` | `AddResult` (long text is chunked into several pages) |
| `search(query, user_id=, limit=4, expand_links=True)` | `list[SearchResult]` |
| `get(id)` / `get_all(query=, limit=, offset=)` | `Page` / `PageList` |
| `update(id, text)` / `delete(id)` / `delete_all(user_id=)` | `None` / `None` / count removed |
| `relink()` / `count()` | `{"pages", "links_written"}` / `int` |
| `answer(query, limit=, model=, chat=)` | `Answer(answer, source_page_ids)` |

`Memory.from_config({...})` builds one from a dict (`data_dir`, `entity_extractor`, `spacy_model`,
`default_user_id`, `auto_relink`, `backend`, `llm`, `embedder`, `custom_instructions`,
`observation_date_from_timestamp`, `keyword_scoring`).

### Fact memory: `Memory(backend="facts")`

The default `raw` engine stores your text as small pages and never calls an LLM. `facts` works like
Mem0: each `add` asks an LLM to distil the text into self-contained facts (resolving dates and pronouns),
stores those, and boosts search results by the entities they share with your query.

```python
m = Memory(
    backend="facts",
    llm={"model": "qwen3.5:9b"},                                   # Ollama at $OLLAMA_HOST by default
    embedder={"provider": "ollama", "model": "qwen3-embedding:8b"},  # default: Chroma's built-in ONNX model
)
m.add("[8 May, 2023] Caroline: I moved to Lisbon last week and adopted a dog named Max.", user_id="alice")
m.search("What is Caroline's dog called?", user_id="alice")   # -> "Caroline adopted a dog named Max ..."
```

- **Same API** as the raw engine, including `relink()` and link-expansion (links are entity overlap over the
  extracted facts).
- **Bring your own model:** `llm` can be a dict, a callable `fn(system, user) -> str`, or any object with
  `generate()`; `embedder` any object with `embed_batch(texts, action)` and a `name`. A store records which
  embedder made its vectors and refuses to open with a different one.
- **Failure is loud:** an unreachable LLM or embedder raises `LLMError` (nothing is stored). "The model found
  nothing worth remembering" is a normal empty result.
- **It costs an LLM call per `add`** (plus embeddings). The `raw` engine costs none.
- **Relative dates:** like Mem0, the extractor resolves "yesterday" against *today* unless told the
  conversation's date. `observation_date_from_timestamp=True` passes your `add(timestamp=...)` as that date.
- **Exact words:** embedding scores are often compressed into a narrow band, so a fact that literally contains
  a query word can rank below generic ones. Search fuses a BM25 keyword score by default (`keyword_scoring=False`
  gives plain semantic + entity scoring, identical to Mem0 on Chroma; on a small LOCOMO subset it lifted F1 from 0.37 to 0.47 at the same token cost — not yet a
  statistical result).
- **Facts are ordinary pages:** each fact is stored as a page in the user's Chroma collection, so links are
  persisted and `get` / `get_all` / `update` / `delete` work on facts exactly as on raw pages.
- **Origin:** the extraction prompt and pipeline are adapted from [Mem0](https://github.com/mem0ai/mem0)
  (Apache-2.0); cortexlayer does not depend on the `mem0ai` package. See `NOTICE`. The larger
  multi-conversation comparison against Mem0 is still open.

## Quickstart: hosted (`CortexClient`)

```bash
pip install cortexlayer
```

```python
from cortexlayer import CortexClient

client = CortexClient(api_key="...")        # or set CORTEX_API_KEY

client.search("Where does Alice live?", limit=5)
# [SearchResult(id='…', title='Alice moved to Lisbon in March.', via='direct', …),
#  SearchResult(id='…', title='…', via='link', linked_from='…'), …]

client.get_all(limit=50)                    # browse pages
client.get(page_id)                         # one page + its links
```

Create a key in the Cortex web app (**Keys**). Point at a self-hosted server with
`base_url="http://localhost:8000"` (or `CORTEX_BASE_URL`).

Writes:

```python
client.add("I moved to Lisbon in March.")   # long text is chunked into several pages
client.update(page_id, "…")
client.delete(page_id)
client.relink()                             # re-run the batch linking pass after adding several
```

> **Writes need a server with REST write endpoints** (added in Cortex server task 0073; the hosted
> server has them). Against an older self-hosted server these four raise `WritesNotSupportedError`;
> reads, search, graph and usage work on every version.

### Async

```python
from cortexlayer import AsyncCortexClient

async with AsyncCortexClient(api_key="...") as client:
    hits = await client.search("Where does Alice live?")
```

Same methods, awaited.

## Results

Plain frozen dataclasses (no pydantic). Unknown fields from a newer server are ignored.

| Call | Returns |
|---|---|
| `search(query, limit=4, expand_links=True)` | `list[SearchResult]` — `id, title, snippet, score, via, linked_from` |
| `get(id)` | `Page` — `id, title, content, links, linked_from, degree, created_at` |
| `get_all(query=None, limit=50, offset=0)` | `PageList` — `pages, total, limit, offset` |
| `add(text, timestamp=None)` | `AddResult` — `page_ids` |
| `me()` | `Account` — `user_id, account_name` |
| `usage(group_by="day", since=None, until=None)` | `Usage` — your own API usage, by `day` / `key` / `operation` |
| `graph(limit=None)`, `neighbors(id, depth=1)` | `Graph` — `nodes, edges, truncated, stale` |

`via` is `"direct"` for a vector hit and `"link"` for a page pulled in by link-expansion
(`linked_from` is the page that led to it). `score` semantics depend on the backend: on `raw` it is a
distance (lower is closer); on `facts` it is the fused semantic + keyword + entity score (higher is
closer). Compare within one store or server, not across.

## Errors

Every failure is a `CortexError`:

| Exception | When |
|---|---|
| `InvalidRequestError` (also a `ValueError`) | bad arguments (caught before any request) or a server 400 |
| `AuthenticationError` | 401 — key missing, wrong or revoked |
| `PermissionDeniedError` | 403 — `.code` is the server's reason (`session_required`, …) |
| `NotFoundError` | 404 — unknown page (another user's page looks the same) |
| `ConflictError` | 409 |
| `RateLimitError` | 429 — `.retry_after` seconds if sent |
| `ServerError` | 5xx or an unreadable reply |
| `ConnectionError` | no response (DNS, refused, timeout) |
| `WritesNotSupportedError` | the server is too old to have REST write endpoints |
| `CortexConfigError` | bad client configuration (e.g. no API key) |

Reads and search are retried (default 2×, with backoff) on connection errors and 502/503/504.
**Writes are never retried**, so an `add` can't be applied twice.

```python
from cortexlayer import CortexClient, NotFoundError

try:
    client.get("does-not-exist")
except NotFoundError:
    ...
```

## Configuration

```python
CortexClient(
    api_key=None,          # or CORTEX_API_KEY
    base_url=None,         # or CORTEX_BASE_URL; default https://api.cortexlayer.net
    timeout=30.0,
    max_retries=2,
    http_client=None,      # bring your own httpx.Client (proxies, transports, tests)
)
```

The key is never included in `repr()`. Use the client as a context manager (or `.close()`) to
release connections; a client you pass in is never closed for you.

## Coming from Mem0

| Mem0 | cortexlayer |
|---|---|
| `Memory()` (embedded) | `Memory()` |
| `MemoryClient(api_key=...)` (hosted) | `CortexClient(api_key=...)` |
| `m.add(messages, user_id=...)` | `m.add(text, user_id=...)` |
| `m.search(query, user_id=...)` | `m.search(query, user_id=...)` — results add `via` / `linked_from` provenance |
| `m.get_all(...)` / `m.get(id)` | `m.get_all()` / `m.get(id)` |
| `m.update(id, data)` / `m.delete(id)` / `m.delete_all(...)` | `m.update(id, text)` / `m.delete(id)` / `m.delete_all(user_id=)` |
| — | `m.relink()` — Cortex links pages in a batch pass, never per insert |

Differences worth knowing: Cortex takes plain text, not chat-message lists. The default `raw` engine
stores your text as small pages and links them by shared entities, so **adding never needs an LLM**;
`Memory(backend="facts")` is the Mem0-style engine that extracts facts with an LLM. On `CortexClient`
there is no `user_id` argument: each API key belongs to exactly one user.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Licence

[Apache-2.0](LICENSE).
