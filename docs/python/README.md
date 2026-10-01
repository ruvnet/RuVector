# ruvector for Python

A low-latency vector search library with a Rust core (PyO3 + maturin), a CLI, and an MCP
server — the Python counterpart to the [Node.js `ruvector` package](../../README.md). Algorithms
stay in Rust; Python is a thin, typed layer over them. NumPy arrays are read without copying
(`PyReadonlyArray`), and heavy calls (`insert`, `insert_batch`, `search`) release the GIL.

This guide covers the Python package end to end. Design rationale, verified benchmark numbers,
security findings, and the full capability-coverage table live in
[`docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`](../adr/ADR-352-ruvector-python-sdk-cli-mcp.md) —
read that for the "why", this guide for the "how".

> **Status**: this package has not been published to PyPI yet. Everything below describes the
> package as built on branch `feat/python-sdk`; install from source until a release ships.

## Install

```bash
# from source, this branch (not yet on PyPI)
cd crates/ruvector-py
pip install maturin
maturin develop --release        # or: maturin build --release && pip install dist/*.whl

# once published:
pip install ruvector
# or
uv add ruvector
```

Plain `pip install ruvector` gives you the library only (`numpy` is the one hard dependency —
`import ruvector` never pulls in anything else). Everything else is an opt-in extra:

| Extra | Pulls in | Needed for |
|---|---|---|
| `ruvector[cli]` | `click`, `rich` | the `ruvector` console script |
| `ruvector[mcp]` | `mcp>=2.0` | `ruvector serve` (the MCP server) |
| `ruvector[langchain]` | `langchain-core>=1.6` | `ruvector.integrations.langchain.RuVectorStore` |
| `ruvector[llamaindex]` | `llama-index-core>=0.14` | `ruvector.integrations.llamaindex.RuVectorStore` |
| `ruvector[salesforce]` | `httpx>=0.27` | `ruvector.integrations.salesforce` + the Agentforce action routes |
| `ruvector[all]` | all of the above | everything |

Each extra is imported lazily by only the module that needs it — `import ruvector` stays
numpy-only regardless of what's installed, and `ruvector.cli` imports `click`/`rich` per
subcommand rather than at module load, which is why `ruvector --help` stays fast (see "Perf
tips" below).

## Quick start

```python
import numpy as np
from ruvector import Collection

# backend="hnsw" is the default (fast approximate search, metadata-filtered
# in Rust). backend="rabitq" is also available (1-bit quantized, exact f32
# rerank) — see "Choosing a backend" below.
coll = Collection.create(dim=384)

vec = np.random.default_rng(0).standard_normal(384).astype(np.float32)
doc_id = coll.insert(vec, metadata={"text": "hello world", "source": "notes"})

hits = coll.search(vec, k=5, filter={"source": "notes"})
for h in hits:
    print(h.id, h.score, h.metadata)

coll.save("my-collection")       # writes my-collection(.npy|.rbpx) + my-collection.meta.json
coll2 = Collection.load("my-collection")
```

## SDK reference

### `Collection`

The main entry point — ids, metadata, filtering, soft-delete, and persistence over either
backend.

```python
Collection.create(dim: int, backend: str = "hnsw", rerank_factor: int = 20, seed: int = 42) -> Collection
Collection.from_vectors(vectors: NDArray[np.float32], metadatas=None, backend="hnsw", ...) -> Collection
Collection.load(path: str | Path) -> Collection
coll.save(path: str | Path) -> None

coll.insert(vector: NDArray[np.float32], metadata: dict | None = None) -> int          # returns new id
coll.insert_batch(vectors: NDArray[np.float32], metadatas: list[dict | None] | None = None) -> list[int]
coll.search(query: NDArray[np.float32], k: int = 10, filter: dict | Callable | None = None,
            rerank_factor: int | None = None) -> list[SearchHit]   # SearchHit: .id .score .metadata
coll.delete(id: int) -> None            # soft delete (tombstone)
coll.vacuum() -> int                    # physically rebuild, reclaim tombstoned space; returns rows dropped
coll.get_metadata(id: int) -> dict | None
coll.export_live_items() -> list[tuple[int, NDArray[np.float32], dict | None]]
coll.stats() -> CollectionStats         # count, dim, backend, rerank_factor, memory_bytes, tombstoned
coll.metric -> str                      # "cosine" (hnsw, configurable) or "squared_l2" (rabitq, fixed)
len(coll) -> int
```

`filter` accepts either an exact-match dict (`{"cat": "news"}`, evaluated in Rust for the `hnsw`
backend — no Python-side round trip per candidate) or an arbitrary Python callable
(`lambda meta: meta.get("score", 0) > 0.5`, always evaluated Python-side since it's arbitrary
code). `rabitq` always filters Python-side regardless of which form is used.

### Choosing a backend

- **`hnsw` (default)** — approximate nearest-neighbor via `hnsw_rs`, metadata-aware, dict-filter
  evaluated in Rust. Good default for most workloads; see the benchmark table below for where it
  stands against `hnswlib`.
- **`rabitq`** — RaBitQ+ symmetric 1-bit quantized scan with an exact f32 rerank pass
  (`rerank_factor` controls how many candidates get the exact rerank). Ids are stored as `u32`
  internally (0..2³²-1) — `Collection` raises `CollectionError` if you try to use an id outside
  that range with this backend. `coll.metric` always reports `"squared_l2"` for this backend
  (it doesn't support a configurable metric, unlike `hnsw`).

### Lower-level classes

`RabitqIndex` and `HnswIndex` are the raw index types `Collection` wraps (ids-as-strings for
`HnswIndex`, ids-as-u32 for `RabitqIndex`) — use these directly only if you need an index without
`Collection`'s id/metadata/soft-delete bookkeeping on top.

### Graph, GNN/attention rerank, clustering, SONA

Bound as standalone classes/functions, not wired into `Collection` — see ADR-352's "Capability
landing" table for exactly what's in and out of scope for each:

```python
from ruvector import GraphDB, GnnLayer, AttentionReranker, kmeans, SonaEngine

g = GraphDB()
n1 = g.create_node(labels=["Person"], properties={"name": "Alice"})
n2 = g.create_node(labels=["Person"], properties={"name": "Bob"})
g.create_edge(n1, n2, "KNOWS", properties={})   # 3rd positional arg is relation_type
rows = g.query_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alice' RETURN a, b")
# MATCH/WHERE only — RETURN is parsed but not projected (matches the upstream
# ruvector-graph-node contract exactly); CREATE/MERGE/SET/DELETE are rejected or no-op.

reranker = AttentionReranker(dim=384)
blended, weights = reranker.rerank(query_vec, candidate_vecs)   # weights: what to re-sort by

# GnnLayer.forward is per-node (not a full-graph batch op): `node` is a
# single (input_dim,) vector, `neighbors` is that node's (n, input_dim)
# neighbor set, `weights` is an optional (n,) per-edge weight.
layer = GnnLayer(input_dim=384, hidden_dim=128, heads=4, dropout=0.0)   # untrained = Xavier/Glorot random
out = layer.forward(node, neighbors, weights=None)                     # projection, NOT a trained quality win

# kmeans returns a plain tuple, not an object with attributes:
assignments, centroids, cohesion, cluster_sizes = kmeans(vectors, k=8, iters=20)

sona = SonaEngine(hidden_dim=384)
out_vec = sona.apply_micro_lora(vec)   # fresh engine: EXACT identity transform (zero-init +
                                        # residual), not "close to one" — inference-only; the
                                        # online-learning API (begin_trajectory/tick/force_learn)
                                        # is bound but needs a real reward signal to do anything
```

### Error handling

Every error `ruvector` raises is a `ruvector.RuVectorError` (or its `ruvector.CollectionError`
subclass for `Collection`-level misuse, e.g. an out-of-range rabitq id). Catch `RuVectorError` to
handle anything the library itself raises, distinct from `KeyError`/`ValueError`/`TypeError` from
ordinary Python-level misuse (missing dict keys, wrong argument types).

```python
from ruvector import RuVectorError

try:
    coll.search(wrong_dim_vector, k=5)
except RuVectorError as e:
    ...
```

## CLI

```bash
pip install 'ruvector[cli]'
```

| Command | What it does |
|---|---|
| `ruvector create --path P --dim D [--rerank-factor N] [--seed N]` | create a new empty collection |
| `ruvector import --path P --vectors V.npy [--metadata M.json] [--rerank-factor N] [--seed N]` | bulk-build a collection from a `.npy` file |
| `ruvector insert-batch --path P --vectors V.npy [--metadata M.json]` | insert rows into an existing collection |
| `ruvector search --path P --query Q.npy [-k N] [--filter '{"k":"v"}'] [--json] [--no-color]` | search; prints a ranked, colored Rich table (or plain JSON with `--json`) |
| `ruvector delete --path P --id N [--id N ...] [--vacuum]` | soft-delete one or more ids |
| `ruvector export --path P --out PREFIX` | export vectors + metadata to `PREFIX.npy` / `PREFIX.meta.json` |
| `ruvector info --path P [--json] [--no-color]` | collection stats, in a Rich panel or JSON |
| `ruvector benchmark [-n N] [--dim D] [--k K] [--queries Q] [--rerank-factor N] [--json]` | in-process build+search latency benchmark (no comparator — see the ADR for a real side-by-side) |
| `ruvector serve [--http] [--host H] [--port P]` | launch the MCP server (stdio by default; `--http` for streamable-HTTP) |

`--json` output on `search`/`info`/`benchmark` is always plain, unstyled JSON — safe to pipe into
`jq` or another script regardless of `--no-color`. Vectors go in and out as `.npy` files
(`np.save`/`np.load`, shape `(n, dim)` or `(dim,)`, dtype `float32`); metadata as a `.json` file
holding a list of dicts (or `null` entries) aligned with the vector rows.

Color honors both `--no-color` and the `NO_COLOR` env var, on top of the usual auto-detection for
non-TTY stdout. `NO_COLOR`/`--no-color` suppress color specifically — bold/dim can still render on
a real TTY; piping to a file strips all ANSI regardless.

## MCP server + ChatGPT `ui://` widget

```bash
pip install 'ruvector[mcp]'
ruvector serve                       # stdio (default) — for Claude Desktop, most MCP clients
ruvector serve --http --port 8420    # streamable-HTTP — for a ChatGPT connector or a remote client
```

### Tools

| Tool | Scope needed | What it does |
|---|---|---|
| `vector_create_collection(name, dim, rerank_factor=20, seed=42)` | write | create a new empty collection |
| `vector_insert(name, vector, metadata=None)` | write | insert one vector |
| `vector_insert_batch(name, vectors, metadatas=None)` | write | insert many vectors |
| `vector_search(name, query, k=10, filter=None, rerank_factor=None)` | read | k-NN search with optional metadata filter |
| `vector_delete(name, id, vacuum=False)` | write | soft-delete (+ optional vacuum) |
| `vector_stats(name)` | read | count/dim/backend/rerank_factor/memory_bytes/tombstoned |
| `vector_list_collections()` | read | list every collection under the server's data root |
| `vector_explore(name, query, k=10)` | read | same as `vector_search`, but also returns the ChatGPT `ui://` widget resource |

Collection names passed as a tool argument go through `ruvector.mcp_server._safe_path`, which
restricts them to `^[A-Za-z0-9_-]{1,128}$` and resolves strictly under
`RUVECTOR_MCP_DATA_DIR` (default: a local data directory) — this is the actual remote trust
boundary for this server (unlike the CLI's `--path`, which runs with the caller's own filesystem
authority, same as `sqlite3 file.db`).

### Auth (streamable-HTTP only; stdio has no network surface to protect)

Unset `RUVECTOR_MCP_TOKEN` (the default) serves `--http` **without auth** — `run_http` prints a
loud startup warning in that case. Set it to require a bearer token on every MCP request:

```bash
export RUVECTOR_MCP_TOKEN="a long random secret, not committed anywhere"
ruvector serve --http
```

```bash
curl -H "Authorization: Bearer a long random secret, not committed anywhere" \
     -X POST http://127.0.0.1:8420/mcp -d '{"jsonrpc":"2.0", ...}'
```

The configured token gets full `read`+`write` scope. The write-gated tools
(`vector_create_collection`/`vector_insert`/`vector_insert_batch`/`vector_delete`) call
`_require_write_scope()` internally — a token without `write` scope is rejected on those calls
specifically (relevant if you ever issue a separate read-only token through the same
`TokenVerifier` mechanism).

### ChatGPT Apps SDK widget

`vector_explore` returns a `ui://ruvector/explore.html` resource alongside its JSON result —
when called from a ChatGPT connector that supports the Apps SDK widget convention, this renders
an inline results view instead of (or alongside) the raw JSON. No extra configuration needed
beyond using `vector_explore` instead of `vector_search` from that client.

## Integrations

### LangChain

```bash
pip install 'ruvector[langchain]'
```

```python
from ruvector.integrations.langchain import RuVectorStore

store = RuVectorStore.from_texts(
    texts=["hello world", "goodbye world"],
    embedding=my_langchain_embeddings,     # any langchain-core Embeddings instance
    metadatas=[{"source": "a"}, {"source": "b"}],
)
docs = store.similarity_search("hello", k=1)
```

Verified against `langchain-core==1.6.6`.

### LlamaIndex

```bash
pip install 'ruvector[llamaindex]'
```

```python
from ruvector import Collection
from ruvector.integrations.llamaindex import RuVectorStore

store = RuVectorStore(Collection.create(dim=384))
store.add(nodes)                          # Sequence[llama_index.core.schema.BaseNode]
result = store.query(query_obj)           # llama_index.core.vector_stores.types.VectorStoreQuery
```

Verified against `llama-index-core==0.14.25`. Distance-to-similarity conversion is exact for
cosine (`1 - distance`, matching `ruvector_core`'s own cosine formula) and an approximation
(`1 / (1 + distance)`) for other metrics — check `Collection.metric` if this distinction matters
for your similarity-cutoff thresholds.

> **Known transitive issue, not a `ruvector` bug**: `llama-index-core` depends directly on
> `nltk`, which has an open, unpatched advisory (`PYSEC-2026-3740`, a file-sandbox bypass in
> `nltk`'s own model-persistence helpers) as of this writing. `ruvector.integrations.llamaindex`
> never calls the affected APIs, but if you separately use `llama-index-core`'s NLP features in
> the same process, that exposure is real and not something `ruvector` can pin around.

### Salesforce Agentforce

```bash
pip install 'ruvector[salesforce]'
```

Exposes vector search as Agentforce actions via **External Services + OpenAPI** — the
self-service extension point (Data Cloud's retrieval surface doesn't accept an arbitrary
external vector store as a "BYO retriever", and Agentforce's own MCP client is Beta and
AE-tier-gated; see ADR-352's Integrations section for the full research behind this choice).

```bash
export RUVECTOR_SALESFORCE_ACTION_TOKEN="a long random secret"
export RUVECTOR_ENABLE_SALESFORCE_ACTIONS=1
ruvector serve --http --port 8420
```

This mounts, on the same server/port as the MCP tools:

| Route | Method | What it does |
|---|---|---|
| `/salesforce/openapi.json` | GET | the OpenAPI 3.0 document to register as an External Service (no auth required on this one) |
| `/salesforce/search` | POST | `{"collection", "query_vector", "k", "filter"}` → `{"hits": [...]}` |
| `/salesforce/upsert` | POST | `{"collection", "vector", "metadata"}` → `{"id", "count"}` |
| `/salesforce/ground` | POST | `{"collection", "query_vector", "k", "text_field"}` → `{"context", "sources"}` — joins top-k text for grounding an agent prompt |

These three action routes require `Authorization: Bearer <RUVECTOR_SALESFORCE_ACTION_TOKEN>`
(falls back to `RUVECTOR_MCP_TOKEN` if that specific var is unset) — **not** the same auth
mechanism as the MCP tools above, because `MCPServer.custom_route` (what these are mounted with)
does not go through the SDK's `token_verifier` at all. Error responses are clean 4xx JSON bodies
(`{"error": "...", "error_description": "..."}`), not raw 500s, for `KeyError`/`ValueError`/
`TypeError`/`RuVectorError` — anything else is a real bug in this module and surfaces as an
ordinary 500.

For syncing Salesforce records into a collection for grounding:

```python
from ruvector.integrations.salesforce import SalesforceConfig, fetch_records, sync_records_to_collection

config = SalesforceConfig.from_env()   # RUVECTOR_SALESFORCE_{INSTANCE_URL,CLIENT_ID,CLIENT_SECRET}
records = await fetch_records(config, "SELECT Id, Name, Description FROM Account")
ids = sync_records_to_collection(coll, records, embed_fn=my_embed_fn, text_field="Description", id_field="Id")
```

Example Named Credential / External Service Registration metadata (explicitly marked
illustrative, not validated against a real org) and setup steps:
[`examples/python-salesforce-agentforce/`](../../examples/python-salesforce-agentforce/).

**What was and wasn't verified against a real org**: every test for this integration runs
against `httpx.MockTransport` or an in-process ASGI `TestClient` — no real Salesforce org, OAuth
exchange, SOQL query, or External Service registration happened in this session. The OpenAPI
document is confirmed valid OpenAPI 3.0 (`openapi-spec-validator`); whether it clears Salesforce
External Services' own import constraints can only be confirmed by a real import. See ADR-352's
Integrations section for the complete list.

## Perf tips

- **Prefer `insert_batch`/`from_vectors` over a Python loop of `insert`** calls — one `py.detach()`
  GIL release and one Rust-side batch op instead of N round trips.
- **Pass C-contiguous `float32` NumPy arrays** — `ruvector` reads vectors via `PyReadonlyArray`
  (no copy) but requires contiguity; a non-contiguous array (e.g. certain slices) raises
  `TypeError` rather than silently copying. `np.ascontiguousarray(arr, dtype=np.float32)` if
  unsure.
- **Use a dict filter over a callable filter when the predicate is exact-match** — dict filters
  on the `hnsw` backend are evaluated in Rust per candidate; a callable always round-trips to
  Python per candidate regardless of backend.
- **`filter=None` takes a fast path** on the `hnsw` backend — no overfetch beyond exactly `k`
  candidates from Rust. This was a real perf fix (previously always overfetched 4x even
  unfiltered) — see the ADR's M2 benchmark section.
- **`ruvector --help` stays fast** because every CLI subcommand imports numpy/click/rich lazily
  inside its own function body, not at module load — don't undo this if you're extending `cli.py`.
- **Under `ruvector serve --http`**, all tool calls share one process-wide lock (`RLock`) around
  state that touches the on-disk cache — this serializes concurrent requests for correctness
  (avoids an id-collision race under concurrent inserts, which was a real, reproduced bug this
  session). It's the right tradeoff for a single vector index, not a high-QPS service; `stdio`
  was never affected (one request at a time regardless).

## Benchmarks

Real text embeddings (10,000 docs, `sklearn`'s 20-newsgroups, `all-MiniLM-L6-v2`, 384-dim) vs.
`hnswlib` at matched `m=16, ef_construction=200`, k=10, 200 queries, cosine distance:

| ef_search | ruvector p50 | ruvector QPS | hnswlib p50 | hnswlib QPS | recall@10 | gap |
|---|---|---|---|---|---|---|
| 50 | 0.252 ms | 3,854 | 0.035 ms | 28,104 | 0.999 / 0.999 | ~7.2x |
| 100 | 0.391 ms | 2,388 | 0.059 ms | 16,758 | 1.0 / 1.0 | ~6.6x |
| 200 | 0.611 ms | 1,514 | 0.108 ms | 9,057 | 1.0 / 1.0 | ~5.7x |

Recall is essentially identical to `hnswlib` at matched parameters — the algorithm does the same
job either way. The latency gap is real and named, not hidden: `HnswIndex.search` copies the
query into an owned `Vec<f32>` and builds a Python dict per hit even when metadata is unused,
costs `hnswlib`'s bare-array return doesn't pay. A `search_many` batched entry point (amortizing
marshaling cost over N queries) is the obvious next lever and is not implemented yet. See the
ADR's M1.5 benchmark section for an earlier, worse comparison on adversarial random-Gaussian data
and the two real bugs (a dead `ef_search` per-call kwarg, a 4x overfetch) found and fixed while
chasing this number down — those fixes are already reflected in the table above.

Reproduce: `crates/ruvector-py/benchmarks/bench_hnsw_real_embeddings.py` (this table;
needs `pip install hnswlib sentence-transformers scikit-learn`) and
`bench_compare_rabitq.py` (the earlier RabitqPlus-vs-hnswlib random-Gaussian comparison in the
ADR, superseded as the default backend but kept for the record; needs `pip install hnswlib`).
Both are standalone scripts, not part of the pytest suite (too slow for CI, and the point is a
real side-by-side against a comparator that isn't always installed) — run with
`python benchmarks/bench_hnsw_real_embeddings.py` from `crates/ruvector-py/`. `ruvector benchmark`
in the CLI gives you ruvector's own numbers in-process without a comparator installed.

## Further reading

- [`docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`](../adr/ADR-352-ruvector-python-sdk-cli-mcp.md) — full design rationale, capability-coverage table, security findings, both benchmark writeups
- [`docs/sdk/LOOP-STATE.md`](../sdk/LOOP-STATE.md) — development resume-point notes (gotchas, what's deferred and why)
- [`examples/python-salesforce-agentforce/`](../../examples/python-salesforce-agentforce/) — Salesforce setup walkthrough
