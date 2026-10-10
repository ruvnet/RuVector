# ruvector — Python SDK, CLI, and MCP server

Ultra-low-latency vector similarity search, backed by a Rust RaBitQ 1-bit
quantization core (`ruvector_rabitq::RabitqPlusIndex`) with native NumPy
interop — zero-copy reads, GIL released on every heavy call. Ships three
things from one `pip install ruvector`:

- **`ruvector`** — the Python library (`Collection`, `RabitqIndex`).
- **`ruvector` console script** — a CLI for scripting/ops
  (`create`/`insert-batch`/`search`/`delete`/`export`/`import`/`info`/
  `benchmark`/`serve`), installed with the base package.
- **`ruvector serve`** — an MCP server (stdio or streamable-HTTP) exposing
  the same surface as tools, plus a ChatGPT Apps SDK `ui://` widget for
  visual search exploration, install with the `mcp` extra.

This crate is the Python half of the ruvector workspace; the underlying
algorithm lives in `crates/ruvector-rabitq/` and is unchanged by this
binding. Scope, benchmarks, and the milestone roadmap (M1 here → M2 ruLake/
HNSW → M3 embeddings → M4 A2A client) are in
[`docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`](../../docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md)
and [`docs/sdk/`](../../docs/sdk/).

## Install

```sh
pip install ruvector            # library + the `ruvector` console script (numpy, click, rich)
pip install ruvector[mcp]       # + `ruvector serve` (MCP server)
pip install ruvector[all]       # everything
```

For local development from a checkout:

```sh
cd crates/ruvector-py
maturin develop --release   # builds the Rust cdylib in-place, links as ruvector._native
pip install -e '.[all]'     # pulls click/rich/mcp for the cli/mcp extras
pytest tests/
```

The `--release` flag matters: a debug build is dramatically slower on the
search loop and will fail the latency acceptance tests.

## 30-second example (library)

```python
import numpy as np
from ruvector import Collection

vectors = np.random.default_rng(42).standard_normal((100_000, 128)).astype(np.float32)
coll = Collection.from_vectors(vectors, rerank_factor=20)

hits = coll.search(vectors[0], k=10)
for h in hits:
    print(h.id, h.score)          # ascending squared-L2 distance

coll.save("my.rbpx")              # writes my.rbpx + my.rbpx.meta.json
coll2 = Collection.load("my.rbpx")
```

`Collection` adds ids, per-vector metadata, soft delete, and filtering on
top of the lower-level `RabitqIndex` (which is still available directly
for the ~50 lines/row case where you don't need any of that):

```python
coll = Collection.create(dim=128)
coll.insert(vectors[0], metadata={"category": "news"})
coll.insert_batch(vectors[1:100], metadatas=[{"category": "sports"}] * 99)

hits = coll.search(vectors[0], k=5, filter={"category": "news"})  # client-side filter
coll.delete(hits[0].id)           # soft delete (tombstone)
coll.vacuum()                     # physically rebuild without tombstoned rows
```

Filtering is applied client-side (the Rust scan has no filter pushdown in
M1.5) — documented in `Collection.search`'s docstring, not hidden.

## CLI

```sh
ruvector create --path my.rbpx --dim 128
ruvector insert-batch --path my.rbpx --vectors vecs.npy --metadata meta.json
ruvector search --path my.rbpx --query q.npy -k 10 --json
ruvector delete --path my.rbpx --id 42 --vacuum
ruvector export --path my.rbpx --out snapshot
ruvector import --path new.rbpx --vectors vecs.npy
ruvector info --path my.rbpx
ruvector benchmark -n 100000 --dim 128     # in-process latency/QPS, no external comparator
ruvector serve                             # MCP server over stdio
ruvector serve --http --port 8420          # MCP server over streamable-HTTP (loopback)
ruvector serve --read-only                 # no create/insert/delete tools
ruvector serve --max-vectors 50000         # cap vectors per collection (default 1,000,000)
```

`serve --http` on a non-loopback `--host` refuses to start unless
`RUVECTOR_MCP_TOKEN` is set to a non-empty secret (bearer auth); an empty or
whitespace-only value counts as unset.

`ruvector --help` starts in ~20ms — every subcommand lazily imports numpy/
click/rich inside its own function body (see `ruvector/cli.py`'s module
docstring), so nothing pays for them until it actually runs.

## MCP server

`ruvector serve` exposes: `vector_create_collection`, `vector_insert`,
`vector_insert_batch`, `vector_search`, `vector_delete`, `vector_stats`,
`vector_list_collections`, and `vector_explore` — a widget-backed tool that
renders results in an HTML table (`ui://ruvector/explore.html`), following
the ChatGPT Apps SDK convention (`_meta["openai/outputTemplate"]`).

Tool arguments name a **collection by name**, never by filesystem path —
names are restricted to `[A-Za-z0-9_-]` and resolved under
`RUVECTOR_MCP_DATA_DIR` (default `~/.ruvector/collections`), so there is no
directory-traversal surface regardless of what a remote MCP client sends.
See `ruvector/mcp_server.py`'s module docstring for the full security
model.

Write size is capped. Per call: 10,000 rows, 2,000,000 floats in total, and
64 KiB of serialized metadata per row. Per collection: 1,000,000 vectors by
default, counting deleted-but-not-vacuumed rows; `vector_insert` and
`vector_insert_batch` fail with a `ToolError` that names the limit once it
would be exceeded. Change it with `ruvector serve --max-vectors N` or
`RUVECTOR_MCP_MAX_VECTORS=N` (the option wins; a non-positive or non-integer
value stops the server at startup). The server re-saves the whole collection
on every write, so keep this cap well below what your disk and latency budget
can take.

```sh
# point an MCP client at: ruvector serve   (stdio)
# or:                      ruvector serve --http --port 8420
```

## API summary (library)

| Call | Returns | Notes |
|---|---|---|
| `Collection.create(dim, *, rerank_factor=20, seed=42)` | `Collection` | empty; lazily builds on first `insert` |
| `Collection.from_vectors(vectors, *, ids=None, metadatas=None, ...)` | `Collection` | bulk build, parallel rotate+pack |
| `coll.insert(vector, *, metadata=None)` / `coll.insert_batch(vectors, *, metadatas=None)` | `int` / `list[int]` | true incremental add, no rebuild |
| `coll.search(query, k, *, filter=None, rerank_factor=None)` | `list[SearchHit]` | `filter`: exact-match dict or predicate callable |
| `coll.delete(id)` / `coll.vacuum()` | `None` / `int` | soft delete / physical rebuild |
| `coll.save(path)` / `Collection.load(path)` | — | `.rbpx` + `.rbpx.meta.json` sidecar |
| `coll.stats()` | `CollectionStats` | count, dim, rerank_factor, memory_bytes, tombstoned |
| `RabitqIndex.build(vectors, *, ids=None, rerank_factor=20, seed=42)` | `RabitqIndex` | lower-level index, no metadata/filter/delete |
| `idx.search(query, k, *, rerank_factor=None)` | `list[(int, float)]` | `(id, score²)` ascending |
| `idx.save(path)` / `RabitqIndex.load(path)` | — | `.rbpx` v1 format |
| `ruvector.RuVectorError` / `ruvector.CollectionError` | exceptions | base / Collection-level |

Non-contiguous or wrong-dtype inputs raise `TypeError` at the boundary
rather than silently copying — predictable beats fast. Ids must fit in
`u32` (the index's storage width); a larger id raises `CollectionError`
(library) or `ValueError` (`RabitqIndex` directly) rather than silently
truncating.

## Benchmark

Real numbers (this host, random-Gaussian data — adversarial for ANN in
general, see the caveat in ADR-352) are in
[`docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`](../../docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md#benchmark-real-numbers-this-session-this-host).
`ruvector benchmark` runs an in-process build+search latency check with no
external comparator.

## Links

  - [ADR-352](../../docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md) — scope, benchmark, security, publishing status
  - [SDK plan and milestones](../../docs/sdk/) — binding strategy + M1-M4 roadmap
  - [`ruvector-rabitq`](../ruvector-rabitq/) — the Rust crate this wraps

## License

MIT, matching the rest of the ruvector workspace (see [`LICENSE`](LICENSE)).
