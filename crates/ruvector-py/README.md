# RuVector Python

An installable, typed Python interface to the real Rust RuVector core. No server,
API key, Node.js process, or embedding model is required.

This experimental interface lives in `crates/ruvector-py` and uses the existing
`ruvector-core` and upstream patched HNSW through local Cargo path dependencies.
The initial tested baseline is core 2.3.1 at commit
`5a93328f2fceb0307c25929ed38cd7a0911fdf00`.
Proposed distribution name: `ruvector-python`; import: `ruvector`.
Nothing is published to PyPI by this change.

## Install

A supplied Linux x86_64 wheel installs without Rust:

```sh
python -m pip install dist/ruvector_python-0.1.0-*.whl
```

Or build from source on CPython 3.10+:

```sh
python -m pip install .
```

Run source builds from this directory in a RuVector checkout. They require Rust/Cargo 1.89+, a C/C++ linker/toolchain, and internet
access to fetch Cargo/build dependencies. The source distribution includes the required local Cargo dependencies;
builds do not clone RuVector. There are no vendored copies in this source tree.
Dependencies are pinned in Cargo.lock. Wheels use CPython's abi3 (3.10+) API.
The tested wheel targets Linux x86_64/glibc >=2.34; macOS, Windows, ARM and other Python
versions require separate verification. Free-threaded Python and PyPy are not
supported claims.

## Use

```python
from ruvector import VectorDB, VectorRecord

with VectorDB(dimensions=3, path="memory.redb") as db:
    ids = db.insert_batch([
        VectorRecord("a", [1.0, 0.0, 0.0], {"tenant": "acme", "text": "first"}),
        VectorRecord("b", [0.0, 1.0, 0.0], {"tenant": "other"}),
    ])
    for result in db.search([1.0, 0.0, 0.0], k=2, filter={"tenant": "acme"}):
        print(result.id, result.score, result.metadata)
    print(db["a"])
    print(db.delete("b"))

# Existing files restore their saved dimensions, metric and HNSW settings.
with VectorDB(path="memory.redb") as db:
    print(db.dimensions, len(db), db.keys())
```

Use `path=None` (the default) for in-memory storage. Data is passed as finite
float32 vectors; Python sequences and NumPy arrays of real numbers work through
copying, with no NumPy dependency. Supply embeddings from your own model. Examples
use explicit numeric vectors to demonstrate retrieval, not semantic embeddings.

## API and semantics

- `insert(vector, id=None, metadata=None) -> str`: returns explicit ID or native UUID.
- `insert_batch(iterable[VectorRecord]) -> list[str]`: native batch insertion.
- `search(vector, k=10, filter=None) -> list[SearchResult]`.
- `search_batch(vectors, k=10, filter=None) -> list[list[SearchResult]]`: native
  search calls in input order under one GIL release; not a parallel search kernel.
- `get(id)`, `db[id]`, `keys()`, `len(db)`.
- `delete(id) -> bool`, `delete_batch(ids) -> list[bool]`.
- `close()` and context manager support release the native handle and file lock.
- `options` reports effective persisted configuration.
- `DistanceMetric` exposes cosine, euclidean and manhattan.
  `HNSWConfig` controls m, ef_construction, ef_search and max_elements.

Search scores are native distances and **lower is better**. HNSW is approximate.
Filters use exact JSON equality, AND across top-level keys, applied **after**
retrieving top-k. This can yield fewer than k matches even when other matching
vectors exist. Increase k deliberately if you need a larger candidate pool.
Empty filters still exclude records without metadata under the upstream rule.
There are no range operators, nested-key paths, or guaranteed filtered top-k.

The wrapper rejects duplicate IDs instead of invoking upstream's unsafe overwrite
behavior. Batch dimensions, nonfinite values and duplicate IDs are checked before
writing. Batch insertion/deletion is not transactional across storage and index;
a native failure can leave partial work. Deletes remove search mappings but leave
HNSW graph tombstones until reopen rebuilds the index. Repeated churn can reduce
returned neighbors; reopening repairs the graph from live persisted vectors.

Persistent writes use native redb transactions. Reopen reconstructs HNSW from
stored vectors. Use one live handle per file: upstream pools storage handles but
separate indexes do not synchronize mutations. Cross-process file locking is
owned by redb. Individual operations on one handle are serialized in Rust and
release the GIL, including close; no asynchronous API is provided.

Errors: `RuVectorError` with `InvalidVectorError`, `DimensionError`,
`DuplicateIDError`, `StorageError`, `IndexError`, `ClosedError`.
Missing `get` returns None, missing delete returns False, and missing `db[id]`
raises KeyError. A closed handle raises ClosedError on data operations; close
is idempotent.

## Scope and evidence

Included: native HNSW, in-memory/redb persistence, metadata, equality filters,
three verified metrics, single/batch CRUD, typed results/errors, GIL release and examples.
Excluded: dot-product search (upstream HNSW clamps positive similarities to tied zero distances), embedding generation, quantization, graph/GNN, SONA/learning,
RaBitQ/ruLake, distributed collections, REST clients and async wrappers.
These are separate upstream APIs with separate dependencies and maturity;
this release does not claim parity with the npm umbrella package.

See [design rationale](docs/PLAN.md) and [verification](docs/VERIFICATION.md).
This implements VectorDB rather than the separate RaBitQ/ruLake SDK roadmap.
The existing core, TurboQuant and patched `hnsw_rs` Rust sources are unchanged.

## Development

```sh
python -m venv .venv
. .venv/bin/activate
pip install 'maturin>=1.13,<2' pytest ruff mypy build
maturin develop --release --locked
pytest
ruff check python tests examples scripts
ruff format --check python tests examples scripts
mypy
cargo fmt --check
cargo clippy --locked --lib -- -D warnings
maturin build --release --locked --out dist
python -m build --sdist --no-isolation --outdir dist
```

Test details and platform limitations are recorded in docs/VERIFICATION.md.
The binding is a standalone Cargo workspace, explicitly excluded from the parent
workspace so default Rust builds do not require Python.
`.github/workflows/python-bindings.yml` runs its native and packaging checks.

The PEP 517 backend delegates to maturin and normalizes the HNSW patch path
when building an sdist, including the original patch sources in that archive.
Use the documented PEP 517 source-build command instead of raw maturin sdist.
