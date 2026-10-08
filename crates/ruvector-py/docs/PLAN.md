# Implementation plan and architecture decision

1. Verify repository identity, current releases/license and implemented APIs.
2. Select the smallest real native surface that meets vector CRUD and persistence.
3. Add a typed Python facade, structured errors, validation and lifecycle management.
4. Prove behavior against installed native Rust code; build reusable wheel and sdist.

## Upstream inspected

Repository: https://github.com/ruvnet/RuVector
Commit: 5a93328f2fceb0307c25929ed38cd7a0911fdf00
Core manifest version: 2.3.1; workspace/TurboQuant version: 2.3.0.
GitHub's latest listed release at inspection was crates-2026-09-26, published
2026-09-27. This package uses the exact newer inspected commit, not that release.

Evidence files: crates/ruvector-core/src/{types,vector_db,storage,error}.rs,
crates/ruvector-core/src/index/hnsw.rs, crates/ruvector-server/src/{lib,state}.rs,
crates/ruvector-server/src/routes, docs/sdk/INDEX.md and 01-survey.md,
Cargo.toml, patches/hnsw_rs, LICENSE.

## Binding versus client

PyO3/maturin calls VectorDB directly and supports local persistence/reopen without
a separately managed server. The server does have real REST routes, but collection
registration is an in-memory DashMap, adding lifecycle and deployment concerns.
A C ABI for the vector core was not present; the router FFI is a separate module.
A Node subprocess bridge would introduce a second runtime and npm version skew.
Upstream Python SDK materials are roadmap documents, not a viable Python package
to reuse. The first-party roadmap starts with RaBitQ/ruLake; this package instead
implements the existing VectorDB API requested for ordinary vector CRUD.

## Native contract

The extension owns Mutex<Option<VectorDB>>. It decodes typed JSON arguments,
releases the GIL before lock acquisition/work, validates all batch entries,
rejects duplicate IDs, then delegates storage/index/search to upstream methods.
Close releases the same locked handle deterministically. The Python layer validates
finite float32 input, JSON metadata and positive parameters and exposes immutable
dataclasses and typed exceptions.

JSON is an internal conversion boundary, not network transport. It copies vectors
and adds serialization overhead; no zero-copy or benchmark claims are made.
No Python search implementation or mock backend exists.

## Deliberate limits

Filters are upstream's equality AND/post-top-k semantics. Query ef_search exists in
SearchQuery but is ignored by VectorDB::search at this commit, so the wrapper
exposes only effective HNSWConfig.ef_search. Unsupported operators are not emulated.
Dot-product is excluded: native HNSW uses max(-dot, 0), losing positive similarity ordering. Real integration tests detected this; no upstream source patch or Python fallback is substituted.
No upsert is exposed because existing IDs can leave stale HNSW mappings.
No quantization is exposed because not every upstream configuration is effective.
Batch preflight rejects foreseeable invalid writes but cannot add transactional
atomicity to storage plus index. Native deletion tombstones remain until rebuild.
One live handle per persistent path is required for coherent mutation visibility.

## Upstream integration

The binding is `crates/ruvector-py`, an explicitly excluded standalone workspace.
It depends directly on `../ruvector-core` and patches `hnsw_rs` to
`../../patches/hnsw_rs`. An exact HNSW dependency pin keeps newer registry releases from displacing
the existing upstream patch. No existing Rust source is copied or changed in the PR.
The separate Cargo.lock pins the Python binding's dependency graph without
changing the default workspace lockfile or requiring Python in default builds.
Dedicated CI checks Python 3.10 and 3.13 on Linux x86_64, macOS ARM64 and
Windows x86_64, plus Python 3.13 on macOS Intel. Every job runs native tests,
static checks, wheel installation and an offline out-of-checkout sdist rebuild.

The PEP 517 backend delegates wheel/editable builds to maturin. Its small
source hook supplements maturin's path-dependency archive with the existing
HNSW patch and normalizes that patch path; native source files are unmodified.
This handles a reproduced maturin gap: Cargo patch paths otherwise retain the
checkout-relative location and fail outside the repository. The distribution name and any release/platform
matrix remain maintainer decisions; this contribution does not publish packages.
