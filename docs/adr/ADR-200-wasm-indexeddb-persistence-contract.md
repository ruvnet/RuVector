---
adr: 200
title: "ruvector-wasm IndexedDB persistence: never report success before the transaction commits"
status: accepted
date: 2026-10-05
authors: [Reuven Cohen]
project: "ruvector-wasm"
related: []
tags: [wasm, indexeddb, persistence, ruvector-wasm, correctness]
---

# ADR-200 — ruvector-wasm IndexedDB persistence contract

## Status

**Accepted.** Fixes the 0.2.0 defect where `saveToIndexedDB()` reported success
but saved nothing and `loadFromIndexedDB` was not implemented.

## Context

`crates/ruvector-wasm/src/lib.rs` shipped `saveToIndexedDB()` as a stub that logged
a message and returned `Promise::resolve(true)`; no database was opened and no
data written. `loadFromIndexedDB` returned a rejected "Not yet implemented"
promise, and was declared static while the JS wrappers
(`npm/wasm/src/browser.ts`, `index.ts`) called it as an instance method, so it
could never have worked. Callers had no signal that their data was lost.

## Decision

The fix is the one that stops reporting success on failure.

1. **Commit-gated success.** `saveToIndexedDB()` resolves only after the
   transaction fires `complete`. `error` and `abort` (including quota exceeded)
   reject with the `DomException` name and message. Success is never inferred from
   the request's `success` event alone.
2. **Atomic snapshot.** The whole database is one JSON record under key
   `snapshot` in object store `ruvector` (IndexedDB schema version 1), written
   with a single `put`, so a save is all-or-nothing.
3. **Format version.** The snapshot carries `format: 1`. A different value makes
   load reject with "Unsupported saved database format", so old or future data
   fails loudly rather than being misread.
4. **Load is a static async returning a `VectorDB`.**
   `VectorDB.loadFromIndexedDB(name)` rejects when nothing is saved under `name`,
   or the payload is oversized, non-JSON, the wrong format, or inconsistent with
   its stored dimensions. Dimensions, metric and HNSW flag come from the snapshot.
5. **Naming.** The default name stays `ruvector_db_<ms>`. `setDbName(name)` and the
   `dbName` getter let callers pick a stable name. Names must match
   `[A-Za-z0-9_.-]{1,128}`; anything else is rejected before touching IndexedDB.
6. **JS wrappers** call the static loader and pass an optional name to save. No JS
   persistence logic is added; all of it is in Rust.

The snapshot stores vectors and metadata; the HNSW graph is rebuilt on load.

## Consequences

- Positive: a resolved save means the data is durable; failures are visible.
- Positive: untrusted stored payloads are length-, format- and
  dimension-checked and parsed with `serde_json`, which returns errors, not panics.
- Negative: large databases are serialized as one JSON string, costing memory
  and a rebuild of the index on load. Chunked or binary storage can come later
  under a new `format` value without breaking this contract.
- Negative: an instance created without `setDbName` saves under a random name, so
  cross-session reload requires choosing a name.

## Verification

Browser tests in `crates/ruvector-wasm/tests/wasm.rs` (`wasm-pack test --headless
--chrome`): the original red test (save resolved, no database existed),
round-trip restore of vectors and search, missing database, invalid names, corrupt
JSON, unsupported format, and dimension mismatch. A mutation removing the
`put` turns the round-trip test red.
