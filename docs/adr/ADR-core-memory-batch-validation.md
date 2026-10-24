# ADR: Validate memory batches before mutating storage

## Status

Accepted for this implementation; publication requires independent revision review.

## Context

`VectorDB::insert_batch` first stores its entries and updates the search index
only after storage succeeds. The persistent backend aborts its redb write
transaction on a dimension error. `MemoryStorage` instead validated and stored
one entry at a time. A valid prefix followed by a wrong-dimensional vector
therefore returned an error after adding storage-only rows, overwriting existing
vectors/metadata, or consuming automatically generated IDs. The search index
retained its prior state.

The public Rust memory database reproduces all three effects. The Python HNSW
wrapper uses a rectangular array and validates metadata before insertion; it
cannot produce this mixed-dimension batch. This change addresses the Rust API.

## Decision

Validate dimensions for every entry before generating IDs or changing the
memory maps. Retain the existing error type, mutation order for accepted
batches, and public API. This adds one O(n) pass over vector lengths, uses O(1)
additional memory, and does not copy vector data during validation.

## Limits

This is an all-or-nothing boundary for dimension validation, not a transaction
covering concurrent readers, allocator failures, or arbitrary later index
errors. No memory map locking or persistent storage format changes are added.

## Validation

Five public `VectorDB` tests cover an invalid later entry, attempted overwrite
of an existing row and its metadata, generated-ID preservation, an invalid
first entry/empty batch, and successful storage plus search. Original source
fails the first three regressions and passes both controls; fixed source passes
all five. Native Rust execution uses the offline, locked current workspace.
