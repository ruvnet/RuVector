# ADR-GL-008: Exact top K selection and borrowed node sets

Date: 2026-10-16
Status: Accepted

## Context

Prepared execution still constructs node sets, ranks vectors and derives evidence
on every query. The current exact cosine implementation sorts all N candidates
even when K is small, and intermediate node sets repeatedly allocate owned IDs.
Both costs can be reduced without changing inference, authority or wire formats.

## Decision

1. Score every candidate with the existing f64 arithmetic and validate all vector
   dimensions. Partition the candidate array with Rust slice selection when K<N,
   retain the first K and sort those winners using descending score and ascending
   node ID. The comparator defines the same total order as the old full sort.
2. Store borrowed node ID slices in intermediate interpreter values. IDs refer to
   the immutable input snapshot, never temporary program strings. Selection,
   traversal and similarity propagate those borrows; evidence, decisions and
   predictor requests own strings at the existing public boundaries.

The selection path avoids sorting the tail and performs no approximate search or
candidate pruning. Scratch scoring remains O(N); no claim of O(K) scratch space
is made. All vectors are examined, including malformed candidates outside the
winning set. Missing or zero-norm vectors remain excluded; score ties still use
node identity. Full order of the top K is preserved.

The historical N*log_work(N) ranking charge remains. Intermediate node sets retain
the same conservative byte accounting as owned strings. These logical bounds
intentionally do not get cheaper when the implementation does: existing budgets,
receipts and replay remain invariant. No application source changes are needed.

## Safety and tradeoffs

No unsafe code or new dependencies. Rust ties borrowed intermediate values to the
snapshot lifetime; public receipts remain independent owned values. Every query
still checks grants and context, and prepared snapshots remain immutable.
Borrowing reduces ID allocation pressure but not snapshot storage. Exact ranking
still scans N vectors; high dimensional embeddings may remain compute dominated.
Neither change is an approximate nearest-neighbor accelerator.

## Acceptance

Two new tests compare selection against an independent full-sort reference over
16 generated datasets and K=1,4,32,64. Cases include exact ties, negative vectors,
zero vectors, missing vectors and maximal finite f32 values. A dimension mismatch
outside top K must still fail. Prepared and ordinary receipts must agree.

Paired release binaries must preserve receipt hashes, deterministic runtime work,
audit work and intervention deltas. Record cold, warm, preparation and audit
measurements separately, retain all raw samples, and do not generalize the
synthetic two-dimensional workload to production embeddings. Existing SDK, native,
provenance, lifetime and budget tests remain required.

## Rollback

Restore owned intermediate IDs and full candidate sorting. No schema, package,
artifact or stored receipt migration is necessary.
