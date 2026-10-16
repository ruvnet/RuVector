# ADR-GL-007: Immutable prepared snapshots for repeated host queries

Date: 2026-10-16
Status: Accepted

## Problem

Repeated Rust host queries over the same snapshot redo snapshot validation,
index construction and canonical hashing. The agent memory benchmark makes this
cost visible. Reducing validation or omitting receipt fields would weaken the
contract. Snapshot reuse must not reuse authorization, time or model responses.

## Decision

Add `runtime::PreparedSnapshot::new(&snapshot)` and
`prepared.execute(&program, &context, predictor)`. Preparation retains borrowed
graph indexes, serialized size, validation/index logical work and the canonical
snapshot hash. Rust's immutable lifetime borrow prevents safe mutation or freeing
of the source snapshot while preparation is alive. There is no global cache,
identity-only lookup, disk cache, unsafe code, new dependency or wire change.

Every query still validates the program and context, checks host grants, applies
fresh logical time, calls any supplied predictor, executes all operations and
hashes the complete receipt. It charges precisely the same deterministic work as
the ordinary execution path, including reused work. Work is a stable logical
budget, not actual CPU time. Existing receipt hashes therefore remain equal.

The ordinary execute function remains available. Prepared queries are a Rust host
API; process SDK calls do not retain state between processes. Audit interventions
change snapshots and continue full independent recomputation. To change the
snapshot, drop preparation, update or replace the snapshot, and prepare again.

## Tradeoffs and rejected experiment

Preparation adds retained indexes proportional to snapshot records and one hash.
The original snapshot is borrowed, not copied. There is an upfront preparation
cost and no benefit promised for a single query. API input validation and existing
100,000-record and 16 MiB serialized bounds remain. Preparation has a hard maximum
logical validation/index budget of one billion and must run only on host-supplied
snapshots. It does not grant graph.read or decision.emit capabilities.

An earlier 8 KiB BufWriter around the JSON-to-SHA256 stream preserved hashes but
showed mixed timings, including regression and large movement in the unchanged
control. It was reverted. The experiment also overlapped local test/build work,
so it cannot support a performance claim. Raw rejected measurements are retained
in hash-buffer-benchmark.json. No buffer code is shipped.

Mutable revision-keyed caches were rejected because incorrect invalidation can
silently produce stale evidence. Reducing logical budget charges was rejected
because it would alter receipts and make a request's budget depend on cache state.

## Verification and acceptance

96 generated query/clock combinations compare complete cold and warm receipts.
Tests cover low budgets, revoked capabilities, invalid contexts, malformed
snapshots and fresh predictor calls. The application benchmark asserts full
receipt equality before measuring both paths. Alternating old/new process blocks
compare old receipt hashes, runtime work, audit work and intervention deltas.
Measure preparation separately and disclose warm-only and amortization results.
All previous SDK, CLI, native, canonicalization and audit tests must remain green.

## Rollback

Remove the additive API and return execute to its original internal path. No
stored artifact migration, package rename, receipt version change or data rewrite.
