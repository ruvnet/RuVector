# ADR-340: Correctness-Hardening Invariants for Hot-Path Primitives

**Status:** Accepted

**Date:** 2026-08-26
**Owners:** ruvector maintainers
**Tracking:** [#825](https://github.com/ruvnet/ruvector/issues/825), [#901](https://github.com/ruvnet/ruvector/issues/901), [#903](https://github.com/ruvnet/ruvector/issues/903), [#907](https://github.com/ruvnet/ruvector/issues/907), [#908](https://github.com/ruvnet/ruvector/issues/908)

## Context

A cleanup pass across the workspace's open defect backlog closed five issues
that, examined together, are not five unrelated bugs. Each is an instance of a
recurring defect class that this codebase — 170+ crates, several of them
containing independent copies of the same primitive — will keep reproducing
unless the fix is stated as an invariant rather than a patch:

1. **#901** — `ruvector-tiny-dancer-core::Router::route()` sorted candidate
   decisions with `partial_cmp().unwrap()`. One NaN confidence from a
   corrupted or badly quantized model panicked the router instead of
   returning an error. The VoI-gated path already rejected non-finite model
   output; the default (ungated) path did not.
2. **#825** — `ruvector-delta-index::DeltaHnsw::connect_node()` held a write
   lock on a neighbor node while its prune branch re-locked the *same* node
   through `self.distance()`. parking_lot locks are not reentrant, so the
   inserting thread deadlocked against itself — but only on graph topologies
   dense enough to trigger pruning, which an unseeded test RNG made
   nondeterministic. One hung test burned a 3h52m CI job.
3. **#907** — `ruvector-graph`'s global redb pool held strong `Arc<Database>`
   entries forever: an erased-and-recreated path was handed the old handle
   pointing at the unlinked inode (deleted rows reappeared), and the map grew
   one entry per path for the life of the process. `ruvector-core` had the
   same bug fixed in #902; the second copy of the pool kept the pre-fix shape.
4. **#908** — `ruvector-context::require_private_root` verified the index
   root's mode (`0700`) but not its owner, so a substituted root owned by an
   attacker passed the check and failed only accidentally, later, with a bare
   `EACCES`.
5. **#903** — the SOTA benchmark harness canonicalized cache-key payloads by
   sorting object keys with `localeCompare`, making the benchmark cache key a
   function of the machine's locale and ICU build.

## Decision

The following invariants are adopted for all ruvector crates. They bind new
code and any code being touched for other reasons; pre-existing violations are
tracked as issues rather than fixed speculatively.

### 1. Float orderings on untrusted or model-produced values must be total

Any `sort`/`max`/`min`/heap over `f32`/`f64` values that originate outside the
module's own control (model output, user input, deserialized state, quantized
data) must either (a) reject non-finite values at the boundary where they
enter, or (b) use `total_cmp`. Prefer both: rejection keeps a degenerate
producer loud; `total_cmp` guarantees the panic can never come back if the
boundary check regresses. `partial_cmp().unwrap()` and
`partial_cmp().unwrap_or(Equal)` are both non-compliant — the first panics,
the second silently produces an unstable order that varies with element
position.

The pattern applied in #901: one choke point both routing paths share rejects
non-finite score/uncertainty with an error *and* trips the circuit breaker, and
the downstream sort uses `total_cmp` as defense-in-depth.

### 2. No lock re-entry: data reachable from a held guard is read through the guard

A method that acquires a per-node or per-entry lock must not call helpers that
lock the same entry, however indirectly. When a helper needs both mutable and
immutable views of one locked value, split the borrow of the guard
(`let Struct { a, b, .. } = &mut *guard`) rather than re-locking. Helpers that
lock entries by index (`self.distance(idx)` style) must document that
requirement and, where a self-reference is representable (a node listed in its
own neighbor list), filter it out before locking.

Additionally, every graph traversal in an index structure must carry an
explicit progress bound sized by an invariant of the algorithm (for greedy
HNSW descent: strictly decreasing distance ⇒ at most `nodes.len()` moves), so
a malformed graph terminates loudly instead of hanging CI for hours.

### 3. Shared-resource pools hold `Weak`, and eviction happens in `Drop` under the slot guard

The canonical pool shape is `ruvector-core::storage`:
`HashMap<PathBuf, Arc<Mutex<Option<Weak<Database>>>>>`, a `Drop` impl that
clears the slot while still holding the per-path guard (so redb's
fsync-bearing close is never observed half-finished by a concurrent open), and
`release_slot_if_unused` reaping the map entry under the pool lock. #907
ported this shape to `ruvector-graph`, which was the second — and last known —
strong-`Arc` copy. A naive `Arc`→`Weak` swap without the ordered `Drop`
eviction is known-broken (~97% concurrent drop-vs-open failure rate) and must
not be cargo-culted; port the whole shape or none of it.

Any future crate needing a per-path handle pool uses this pattern. The two
regression tests travel with it: erased-path recreation must observe an empty
database, and a barrier-synchronized concurrent drop-vs-open probe must show
zero failures.

### 4. Privilege checks verify identity, not just mode

A filesystem trust check that gates writes (private index roots, lock
directories) must verify **ownership** (`uid == geteuid()`) in addition to
permission bits. A mode check alone accepts a directory the attacker owns.
Crates that forbid `unsafe` obtain the euid via `rustix::process::geteuid()`
(already in the dependency graph via `fs4`), not `libc`.

### 5. Canonical encodings sort by code unit, never by locale

Any canonicalization that feeds a hash, digest, cache key, or signature sorts
object keys with code-unit comparison (`a < b`), per RFC 8785 (JSON
Canonicalization Scheme). `localeCompare` and `Intl.Collator` are forbidden in
these paths: collation varies by locale and ICU build, so the "same" payload
digests differently across a heterogeneous fleet. The regression-test shape
(stub `Intl.Collator` with a reversed comparator and assert the digest is
unchanged) travels with the rule.

### 6. Tests of nondeterministically-triggered behavior are seeded

A test whose failure mode depends on randomized structure (graph topology,
sampled inputs) must use a seeded RNG or fixed fixture, so a triggered defect
is reproducible and the fix is regression-testable. `rand::thread_rng()` in a
test that exercises data-dependent control flow is non-compliant — #825 hid a
guaranteed deadlock behind an unseeded RNG for months.

## Consequences

- The five fixes above land together with regression tests for each
  (`ruvector-delta-index` rejoins CI's `core-platform` shard, ending its
  #825 hold-out).
- The benchmark cache-key change (#903) invalidates existing
  `.metaharness/cache` entries once, by construction; results are recomputed
  on next run. This is the documented, accepted cost of a locale-independent
  key.
- The `ruvector-context` owner check (#908) rejects no working deployment:
  every newly-rejected root already failed later with `EACCES`. It converts a
  confusing late failure into an early, diagnosable one.
- Review posture: a PR that adds a float sort over external values, a new
  handle pool, a mode-only trust check, or a locale-collated digest is asked
  to cite this ADR and either comply or record the exception here.
- Known remaining violations at time of writing are tracked as issues:
  the non-canonicalized pool keys in both storage pools (#907 tail note), and
  any `partial_cmp().unwrap()` sites outside hot paths surfaced by later
  sweeps.
