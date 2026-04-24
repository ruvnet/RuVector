# ADR-156: ruLake as Memory Substrate for Agent Brain Systems

## Status

**Proposed** — positioning addendum, not a replacement. ADR-155 still
governs what ruLake *is* as a crate. This ADR records what it *also
happens to be good for* if an agent brain system wants to sit on top.

## Date

2026-04-23

## Authors

ruv.io · RuVector research. Strategic review 2026-04-22 surfaced the
framing; ADR is a recording of the position, not a commitment to build
the brain system inside ruLake.

## Relates To

- ADR-155 — ruLake: cache-first vector execution fabric (the substrate)
- ADR-154 — RaBitQ rotation-based 1-bit quantization (the compression)
- ADR-057 — Federated RVF transfer learning (PII, DP, lineage)
- RVF Four Laws — appendability, witness, determinism, lineage
- External: RVM / Cognitum — proof-gated mutation (not in this tree)

---

## Context

ADR-155 built ruLake as a cache-first vector execution fabric: a
RaBitQ-compressed cache with witness-addressed sharing, federated
refill, and a bundle protocol for warehouse-driven coherence. The
strategic review (2026-04-22) observed that the measured properties —

- ~1.02× overhead on the cache-hit path vs direct RaBitQ;
- witness-authenticated freshness with three modes (Fresh / Eventual /
  Frozen);
- deterministic compression so two processes reading the same bytes
  produce byte-identical codes;
- content-addressed cache sharing;
- atomic sidecar publish + refresh

— are exactly the properties an **agent brain memory hierarchy** needs
from its storage layer. The question this ADR answers is whether
ruLake should grow *into* that role or stay as substrate underneath it.

## Decision

**ruLake stays as substrate. It is the memory hierarchy, not the brain.**

An agent brain system (the hypothetical consumer) owns:

- Memory *type* semantics — episodic, semantic, procedural, identity,
  policy, observation. These are cognitive labels; the substrate stores
  them as opaque strings if asked, but never interprets them.
- Recall *policy* — which candidates matter, how to combine vector
  similarity with graph neighborhood / recency / trust / contradiction
  / mincut boundary. Substrate returns ranked candidates; the brain
  decides what "best" means.
- Mutation *policy* — when to write, merge, delete, compact, rehydrate.
  Substrate exposes the primitives (prime, invalidate, publish, refresh);
  the brain decides the schedule.

ruLake owns (and already ships as of ADR-155 M1):

| Brain-system concern      | ruLake primitive                                    |
|---------------------------|-----------------------------------------------------|
| Hot memory                | `VectorCache` + RaBitQ codes (measured 1.02× tax)   |
| Warm memory               | `BackendAdapter::pull_vectors` + `RuLakeBundle`     |
| Cold memory               | Backend-adapter contract (Parquet/GCS/BQ/Iceberg)   |
| Freshness contract        | `Consistency::{Fresh, Eventual, Frozen}`            |
| Witnessed state           | SHAKE-256 witness over the bundle                   |
| Cross-process handoff     | Sidecar protocol: `publish_bundle` / `refresh_from_bundle_dir` |
| Observability             | `CacheStats::{hit_rate, avg_prime_ms, last_prime_ms}` |
| Multi-tier eviction       | LRU cap over unpinned entries (`with_max_cache_entries`) |

The addition this ADR *requires* is small — mostly nomenclature:

1. **`Consistency::Frozen`** (shipped 2026-04-22): caller asserts the
   bundle's witness is immutable for the cache's lifetime. Maps to
   "Frozen for audit" from the reviewer's three-mode knob. Explicit
   `refresh_from_bundle_dir` still works; only the *automatic*
   coherence check is suppressed.

2. **Memory-class tag on the bundle (proposed, not yet shipped):**
   `RuLakeBundle.memory_class: Option<String>` — caller-defined,
   opaque, surfaces through stats. Agent systems tag episodic vs
   semantic; ruLake never interprets.

3. **Explicit separation of read / write / compact paths (doc, not
   code):** ruLake v1 is read-optimized and append-only; writes go
   through RVF ingest, not through ruLake. This ADR records that
   split as load-bearing. Compaction belongs to RVM / Cognitum, not
   ruLake.

### The substrate acceptance test

From the reviewer:

> An agent can recall, verify, forget, compact, and rehydrate memory
> across hot, warm, and cold tiers **without knowing where the memory
> physically lives**.

Concretely, this decomposes into six guarantees that ruLake must
preserve. All five that are already shipped are checked; the sixth
(forget) needs an eviction path that doesn't need the original
backend.

| Guarantee | Substrate primitive | Status |
|-----------|---------------------|:------:|
| Recall | `RuLake::search_one`, `search_federated` | ✓ (M1) |
| Verify | `RuLakeBundle::verify_witness` | ✓ (M1) |
| Forget | `invalidate_cache` + `Consistency::Fresh` re-prime | ✓ (M1) |
| Compact | *Out of scope — belongs to RVM/Cognitum* | n/a |
| Rehydrate | `prime` on cache miss (transparent) | ✓ (M1) |
| Location-transparency | `SearchResult` carries `backend` + `collection` — caller never touches `data_ref` | ✓ (M1) |

"Forget" in the full brain sense (crypto-shred the underlying bytes)
stays as the ADR-155 GDPR follow-up; substrate-level forget is the
cache pointer drop + invalidation, which is sufficient for the
agent's recall semantics.

## Alternatives considered

### A. Absorb the brain system into ruLake

Make ruLake own memory classification, recall policy, contradiction
scoring, mincut-based routing. Rejected: violates the substrate
separation. A cache-first execution fabric does not know what
"episodic" means; if it does, it has stopped being a substrate.

### B. Build a new crate for the memory hierarchy

A sibling crate `rulake-memory` on top. Premature — nothing in M1
needs it. The tag + Frozen additions inside ruLake are small and
load-bearing; anything bigger waits for a brain-system consumer to
actually exist.

### C. Fork into two products

"ruLake" for lakehouse use cases, "ruMem" for brain use cases.
Rejected: the measured properties are identical; the split would
double maintenance for no technical win. The positioning serves both.

## Consequences

### Positive

- **Zero-cost positioning.** The substrate is already shipped; ADR-156
  is mostly a naming exercise. Future brain-system work can adopt the
  substrate without waiting for new primitives.
- **Clean separation.** ruLake vs brain keeps the substrate stable
  when cognitive features (contradiction, trust graphs, mincut recall)
  evolve — and they will evolve fast.
- **Frozen-mode opens the audit market.** "Frozen for audit" is
  sellable to compliance-heavy customers who can't tolerate automatic
  coherence drift on historical snapshots.

### Negative

- **Adds a third framing.** ADR-155 v2 said "intermediary", v3 said
  "cache-first fabric", ADR-156 says "memory substrate." Risk that
  customers conflate them. Mitigation: keep the one-liner in ADR-155
  as the authoritative crate description; ADR-156 is the *consumer
  positioning* for agent systems.
- **Tag-on-bundle is a schema change.** Adding
  `memory_class: Option<String>` bumps a backwards-compatible field
  but still needs a `format_version` decision. Deferred until a brain
  system asks for it.
- **"Forget" is ambiguous.** Substrate-level forget (cache pointer
  drop) is not cryptographic forget. Customers with strict GDPR needs
  must pair substrate invalidation with RVF crypto-shredding. This is
  already the ADR-155 position but worth restating.

### Neutral

- Does not commit to building the brain. Does not prohibit it.
- Does not change any public API shipped under ADR-155 — the
  addendum is additive.

## Open questions

1. **Where does `memory_class` live — on the bundle, on the cache
   entry, or both?** Leaning bundle (travels with the witness) but a
   cache-entry override might be useful for mixed-class federation.
   Defer until a consumer asks.
2. **Do we expose per-memory-class stats?** `CacheStats::by_class` as
   a `HashMap<String, CacheStats>` would let a brain operator see hit
   rate per memory type. Tiny change, but scope-creep toward brain
   features.
3. **Is Frozen the right default for bundles produced by RVF ingest
   jobs?** Probably yes — the ingest job already witness-signs the
   output — but codify the handoff rather than leaving it to
   convention.
4. ~~**Does the substrate acceptance test need to live as a runnable
   test?**~~ **Resolved:** shipped as
   `brain_substrate_acceptance_recall_verify_forget_rehydrate` in
   `tests/federation_smoke.rs`. Drives the six-guarantee loop
   (recall → verify → forget → rehydrate → location-transparency,
   compact explicitly deferred per this ADR) against a single
   `LocalBackend`. If this test stays green on every commit, the
   agent-facing memory substrate claim is mechanical, not
   aspirational.
