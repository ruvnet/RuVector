# ruLake — Vector-Native Federation Intermediary

**Status:** Research spike · proposed
**Date:** 2026-04-23
**Branch:** `research/rulake-datalake-analysis`
**Companion ADR:** [ADR-155](../../adr/ADR-155-rulake-datalake-layer.md)

---

## Elevator pitch

**ruLake is a vector-native federation intermediary.** An application or
agent speaks the RVF wire protocol to `rvf-server`; `rvf-server` routes
each query through a planner that dispatches sub-queries to whichever
backend holds the raw vectors — BigQuery, Snowflake, Iceberg, Delta,
S3-Parquet, or a local file. A RaBitQ-compressed cache sits between the
planner and the backends so the hot working set is answered in memory at
~957 QPS / 100 % recall@10 (per
[`ruvector-rabitq/BENCHMARK.md`](../../../crates/ruvector-rabitq/BENCHMARK.md)),
while cold reads fall through to the source of truth.

The shape is **Trino/Presto-for-vectors, not Pinecone-v2**. ruLake does
not own the storage; it owns the wire format, the compression, the cache
coherence protocol, the query plan, and a single governance choke point
across whichever backends are plugged in.

---

## 4-layer architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ L4  Governance                                                   │
│     RBAC · column mask · lineage · GDPR 2-phase delete · audit   │
│     (single choke point across all backends)                     │
├──────────────────────────────────────────────────────────────────┤
│ L3  Query plane                                                  │
│     rvf-server (HTTP/SSE + RVF wire) → planner → router          │
│     Federated ANN: fan-out per backend, merge-by-score, rerank   │
├──────────────────────────────────────────────────────────────────┤
│ L2  Cache + Index                                                │
│     RaBitQ-compressed hot cache · HNSW graph (per collection)    │
│     · deterministic rotation seed · witness-chained manifest     │
├──────────────────────────────────────────────────────────────────┤
│ L1  Backend adapters                                             │
│     ParquetBackend  BigQueryBackend  SnowflakeBackend  …         │
│     IcebergBackend  DeltaBackend     LocalBackend (tests)        │
│     Each adapter: list / pull-vectors / optional-push-down       │
└──────────────────────────────────────────────────────────────────┘
```

App talks to L3 via RVF wire. L3 asks L2 "is this collection cached,
fresh?". Cache miss → L1 pulls vectors from the authoritative backend,
L2 compresses them into RaBitQ codes, L3 answers the query. Cache hit →
L2 answers directly. L4 instruments the whole path.

---

## What changed vs the first cut of this spike

The first cut framed ruLake as a **plug-in**: teach BigQuery to read RVF
via external tables, remote functions, UDF-with-a-RaBitQ-kernel. The
intermediary reframing (this version) swaps the relationship: ruLake is
the front door, backends plug into *it*. Justifications:

- RVF is already format-native, not storage-native. `rvf-runtime`,
  `rvf-server`, `rvf-federation` already assume "we speak RVF over
  whatever bytes you hand us".
- RaBitQ rotation + 1-bit codes are backend-agnostic — compress once,
  serve from any backend.
- Governance (RBAC, PII, lineage) is a single choke point instead of N
  parallel integrations.
- The BigQuery-native integration becomes a Tier-2 push-down
  optimization inside the `BigQueryBackend` adapter, not a new product
  shape.

The cost: ruLake now owns a cache-coherence problem (backend updates
under the cache) and a latency hop for cases where the app is fine
calling a native vector API directly. Those are named in
[`05-performance-budget.md`](05-performance-budget.md) §"Intermediary tax".

---

## Contents

| File | Role |
|---|---|
| [`00-master-plan.md`](00-master-plan.md) | Goal tree, 5 milestones, 12-wk timeline, risk register |
| [`01-architecture.md`](01-architecture.md) | The four layers in detail; interface contracts; query-path walk |
| [`02-datalake-comparison.md`](02-datalake-comparison.md) | Per-backend adapter story: BQ, Snowflake, Databricks, Iceberg, Trino, DuckDB |
| [`03-bigquery-integration.md`](03-bigquery-integration.md) | Tier-2 push-down: what BQ-native compute buys over pure federation |
| [`04-governance-and-compliance.md`](04-governance-and-compliance.md) | The 10 enterprise deal-breakers + what ruLake must own at the choke point |
| [`05-performance-budget.md`](05-performance-budget.md) | Honest numbers (measured vs "target, unmeasured"), intermediary tax analysis |
| [`06-positioning.md`](06-positioning.md) | What ruLake is NOT; hype rubric; 3 win / 3 lose shapes |
| [`07-implementation-plan.md`](07-implementation-plan.md) | Week-by-week 12-wk plan, acceptance tests per milestone, v2 deferrals |

## One-sentence answer to "what is this?"

**Trino for vectors**: you write one query in RVF; ruLake fans out to every
backend that holds a piece of the answer, merges under uniform governance,
and hands you top-k from a RaBitQ-compressed cache sitting in front.
