# Field report: ruvector-postgres 2.0.5 in production agent-memory service (agentbox)

**Date:** 2026-07-22 · **Reporter:** DreamLab AI (agentbox operators)
**Environment:** image `ruvnet/ruvector-postgres:2.0.5@sha256:7fb09d43…` (PG 17.9, extension `ruvector_version()` = 2.0.5, AVX-512 active) · 178k-row `memory_entries`, 384-dim client-side embeddings (Xinference bge-small-en-v1.5) · HNSW `ruvector_cosine_ops`.

Context: we closed a full trajectory→aggregate→distilled-pattern learning loop on this
sidecar and built a recall-regression harness (200 self-recall / 120 true-recall /
26 exact-token stratified queries, median-of-3). The harness surfaced three upstream
findings, each verified live and reproducible.

## Finding 1 — `CREATE INDEX CONCURRENTLY` double-inserts every tuple (HNSW AM)

**Severity: high (silent correctness).**

```sql
CREATE INDEX CONCURRENTLY idx2 ON memory_entries
  USING hnsw (embedding ruvector_cosine_ops) WITH (m='16', ef_construction='128');
-- then any k-NN scan:
SET enable_seqscan = off;
SELECT id, embedding <=> :qv AS d FROM memory_entries ORDER BY embedding <=> :qv LIMIT 10;
```

Every row appears **exactly twice** in the result stream (identical id + distance pairs,
observed on all queries; a 200-query recall probe returned 368 hits from ≤200 possible).
The concurrent build path appears to insert each heap tuple into the graph in both table
passes. A non-concurrent `CREATE INDEX` on the same data is correct. Impact: silent —
duplicates halve effective k, degrading recall with no error. Suggested fix: dedupe TIDs
across the two concurrent-build passes, or reject CONCURRENTLY for the AM until then.

## Finding 2 — SONA engine ignores trajectory dimension; hardcoded 256

**Severity: high for any non-256-dim deployment.**

```sql
-- fresh scope, learn-FIRST with a 384-dim trajectory:
SELECT ruvector_sona_learn('probe384', jsonb_build_object(
  'initial', (SELECT array_agg(0.05) FROM generate_series(1,384)),
  'steps', jsonb_build_array(jsonb_build_object(
    'embedding', (SELECT array_agg(0.04) FROM generate_series(1,384)), 'reward', 0.9)),
  'final_reward', 0.9));
-- → {"status":"learned","steps":1,...}
SELECT ruvector_sona_stats('probe384');
-- → {"embedding_dim":256,"hidden_dim":256,"trajectories_buffered":0,"patterns_stored":0,...}
```

`ruvector_sona_learn` returns `status:learned` but the engine reports `embedding_dim: 256`
and accumulates nothing (`trajectories_buffered`/`patterns_stored` stay 0). Reproduced on
a virgin scope with learn-first ordering, so it is not create-order poisoning: the
`get_or_create_engine_with_dim` path in `src/sona/operators.rs` does not propagate the
detected dim (or the engine constructor pins 256). We fed 405 real judged trajectories
(8,855 steps) — all accepted, all discarded. Request: make the engine dimension follow the
first learn's detected dim (or a config param), and surface a dim-mismatch **error** rather
than a silent no-op learn.

## Finding 3 — scan-time ef_search appears non-configurable (question)

`hnsw_am.rs` documents `SET ruvector.hnsw_ef_search = N` (default 40, "dynamic ef_search
adjustment based on recall target"). On a degraded 178k-row graph with self-recall@10 =
141/200 we swept `ruvector.hnsw_ef_search` 40/100/200/400: recall and latency were both
flat (141, ~4s per 200-query probe). Either the GUC is not read by the scan path, or the
dynamic adjustment overrides it. If the latter, please document; if the former, this
mirrors the known `VectorDB::search ignores SearchQuery::ef_search` note in
`ruvector-sota-bench/src/runners/core_hnsw.rs`.

## Supporting field observation (not a bug)

HNSW graph quality degrades materially under churn: after ~1 year of incremental writes,
a 132k-row bulk ingest and a 2M-row bulk delete, self-recall@10 had fallen to 141/200.
A non-concurrent rebuild (m=16, ef_construction=128; 4m51s at 178k×384, parallel build)
recovered 177/200 self / 109/120 true. A maintenance note recommending periodic REINDEX
after bulk operations would help operators; recall decay is silent without a fixture-based
harness.

We are happy to run candidate fixes against our recall harness and live corpus.
