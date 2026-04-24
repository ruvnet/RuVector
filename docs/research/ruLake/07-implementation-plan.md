# 07 — Implementation Plan

12 weeks, 5 milestones, week-by-week. This plan matches the goal tree
in `00-master-plan.md` and assumes an average of ~1.7 FTE across the
spike window. Solo-dev mode stretches to ~20 weeks; the dependency
DAG still holds.

---

## Ground Rules

- New code lives under `crates/` following the existing RVF workspace
  convention. Proposed new crates: `rvf-parquet`, `rvf-object-store`,
  `rulake-udf`, `duckdb-rulake`, `rulake-catalog`.
- No new files in the repo root. Tests under `crates/*/tests/`,
  integration tests under `crates/rvf/tests/rulake-integration/`.
- All numbers in docs carry "target, unmeasured" unless a
  reproducible command is named next to them.
- Every week ends with a green-or-red gate. Red gates are named here,
  not invented during the week.

---

## Week 1 — M1 scoping (0.5 E-wks)

**Goal.** Inventory the format bridge surface area so week 2–3 is
mechanical.

Work:
- Walk `crates/rvf/rvf-types/src/segment_type.rs` and enumerate every
  segment type we need to propagate into Parquet metadata
  (VEC_SEG, QUANT_SEG, INDEX_SEG, META_SEG, META_IDX_SEG, WITNESS_SEG,
  CRYPTO_SEG, MANIFEST_SEG). Output: a short mapping table in
  `crates/rvf/rvf-parquet/docs/segment-mapping.md`.
- Confirm `FixedSizeList<Float32, D>` round-trips through
  BQ / Snowflake / DuckDB / Trino Parquet readers. This is R10 in the
  risk register.
- Verify BQ remote function request/response body caps against
  current GCP docs.

Gate: segment mapping table lands and the four Parquet-reader
confirmations are in writing. If `FixedSizeList<Float32>` is not
universally accepted, fall back to `BYTES` + length prefix and absorb
1.5 E-wks.

## Week 2 — M1 format spec (1.5 E-wks)

**Goal.** Freeze the ruLake-over-Iceberg wire shape.

Work:
- Author `docs/adr/ADR-155-rulake-datalake-layer.md` companion sections
  for Iceberg v2 manifest conventions.
- Write the `table.rulake.json` bundle manifest schema (documented in
  `crates/rvf/rvf-parquet/docs/bundle-manifest.md`, JSON schema under
  `crates/rvf/rvf-parquet/schemas/`).
- Decide on the new SegmentType discriminator for RaBitQ codes
  (proposal 0x24 per `01-architecture.md`); if accepted, extend
  `rvf-types::SegmentType` with the new variant, plus the `TryFrom`
  impl and discriminant tests.

Gate: JSON schema validates against a hand-crafted example;
`segment_type.rs` tests pass with the new discriminant.

## Week 3 — M1 round-trip tests (1.5 E-wks)

**Goal.** Prove the bridge round-trips.

Work:
- Build `rvf-parquet::ParquetBridge` enough to emit a single
  RecordBatch from a live `RvfStore`.
- Write a test that ingests 1000 f32 vectors via `rvf-import`,
  exports to Parquet via `rvf-parquet`, and reads back via
  `arrow-rs`. Byte-compare.
- Write an Iceberg v2 manifest for the Parquet file using the
  `iceberg-rust` crate (verify crate maturity before committing).

**Gate (M1 exit):** round-trip test green, Iceberg manifest validates
against an Iceberg catalog (Polaris / Nessie — test server is fine).

## Week 4 — M2 remote function harness (2.0 E-wks)

**Goal.** A Cloud Run service that BigQuery can call with a fixed
fixture response.

Work:
- New crate `rulake-udf` with an axum HTTP server exposing
  `POST /search` accepting BQ remote-function request body shape:
  ```json
  {
    "sessionUser": "...",
    "requestId": "...",
    "calls": [ [query_vec, k, rerank_factor], ... ]
  }
  ```
  and returning:
  ```json
  { "replies": [ "json-encoded array of {id, distance}", ... ] }
  ```
- Package as a container. Deploy to Cloud Run us-central1 via
  `scripts/deploy-rulake-udf.sh` (new).
- Register as a remote function in a test BQ dataset. Call it with a
  hand-crafted query. Verify the fixture flows.

Gate: BQ `SELECT ruLake.search_test(...)` returns the fixture reply.

## Week 5 — M2 kernel packaging (2.0 E-wks)

**Goal.** The UDF runs the actual RaBitQ kernel.

Work:
- Wire `ruvector-rabitq::RabitqPlusIndex` into the `rulake-udf`
  warm state. Open the `.rvf` bundle lazily on first request.
- New crate `rvf-object-store` implementing the `ObjectStore` trait
  from `01-architecture.md` against GCS using `google-cloud-storage`
  (verify crate; fallback is raw `reqwest` against the JSON API).
- Cache rotation matrix + Level-A hotset in the container memory
  across calls. Layer-B/C loaded on demand.

Gate: UDF returns real RaBitQ+ top-k against a 100k fixture on GCS;
warm p50 under 200 ms round-trip from BQ.

## Week 6 — M2 fixture validation (1.5 E-wks)

**Goal.** Sanity-check recall against ground truth.

Work:
- Upload a 100k-vector BENCHMARK.md fixture to GCS.
- Run 200 queries through the BQ remote function.
- Compare top-10 IDs to Flat-exact top-10 computed in-process.
- Land numbers in `05-performance-budget.md` (replacing "target,
  unmeasured" for the scan-time lines).

**Gate (M2 exit):** recall@10 = 100% on the fixture; p50 warm
≤ 200 ms. If 95%–100% but not exactly 100%, ship with a note; if
< 95%, rerank_factor is too low — tune before advancing.

## Week 7 — M3 external table wiring (2.0 E-wks)

**Goal.** The `SELECT ... FROM external_table, search(...)` shape
works.

Work:
- Write a BigLake external table definition pointing at the Parquet
  side of a ruLake bundle.
- Write the `CREATE FUNCTION ruLake.SEARCH(...) RETURNS ARRAY<STRUCT<...>>`
  SQL that delegates to the remote function.
- SQL pattern:
  ```sql
  WITH hits AS (
    SELECT s.id AS hit_id, s.distance
    FROM UNNEST(ruLake.SEARCH(@q, 10, 20)) AS s
  )
  SELECT e.*, h.distance
  FROM `ds.emb` e JOIN hits h ON e.id = h.hit_id
  ORDER BY h.distance;
  ```
- Document the pattern in `crates/rvf/rulake-udf/docs/bigquery.md`.

Gate: the pattern runs against the 100k fixture and returns correct
rows.

## Week 8 — M3 1M-vector acceptance (2.0 E-wks)

**Goal.** Hit the exit criterion from `03-bigquery-integration.md`.

Work:
- Generate 1M-vector BENCHMARK-style fixture; upload.
- Run 200 queries with rerank_factor=20.
- Measure p50/p95 warm, cold-start p50.
- Update `05-performance-budget.md` with measured lines.

**Gate (M3 exit):** recall@10 == 100%, p50 warm ≤ 300 ms (looser
than target to absorb GCP variance), cold start ≤ 5 s. Miss → replan
or re-scope to 100k as v1 scale.

## Week 9 — M4 governance primitives (2.0 E-wks)

**Goal.** Lineage and GDPR phase-1 visible.

Work:
- New crate `rulake-catalog` with a `DataplexAdapter` that:
  - Reads the BQ job id from the UDF request (BQ sends this in
    headers — verify before relying).
  - Emits a lineage edge via Data Lineage REST.
- Implement GDPR phase-1 (logical delete): a management endpoint
  `POST /admin/delete` that appends a JOURNAL_SEG tombstone, then
  the next query returns no hit.
- Unit test: `delete → query → expect no hit`.

Gate: Dataplex UI shows a lineage edge for a BQ query; the tombstone
test is green.

## Week 10 — M4 Tier-2 adapters (2.5 E-wks)

**Goal.** Iceberg + DuckDB demos green.

Work:
- Iceberg v2 manifest writer in `rvf-parquet`. Test against Polaris
  test server.
- New crate `duckdb-rulake` using `duckdb-rs`. Register:
  - Scalar: `rulake_distance(lhs BYTES, rhs FLOAT[D]) -> FLOAT`
  - Table: `rulake_search(bundle STRING, q FLOAT[D], k INT) -> TABLE`
- Example script in `examples/rulake-quickstart/` demonstrating the
  same `.rvf` bundle read by DuckDB locally and by BQ remotely.
- `ruLake inspect` CLI subcommand in `crates/rvf/rvf-cli` walking the
  witness chain and printing an integrity report.

**Gate (M4 exit):** examples script runs end-to-end on two laptops
(analyst laptop with DuckDB; engineer laptop with `bq` CLI).

## Week 11 — M5 measurement + failure-mode docs (1.5 E-wks)

**Goal.** Numbers and operator docs in the repo.

Work:
- Finalise `05-performance-budget.md` with all M2/M3 measurements.
- Write the operator runbook: `docs/ops/rulake/` (new directory)
  containing: deployment, scaling, cold-start mitigation, GDPR
  two-phase delete, forensic freeze, incident response.
- Unity Catalog + Polaris lineage sketches (one page each, code
  left as "follow-up").
- Write a one-page "known limits" appendix listing every named
  trade-off from `01–06`.

Gate: docs land in repo, operator runbook reviewed by someone who
has been on-call for a GCP service in the last 12 months.

## Week 12 — M5 public spike + ADR flip (1.0 E-wks)

**Goal.** Close the spike cleanly.

Work:
- Polish `examples/rulake-quickstart/` to one-copy-paste runnable.
- Flip `docs/adr/ADR-155-rulake-datalake-layer.md` from "Proposed"
  to "Accepted" OR document the blockers and flip to "Rejected
  (deferred)". No silent "Draft" parking.
- File v2 follow-up work as individual issues (see deferred list in
  `00-master-plan.md`).
- Final review of every "verify before relying" tag in the docs —
  either verified (remove tag) or stale (escalate).

**Gate (M5 exit):** ADR accepted OR rejected. Spike ends with a
binary outcome. No zombies.

---

## Acceptance Tests by Milestone

### M1
- `cargo test -p rvf-parquet --test roundtrip` green.
- `iceberg-rust` validates our manifest.
- Bundle manifest JSON schema validates.

### M2
- `cargo test -p rulake-udf --test fixture_100k` green.
- Deployed Cloud Run service handles 200 sequential queries without
  error.
- recall@10 = 100% on BENCHMARK.md 100k fixture.

### M3
- `scripts/m3-acceptance.sh` runs end-to-end against BQ us-central1.
- `05-performance-budget.md` has measured p50/p95 warm, cold-start
  p50 — not "target, unmeasured."

### M4
- Dataplex shows a lineage edge from the BQ job id to the bundle
  hash.
- `cargo test -p rulake-catalog --test gdpr_phase1` green.
- `examples/rulake-quickstart/` runs under DuckDB locally.

### M5
- ADR-155 state is "Accepted" or "Rejected."
- `docs/ops/rulake/` exists.
- No "target, unmeasured" tags remain in `05-performance-budget.md`
  for the scan-path lines.

---

## Go / No-Go Decision Table

| End of | Condition                                               | Go               | No-go               |
|--------|---------------------------------------------------------|------------------|---------------------|
| Week 3 | Round-trip test green; Iceberg manifest validates       | Advance to M2    | Fall back to "RVF side-by-side" — no bridge; close spike |
| Week 6 | recall = 100% on 100k fixture; warm p50 ≤ 200 ms        | Advance to M3    | Tune rerank / scan; if still red, drop BQ, promote DuckDB |
| Week 8 | recall@10 = 100% on 1 M fixture; p50 warm ≤ 300 ms      | Advance to M4    | Re-scope v1 scale to 100k; extend timeline 2 wks       |
| Week 10| Dataplex lineage visible; DuckDB example runs            | Advance to M5    | Ship without lineage; mark "v2"                         |
| Week 12| ADR flipped                                              | Ship             | Close as "not yet ready"                                |

---

## Deferred to v2 (with E-wk estimates so future planners know the scale)

| Item                                                       | E-wks |
|------------------------------------------------------------|------:|
| Snowflake external function path                           | ~5    |
| Databricks Photon UDF path                                 | ~7    |
| Trino connector (Rust sidecar)                             | ~8    |
| Multi-region active-active replication                     | ~10   |
| Per-row encryption + crypto-shredding                      | ~6    |
| GPU (CUDA) RaBitQ kernel                                   | ~4    |
| HNSW + RaBitQ integration inside rvf-index                 | ~3    |
| SIFT1M / GIST1M / DEEP10M acceptance suite                 | ~1    |
| SIMD popcount via std::arch                                | ~1    |
| Parallel scan (Rayon) at UDF level                         | ~0.5  |
| Delta Lake direct writer                                   | ~4    |
| Hudi integration                                           | ~4    |
| Azure Synapse / Fabric integration                         | ~4    |
| ClickHouse integration                                     | ~3    |
| SONA federated-learning bridge                             | ~5    |

The v2 backlog is ~65 E-wks. If the v1 spike is accepted at M5, v2
is a team-sized effort (4 engineers × 4 months), not a spike.

---

## How to Read This Plan If You Are the Engineer

- Start with `01-architecture.md` and the crates it names. You will
  spend week 1 in `crates/rvf/rvf-types`, `crates/rvf/rvf-wire`,
  `crates/rvf/rvf-index`, `crates/rvf/rvf-manifest`, and
  `crates/ruvector-rabitq`.
- Every week's "work" list is the minimum. If something on that list
  takes less than the named time, move to the next week's list — do
  not invent scope.
- Every gate is a commit that either lands or triggers the named
  mitigation. No silent slips.
- If you can only do one thing per week, do the measurement, not the
  feature. A measured M2 with a narrow feature beats an unmeasured
  M3 with everything.
