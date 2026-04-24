# 03 — BigQuery Integration (Tier 1)

BigQuery is the load-bearing v1 target. This doc enumerates three
candidate integration architectures, names their costs and compromises
head-on, and picks the primary. The picked architecture is what
`07-implementation-plan.md` builds.

---

## What We Need BigQuery To Do

Analyst runs a SQL query that looks like:

```sql
SELECT id, title, score
FROM `project.ds.embeddings` AS e,
     `project.ds.LIB`.RULAKE_SEARCH(
        @query_vec,
        k => 10,
        rerank_factor => 20
     ) AS s
WHERE e.id = s.id
ORDER BY s.distance
LIMIT 10;
```

Where `RULAKE_SEARCH` is whatever UDF / remote function / BigLake table
valued function we ship. The exit bar: this query returns correct top-k
against a 1 M-vector corpus, 100% recall@10 vs the exact answer, within
a latency budget `05-performance-budget.md` commits to.

---

## Candidate A — Parquet Bridge + BigLake External Table + Remote Function

```
GCS bucket
 ├── embeddings_v1/data_*.parquet        (Arrow FixedSizeList<Float32, D>)
 ├── embeddings_v1/vec.rvf, idx.rvf, ... (opaque RVF bundle, read by UDF)
 └── embeddings_v1/table.rulake.json     (bundle manifest)

BigQuery
 ├── BigLake external table over the Parquet files
 ├── Remote function  ruLake.SEARCH(vec, k, rerank)
 │     → HTTPS → Cloud Run service (Rust, stateful across cold-starts)
 │     → Cloud Run service reads the `.rvf` bundle from GCS via range reads
 │     → Runs RaBitQ+ symmetric scan + rerank×N
 │     → Returns top-k {id, distance} as a repeated JSON field
 └── Lineage → Dataplex via our adapter emitting edges from job metadata
```

### Pros

- Every piece is a **standard BQ primitive**. No private APIs.
- BigLake + Parquet is the normal enterprise ingest path. Security
  review is familiar territory.
- The `.rvf` bundle is the same bundle DuckDB, edge, and WASM runtime
  read. True portability.
- RaBitQ warm state (rotation matrix, Level-A hotset) stays in Cloud
  Run memory; per-call overhead drops after first warm call.

### Cons / What we give up

- **Round-trip latency.** Every remote function call is HTTPS ~30–80 ms
  p50 just for the Cloud Run hop. Not sub-ms. See
  `05-performance-budget.md`.
- **Payload caps.** BQ remote function request and response bodies are
  capped at 10 MB each (verify before relying; this has changed). For
  k > ~1000 we must fan out.
- **Cost.** Cloud Run compute + egress. At heavy query load, cheaper
  than BQ slot time; at light load, more expensive than doing nothing.
- **Parquet vector column is semi-opaque.** BQ will read the `vec`
  column as `ARRAY<FLOAT64>` (or `ARRAY<FLOAT32>` depending on Parquet
  logical type support in the current BQ release — verify). Good for
  display, not useful for native filter pushdown because RaBitQ bits
  live in the sidecar `.rvf`, not in the Parquet column.

### Engineer-weeks

- Parquet bridge: 2.0
- Cloud Run service (the UDF): 2.5
- BigLake table + SQL TVF glue: 1.0
- **Total: 5.5 E-wks**

---

## Candidate B — Iceberg Catalog + Trino/BigQuery-via-BigLake

```
GCS bucket
 ├── iceberg/embeddings_v1/metadata/...  (Iceberg v2 tables)
 └── iceberg/embeddings_v1/data/...      (Parquet + .rvf sidecars)

BigQuery (BigLake)
 ├── Iceberg external table via BigLake Iceberg integration
 └── No native kernel integration (just column access)

Separate query engine (Trino / Spark / DuckDB)
 └── Runs the RaBitQ kernel via its own UDF path
```

### Pros

- Single source of truth. One Iceberg table is readable by BQ, Trino,
  Snowflake (via Polaris), Databricks, DuckDB — all at once.
- Governance lives in the Iceberg catalog (Polaris / Nessie / Glue /
  Dataplex). Lineage is clean.
- Matches the open-lake trend.

### Cons / What we give up

- **No BQ-native vector search.** BQ can read the Iceberg table but
  cannot call our kernel. Customers wanting "do it in BQ" get nothing.
- **Requires a second query engine** for the acceleration path
  (Trino/Spark/DuckDB). Doubles the ops surface.
- **BigLake Iceberg support maturity** — has been improving since 2024
  but still has edge cases around snapshot evolution and schema
  changes. **Verify before relying** each quarter.

### Engineer-weeks

- Iceberg v2 manifest writer: 1.5
- BigLake wiring: 0.5
- Separate Trino/DuckDB acceleration path: 3.0
- **Total: 5.0 E-wks** — but only half of a BQ story

---

## Candidate C — Native BQ Kernel via JS UDF + Binary Blob

```
BigQuery
 ├── Native BQ table with BYTES column = RaBitQ 1-bit codes + rotation metadata
 ├── JavaScript UDF:  ruLake.ANGULAR_DISTANCE(query_code BYTES, stored_code BYTES)
 │     → XNOR-popcount-and-angular-estimator in JS
 └── Standard SQL rewrites: SELECT id, ANGULAR_DISTANCE(...) AS d ORDER BY d LIMIT k
```

### Pros

- **Zero external services.** No Cloud Run, no API Gateway, no sidecar.
- Billing is pure BQ slot time — already a budget line item.
- Lineage is automatic — it is a native BQ query.

### Cons / What we give up

- **JavaScript UDFs are slow.** V8 inside BQ is ~1000× slower than
  native RaBitQ. At n=1M, D=128, a JS popcount loop will be ~seconds
  per row, not microseconds. Fundamentally unfit for scale beyond
  ~100k vectors.
- **No HNSW.** JS UDFs cannot maintain a warm graph structure between
  calls. We lose the biggest win of RVF's Layer-A/B/C progressive
  index.
- **No rotation caching.** Every call re-materialises the D×D rotation
  matrix. At D=1536 that is 3.6 B ops per call. Unusable.
- **BQ persistent Python UDFs** (2024 preview; verify current GA
  status) could fix some of this — Python in a container, can call
  native libs via cffi, can cache state per slot. But: a) same cold
  start cost as remote function, b) BQ pricing for Python UDFs was not
  clearly better than remote function cost in our last check
  (verify), c) less control over memory than Cloud Run.

### Engineer-weeks

- JS UDF kernel: 1.0
- BQ schema + ingest path: 0.5
- **Total: 1.5 E-wks** — but does not clear the acceptance bar above
  ~100k vectors. Only usable as a fallback for tiny tables.

---

## Pick: Candidate A (Primary), With Candidate B as the Lake-Distribution Fallback

**Why A as primary:**

- It is the only option that hits "real BQ query, real scale, real
  performance" on day one.
- Every piece is a standard BQ primitive, which keeps the security
  review tractable.
- The Cloud Run UDF is reusable as-is for Snowflake external functions
  and for a generic HTTPS vector-search endpoint.
- Warm-state caching (rotation matrix, HNSW hotset) maps cleanly onto
  Cloud Run's concurrency model.

**Why B in parallel, not instead:**

- Iceberg distribution is cheap (1.5 E-wks) and gives us Tier-2
  compatibility for free once the Parquet bridge exists. We write the
  Iceberg manifest as part of M1.
- We do **not** do the "separate Trino/Spark engine" half of Candidate
  B in v1 — that is deferred.

**Why not C:**

- JS UDF is a tech-demo, not a production path. We document it as a
  "cheap fallback for < 100 k vectors, no HNSW" in the docs but do
  not maintain it.

---

## Picked Architecture in Detail

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        BigQuery                                          │
│ ┌────────────────────────┐   ┌────────────────────────────────────────┐  │
│ │ BigLake external table │   │ Remote function: ruLake.SEARCH(...)   │  │
│ │  (Parquet, `vec`)      │   │  signature: (vec ARRAY<FLOAT64>,       │  │
│ │  schema + RBAC + audit │   │             k INT64,                    │  │
│ │  masking applied here  │   │             rerank_factor INT64)        │  │
│ └────────────┬───────────┘   │  returns: ARRAY<STRUCT<id INT64,        │  │
│              │               │                       distance FLOAT64>>│  │
│              │               └──────────┬──────────────────────────────┘  │
└──────────────┼──────────────────────────┼──────────────────────────────────┘
               │                          │ HTTPS (JSON batch)
               ▼                          ▼
        ┌──────────────────────────────────────────────┐
        │           Cloud Run: ruLake-udf              │
        │  Rust binary, 1 container, stateful warm set │
        │  - HTTP server (axum) on port 8080           │
        │  - Warm state: RVF manifest, rotation matrix,│
        │    Layer-A hotset (~4 MB), RaBitQ codes      │
        │  - Per-call: parse JSON → RaBitQ+ scan →     │
        │    rerank×N → JSON response                  │
        │  - Cold start: pull 4 KB Level-0 root, then   │
        │    4 MB Layer A; subsequent requests warm.   │
        └──────────────┬───────────────────────────────┘
                       │ GCS range reads (HTTP Range)
                       ▼
        ┌─────────────────────────────────────────────┐
        │           GCS bucket: gs://.../embeddings_v1│
        │  Parquet files       (BQ reads these)        │
        │  .rvf bundle         (UDF reads these)       │
        │  table.rulake.json   (bundle manifest)       │
        │  iceberg/metadata/...(Tier-2 fallback)       │
        └─────────────────────────────────────────────┘
```

### Key design decisions in this picture, each with a named cost

1. **Warm state lives in Cloud Run memory, not in a shared Redis.**
   Simpler ops; costs us one cold-start per new container instance.
2. **Parquet and `.rvf` are the same data, written twice.** Doubles
   storage cost. We eat it for now to keep BQ-native column access.
3. **No row-filter pushdown.** BQ evaluates filters after the UDF
   returns top-k. For `WHERE genre = 'sci-fi'`, the UDF must either
   over-fetch (return 10× k) or accept filters as a UDF argument. We
   start with the former; see §"Filter pushdown" below.
4. **One UDF per region.** Cloud Run is regional; BQ region must match
   UDF region. Multi-region BQ datasets need one UDF per region.

### Filter pushdown

The UDF accepts an optional `filter: STRING` argument encoding an
`rvf-runtime::filter::FilterExpr`. The UDF applies it on RVF's
`MetadataStore` before the RaBitQ scan. This requires shipping the
filter metadata inside the `.rvf` bundle (META_SEG + META_IDX_SEG,
which RVF already encodes).

The trade-off: we build filter serialization and evaluation ourselves.
BQ column-level security and row-level policies are still applied
**after** the UDF returns, so a filter-aware UDF is a **performance**
optimisation, not a **security** boundary. Security remains in BQ.

---

## Things To Verify Before M2

Every one of these has moved in the last 12 months. Confirm at the
start of M1.

- [ ] BQ remote function request/response body cap (current doc says
      10 MB each; we previously saw 1 MB in an old doc).
- [ ] BQ remote function concurrency model (per-replica vs global).
- [ ] BQ remote function cold-start latency in the target region.
- [ ] BQ BigLake Iceberg maturity for schema evolution.
- [ ] BQ native `VECTOR(FLOAT32, D)` column type — is there one? (As
      of early 2026 BQ had `ARRAY<FLOAT64>` for embeddings but no
      fixed-size vector type.)
- [ ] BQ persistent Python UDF GA status and pricing.
- [ ] Dataplex lineage API stability (REST, not SDK).

Each unresolved item becomes an M1 task.

---

## Acceptance Test (Exit Criterion for M3)

```
Dataset: 1,000,000 vectors, D=128, clustered Gaussian (same generator as
BENCHMARK.md).

1. Build Parquet + .rvf bundle via `rvf-import` + `rvf-parquet` crates.
2. Upload to GCS (us-central1).
3. Deploy ruLake-udf to Cloud Run us-central1.
4. Register BigLake external table + remote function in BQ us-central1.
5. Run: SELECT ... FROM VECTOR_SEARCH(...) with 200 queries.
6. Compare top-10 IDs against Flat-exact top-10 computed locally.

Pass criteria:
  - recall@10 == 100%  (rerank×20 is sized for this)
  - p50 latency    <= 150 ms warm
  - p95 latency    <= 300 ms warm
  - p50 cold start <= 2 s (container start + Layer-A read)

Targets, unmeasured until M3 runs.
```
