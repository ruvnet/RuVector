# 01 — Architecture

ruLake is a four-layer adapter stack sitting on top of the existing RVF
workspace. This document walks each layer, names the **existing RVF
crates** that already cover it, identifies the **gap**, and specifies the
**interface contract** for the new work.

The tone here is: "what exists, what is missing, what the seam looks like."
No aspirational math.

---

## Layer L1 — Storage

### What it does

- Holds append-only `.rvf` files on an object store (S3, GCS, Azure Blob) or a local filesystem.
- Every segment is independently valid (RVF Four Laws, `docs/research/rvf/spec/00-overview.md`).
- The tail `MANIFEST_SEG` is the sole source of truth; readers scan backward from EOF.

### What RVF already covers

| Concern                         | Crate                                                    | Status |
|---------------------------------|----------------------------------------------------------|--------|
| Segment model, magic, headers   | `crates/rvf/rvf-types` (`segment.rs`, `segment_type.rs`) | Done   |
| Wire encode / decode            | `crates/rvf/rvf-wire` (`reader.rs`, `writer.rs`)         | Done   |
| Varint IDs, delta-coded offsets | `crates/rvf/rvf-wire/{varint,delta}.rs`                  | Done   |
| Tail-manifest scan              | `crates/rvf/rvf-wire/tail_scan.rs`                       | Done   |
| Two-level manifest + hotset     | `crates/rvf/rvf-manifest` (`level0.rs`, `level1.rs`)     | Done   |
| Progressive boot (4 KB read)    | `crates/rvf/rvf-manifest/boot.rs`                        | Done   |
| Local-FS store handle           | `crates/rvf/rvf-runtime::RvfStore`                       | Done   |

### Gap

- **Object-store backend.** `RvfStore` opens a local path. The cloud read
  path today is "`rvf-server` fronting a local file." For ruLake we need a
  first-class `S3Backend` / `GcsBackend` / `AzBackend` that implements
  HTTP range reads and obeys the progressive-boot contract (Layer-A read
  is ~4 KB; Layer-B read is ~MB; Layer-C is the full index).
- **Multi-file bundles.** An RVF "table" in ruLake is a directory of
  related `.rvf` files (vectors, index, quant, witnesses). RVF already
  supports multi-file via manifest chain refs; ruLake needs a **bundle
  manifest** (a tiny JSON or YAML sidecar) that tells a Parquet/Iceberg
  reader which `.rvf` files make up a logical table.

### Interface contract

```rust
pub trait ObjectStore {
    /// Read a byte range. Returns Err on 4xx / 5xx / IO.
    async fn range(&self, key: &str, start: u64, len: u64) -> Result<Bytes, Error>;

    /// Length in bytes. Needed for tail-scan (EOF-4 KB).
    async fn len(&self, key: &str) -> Result<u64, Error>;

    /// Optional: conditional read on etag, for cache correctness.
    async fn range_if(&self, key: &str, start: u64, len: u64, etag: Option<&str>) -> Result<ConditionalBytes, Error>;
}
```

Bundle manifest sidecar (`table.rulake.json`):

```json
{
  "format_version": 1,
  "table_name": "embeddings_v1",
  "dimension": 1536,
  "files": [
    { "role": "vec",    "uri": "gs://bkt/embeddings_v1/vec.rvf" },
    { "role": "index",  "uri": "gs://bkt/embeddings_v1/idx.rvf" },
    { "role": "quant",  "uri": "gs://bkt/embeddings_v1/quant.rvf" },
    { "role": "witness","uri": "gs://bkt/embeddings_v1/witness.rvf" }
  ]
}
```

**What we give up:** one-file portability. A ruLake bundle is no longer a
single `.rvf`; it is a directory. Customers who want the "single file" RVF
story can still use the monolithic mode — the sidecar is optional.

---

## Layer L2 — Index & Format

### What it does

- Packs vectors into columnar blocks that an external reader (Parquet,
  Arrow, Iceberg) can traverse without knowing anything about RVF.
- Stores HNSW adjacency + RaBitQ binary codes + optional PQ codes.
- Exposes a tail-manifest that answers: "where is the top HNSW layer?
  where are the cluster centroids? where is the binary codebook?"

### What RVF already covers

| Concern                                | Crate                                                           | Status |
|----------------------------------------|-----------------------------------------------------------------|--------|
| HNSW progressive indexing (A/B/C)      | `crates/rvf/rvf-index` (`layers.rs`, `progressive.rs`)          | Done   |
| Index segment codec                    | `crates/rvf/rvf-index/codec.rs` (IndexSegHeader, IndexSegData)  | Done   |
| Quantization: scalar, PQ, binary       | `crates/rvf/rvf-quant` (`scalar.rs`, `product.rs`, `binary.rs`) | Done   |
| Temperature tiering + count-min sketch | `crates/rvf/rvf-quant/{tier,sketch}.rs`                         | Done   |
| 1-bit RaBitQ with rotation + rerank    | `crates/ruvector-rabitq/{rotation,quantize,index}.rs`           | Done (standalone; merged `2c028aee3`) |

### Gap

- **Parquet bridge crate** (`rvf-parquet`, new). Converts an RVF
  `VEC_SEG` + `QUANT_SEG` + `INDEX_SEG` into an Arrow RecordBatch with
  schema:
  - `id: UInt64`
  - `vec: FixedSizeList<Float32, D>`
  - `quant: FixedSizeBinary(D/8)` for RaBitQ 1-bit
  - `meta: Struct<...>` (optional, from META_SEG)
- **Iceberg v2 manifest entry**. Iceberg already supports opaque binary
  columns. The manifest entry just needs a `data_file.file_format =
  "PARQUET"` pointing at the Parquet bridge output, plus a custom
  `table_property` pointing at the `.rvf` sidecar for the index.
- **RaBitQ segment type.** `crates/rvf/rvf-types::SegmentType` has 29
  variants (0x01..0x36). RaBitQ codes today live in `ruvector-rabitq`
  outside the RVF segment taxonomy. ruLake needs a new segment type
  discriminator (proposal: `0x0D RaBitQCodes` — but 0x0D is already
  `MetaIdx`; propose `0x24 RabitqCodes` in the 0x20+ block, following
  the same reservation pattern as federation used at 0x33–0x36).
- **Vector column metadata propagation.** Dimensionality, distance
  metric, quantization tier — today these live in `rvf-types` structs.
  They need a canonical Parquet/Iceberg key-value metadata mapping.

### Interface contract

```rust
// New crate: rvf-parquet
pub struct ParquetBridge {
    pub rvf: RvfStore,
    pub schema: Arc<ArrowSchema>,
}

impl ParquetBridge {
    /// Emit a single Arrow RecordBatch covering the given ID range.
    pub fn record_batch(&self, id_range: Range<u64>) -> Result<RecordBatch, Error>;

    /// Stream Parquet row groups for a full table.
    pub fn write_parquet<W: Write>(&self, w: W, opts: ParquetOpts) -> Result<WriteStats, Error>;
}
```

**What we give up:** vector columns written this way are **opaque to
Parquet-native scan pushdown**. A query `WHERE distance < 0.5` cannot be
predicate-pushed by Parquet; it has to be pushed through the UDF
(Layer L3). This is an explicit trade-off — we keep Parquet compatibility
at the cost of in-reader filter pushdown.

---

## Layer L3 — Query & Kernel

### What it does

- Exposes the RaBitQ distance + HNSW traversal kernel as a **user-defined
  function (UDF)** the host SQL engine calls.
- For each target engine, wraps the kernel in that engine's UDF contract
  (BQ remote function, Snowflake external function, Trino plugin, DuckDB
  loadable extension, Databricks Photon UDF, Spark UDF).
- Handles top-k merge, either inside the UDF (low k) or via a
  fan-out/merge pattern in host SQL (high k).

### What RVF already covers

| Concern                            | Crate                                                             | Status |
|------------------------------------|-------------------------------------------------------------------|--------|
| Distance kernels (L2, cosine, dot) | `crates/rvf/rvf-index/distance.rs`                                | Done   |
| RaBitQ symmetric + asymmetric      | `crates/ruvector-rabitq/{quantize,index}.rs`                      | Done   |
| HNSW graph traversal               | `crates/rvf/rvf-index/hnsw.rs`                                    | Done   |
| Progressive query (Layer A→C)      | `crates/rvf/rvf-index/progressive.rs`                             | Done   |
| HTTP query endpoint                | `crates/rvf/rvf-server/http.rs` (`POST /v1/query`)                | Done   |
| TCP streaming protocol             | `crates/rvf/rvf-server/tcp.rs`                                    | Done   |
| 5.5 KB WASM fallback runtime       | `crates/rvf/rvf-solver-wasm`, `rvf-wasm`                          | Done   |
| `no_std` types for embedded/BPF    | `crates/rvf/rvf-types` (feature-gated `std`/`alloc`)              | Done   |

### Gap

- **UDF wrappers** — one per target engine. Each has a different ABI:
  - **BigQuery remote function:** HTTP POST with a JSON batch of rows; we
    deploy to Cloud Run; returns a JSON batch. Payload capped at 1 MB
    (verify before relying).
  - **Snowflake external function:** HTTPS POST to an AWS API Gateway or
    Azure Function; JSON in, JSON out; 5 MB response cap (verify).
  - **Trino connector:** a JVM SPI plugin. We avoid JVM by shipping a
    Rust-side process Trino talks to via gRPC (protocol: verify).
  - **DuckDB extension:** a loadable `.duckdb_extension` file. DuckDB's
    C ABI is stable; we ship a Rust shim using the `duckdb-rs` crate.
  - **Databricks Photon UDF:** same as Spark UDF — a pickled Python
    wrapper calling our native lib via cffi/JNI. v1 Tier-2.
- **Top-k merge strategy.** For low k (k ≤ 100), the UDF can return
  per-shard top-k and the host SQL does `UNION ALL + ORDER BY + LIMIT`.
  For high k, we need a scan orchestrator that streams candidates.
- **Warm-state caching.** The RaBitQ rotation matrix is expensive to
  materialise (Gram–Schmidt is O(D³); for D=1536 that is 3.6 B ops, see
  ADR-154). It must be cached across UDF invocations. BQ remote
  functions give us a stateful Cloud Run instance; Snowflake external
  functions may cold-start per call — verify before M2.

### Interface contract

Every UDF wrapper implements the same inner trait:

```rust
pub trait VectorUdf {
    /// A single scan request: query vector, k, optional filter.
    fn scan(&self, req: ScanRequest) -> Result<ScanResponse, Error>;
}

pub struct ScanRequest {
    pub query: Vec<f32>,             // D-dim
    pub k: u32,
    pub rerank_factor: u32,          // 0 = no rerank, 20 = default for RaBitQ+
    pub filter: Option<FilterExpr>,  // re-use rvf-runtime::filter::FilterExpr
}

pub struct ScanResponse {
    pub ids: Vec<u64>,
    pub distances: Vec<f32>,
    pub scanned: u64,                // candidates examined, for cost reporting
}
```

The HTTP JSON shape is a thin wrapper around this struct — the same body
the existing `rvf-server`'s `POST /v1/query` handler accepts, with one
added field (`rerank_factor`) and one removed (`options`, which is a
BQ-level concern).

**What we give up:** every UDF is a per-engine integration. There is no
single "ruLake UDF." We commit to BigQuery + DuckDB in v1 and sketch the
rest.

---

## Layer L4 — Governance & Compliance

### What it does

- Maps RVF's cryptographic provenance (witness chain, CRYPTO_SEG, MANIFEST
  lineage) into the **catalog** the host warehouse uses for RBAC, audit,
  lineage visualisation, GDPR deletion, and region pinning.
- For each host, speaks to: Dataplex (GCP), Unity Catalog (Databricks),
  Polaris / Nessie (Snowflake / Iceberg), AWS Glue (for AWS-adjacent).

### What RVF already covers

| Concern                               | Crate / file                                                             | Status |
|---------------------------------------|--------------------------------------------------------------------------|--------|
| Witness chain (SHAKE-256, Ed25519)    | `crates/rvf/rvf-crypto`, `rvf-types/witness.rs`                          | Done   |
| PQ signatures (ML-DSA-65, SLH-DSA)    | `crates/rvf/rvf-crypto`                                                  | Done   |
| FileIdentity + lineage chain          | `crates/rvf/rvf-types/{lineage,manifest}.rs`                             | Done   |
| COW branching for snapshot safety     | `crates/rvf/rvf-types/cow_map.rs`, `rvf-runtime/cow.rs`                  | Done   |
| PII stripping (12 detectors)          | `crates/rvf/rvf-federation/pii_strip.rs`                                 | Done   |
| Differential privacy (RDP accounting) | `crates/rvf/rvf-federation/diff_privacy.rs`                              | Done   |
| Redaction log segment (0x35)          | `crates/rvf/rvf-federation/types.rs`                                     | Done   |
| TEE attestation records               | `rvf-types` (`CRYPTO_SEG` + TEE quotes)                                  | Done   |
| Deletion bitmap + JOURNAL_SEG         | `crates/rvf/rvf-runtime/{deletion,write_path}.rs`                        | Done   |

### Gap

- **Catalog adapter crates.** One each for Dataplex / Unity / Polaris.
  They translate RVF witness + lineage into the catalog's native
  data-contract:
  - Lineage: emit a "produced by" edge tying a BQ query ID to the
    witness hash of the vectors it read.
  - RBAC: consume the catalog's ACLs and translate them into membership
    filters (`MembershipFilter`, `crates/rvf/rvf-runtime/membership.rs`)
    at the RVF level. This is **per-user** filtering; column-level RBAC
    is the host's job.
  - Region pin: encode the GCS / S3 region in the bundle manifest and
    reject reads whose caller is in a mismatched region.
- **GDPR hard-delete orchestrator.** Two phases:
  1. Logical delete via JOURNAL_SEG + tombstone + deletion bitmap.
     Queries stop returning the vector immediately.
  2. Cryptographic delete via COW compaction: rebuild the vector
     segment without the tombstoned rows, rewrite the witness chain
     with a redaction-log entry (0x35), and retire the old file.
  Until phase 2 completes, the witness chain still references the
  deleted ID by hash — this is a named trade-off and is called out in
  `04-governance-and-compliance.md`.
- **Audit sink.** A small adapter that tails the RVF witness chain and
  forwards hash-linked entries to the host's audit log (Cloud Logging,
  Azure Monitor, Splunk, Datadog).

### Interface contract

```rust
pub trait Catalog {
    /// Emit a lineage edge: "query Q read table T (bundle hash H)".
    async fn emit_lineage(&self, edge: LineageEdge) -> Result<(), Error>;

    /// Get the ACL for a given user on a given table.
    async fn get_acl(&self, user: UserId, table: TableId) -> Result<Acl, Error>;

    /// Request hard delete of the rows matching a filter.
    async fn request_gdpr_delete(&self, table: TableId, filter: FilterExpr) -> Result<DeleteJob, Error>;
}
```

**What we give up:** we do **not** implement column-level masking inside
ruLake. That lives in the host SQL engine (BQ column-level security,
Snowflake masking policies, Unity Catalog column ACLs). ruLake carries
the PII-stripping provenance (via `RedactionLog`) so the host can audit
that masking actually happened; it does not enforce masking itself.

---

## Putting It Together: Query Path

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 1. Analyst runs:                                                     │
   │      SELECT id FROM `project.ds.emb` ,                               │
   │        VECTOR_SEARCH(...) USING RULAKE('[0.1, 0.2, ...]', k=>10)    │
   ├──────────────────────────────────────────────────────────────────────┤
   │ 2. BigQuery planner resolves the UDF, packages rows into a batch.   │
   ├──────────────────────────────────────────────────────────────────────┤
   │ 3. Remote function POST hits Cloud Run (ruLake UDF service).        │
   │    - Warm state: rotation matrix, top-layer HNSW adjacency.         │
   │    - Cold state (first call of the day): pull 4 KB Level-0 root     │
   │      from GCS, then ~4 MB hotset for Layer A.                       │
   ├──────────────────────────────────────────────────────────────────────┤
   │ 4. UDF runs RaBitQ+ sym scan, rerank×20, returns {id, dist}[].      │
   ├──────────────────────────────────────────────────────────────────────┤
   │ 5. BigQuery merges UDF output into the query; row-level RBAC and    │
   │    column masking applied by BQ before the user sees results.       │
   ├──────────────────────────────────────────────────────────────────────┤
   │ 6. Dataplex lineage adapter emits an edge:                          │
   │      job_id → bundle_hash → witness_chain_head                       │
   └──────────────────────────────────────────────────────────────────────┘
```

No new storage system. No new planner. No new UI. The only ruLake
artefacts are the UDF service, the bundle manifest, and the catalog
adapter.

---

## What Does NOT Fit in This Architecture

- **Write amplification from in-place mutations.** RVF is append-only.
  High-churn OLTP workloads (tens of millions of per-row updates per day)
  will accumulate JOURNAL and COW segments faster than compaction
  retires them. ruLake inherits this. Mitigation: batch upserts and
  compact daily. Not suitable for real-time feature-store-style churn
  without more work.
- **Cross-table JOINs on vector columns.** If you want `SELECT a.id,
  b.id FROM a, b WHERE VECTOR_DIST(a.v, b.v) < 0.3`, the quadratic
  expansion happens in BQ before the UDF sees anything. Our UDF is
  point-query-shaped. JOIN-style vector similarity is a v2 concern.
- **Arbitrary distance metrics.** `rvf-index/distance.rs` has L2, cosine,
  dot. RaBitQ+ is optimised for angular distance. Hamming, Manhattan,
  Jaccard, and custom metrics are out of scope for v1.
- **Vector columns inside row-oriented OLTP databases.** ruLake is a
  lake layer, not an OLTP index. Postgres / MySQL / DynamoDB users
  should keep pgvector / native extensions.
