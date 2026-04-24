# 02 — Datalake Comparison

A candid look at the six most common targets an enterprise "vector
datalake layer" has to plug into as of **2026-04**. For each: what is
their native vector story, what format they ingest, what their
governance looks like, what RVF needs to expose to plug in, and — named
up front — **where RVF does not fit today**.

Every table cell that could be out of date carries "**verify before
relying**" if the underlying vendor feature has changed in the last
12 months.

---

## The Six Targets

| Target                  | Why it matters                                                   |
|-------------------------|------------------------------------------------------------------|
| **BigQuery**            | Tier 1. Largest enterprise warehouse. Native vector search (GA 2024). |
| **Snowflake**           | Tier 2. Second-largest. Cortex vector functions 2024–2025.       |
| **Databricks / Delta**  | Tier 2. Mosaic AI Vector Search built on Delta Lake.             |
| **Apache Iceberg**      | Tier 2. Open table format; lingua franca for open lakes.         |
| **Trino / Presto**      | Tier 2. Query federation layer; relevant for multi-lake.         |
| **DuckDB**              | Tier 2. Embedded OLAP; ideal for the laptop / notebook path.     |

Azure Synapse / Fabric, ClickHouse, Dremio, AWS Redshift, Apache Hudi,
and Delta-Airbyte are **out of scope for v1** and tracked as v2
candidates.

---

## BigQuery

| Dimension                 | Where it stands (2026-04)                                                                                             |
|---------------------------|------------------------------------------------------------------------------------------------------------------------|
| Native vector support     | BigQuery Vector Search GA Sept 2024. `VECTOR_SEARCH()` function + `CREATE VECTOR INDEX` DDL. IVF index internally (verify current). |
| Ingest format             | Native BQ table, BQ external tables over Parquet / ORC / Avro / Iceberg.                                               |
| Index format              | Opaque BQ-managed. Not portable. No reading the index outside BQ.                                                      |
| Governance                | Dataplex (catalog + lineage). IAM. Column-level + row-level security policies.                                         |
| Region pinning            | Per-dataset. Multi-region is an explicit dataset option.                                                               |
| UDF surface               | Remote functions (HTTP to Cloud Run), JavaScript UDFs (limited), persistent Python UDFs (preview as of 2024; verify).  |
| GDPR deletion             | `DELETE` DML on base tables. Vector index rebuilds on next DDL. No cryptographic deletion.                             |
| What RVF has to expose    | (a) Parquet bridge for ingest; (b) Remote function UDF calling RaBitQ; (c) External table manifest.                    |
| Where RVF does not fit    | BQ's native `VECTOR_SEARCH` is already fast and integrated; the case for ruLake is not speed, it is **portability** (same `.rvf` in BQ, DuckDB, edge) and **provenance** (witness chain tied to `job_id`). If a customer only ever uses BQ, the native path wins. |

**ruLake answer:** Tier 1. See `03-bigquery-integration.md` for three
candidate architectures and the pick.

---

## Snowflake

| Dimension                 | Where it stands (2026-04)                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| Native vector support     | Cortex vector functions (`VECTOR_COSINE_DISTANCE`, `VECTOR_L2_DISTANCE`) on `VECTOR(FLOAT, D)` column type, 2024. Native ANN index preview (verify current GA status). |
| Ingest format             | Native Snowflake table, external tables over Parquet / CSV / JSON / Iceberg (Polaris catalog).                      |
| Index format              | Opaque Snowflake-managed.                                                                                           |
| Governance                | Horizon Catalog / Polaris. Column masking, row access policies, object tagging, classification.                      |
| Region pinning            | Account-level + DB-level.                                                                                           |
| UDF surface               | External functions (AWS API Gateway / Azure Function), Python UDFs (Snowpark), Java UDFs, native SQL UDFs.          |
| GDPR deletion             | `DELETE` DML. Time Travel retains deleted data up to 90 days — customers must plan around this for right-to-erasure. |
| What RVF has to expose    | External function UDF (HTTP shim to AWS API GW / Azure Fn / GCP Cloud Run), Polaris Iceberg catalog entry, Parquet bridge. |
| Where RVF does not fit    | Snowflake external functions require a VPC peering / API Gateway front door per account. Ops overhead is non-trivial. For pure-Snowflake shops with Cortex adequate, ruLake is a hard sell. Hybrid shops (Snowflake + DuckDB / edge) are the win condition. |

**ruLake answer:** Tier 2. Via Polaris Iceberg catalog + external
function. Documented in v1, implemented in v2 unless a customer asks.

---

## Databricks / Delta Lake

| Dimension                 | Where it stands (2026-04)                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| Native vector support     | Mosaic AI Vector Search (Databricks Vector Search) over Delta tables; HNSW + filtered search. GA 2024.               |
| Ingest format             | Delta Lake (Parquet + transaction log). Direct from Spark / Structured Streaming / Auto Loader.                      |
| Index format              | Managed service ("Vector Search endpoints"). Not portable.                                                            |
| Governance                | Unity Catalog (central governance). Column-level RBAC, lineage, tags, classifications, row-level filters.            |
| Region pinning            | Workspace-level.                                                                                                     |
| UDF surface               | Spark UDFs (Python, Scala, Rust via Arrow), Photon UDF path for vectorised execution (native-code UDFs preview;     verify). |
| GDPR deletion             | `DELETE` DML on Delta tables. VACUUM removes old files after retention. Supports cryptographic deletion via key-based encryption + key deletion (verify). |
| What RVF has to expose    | (a) Delta / Iceberg-compatible sidecar (Delta 3.x supports reading Iceberg); (b) native-code UDF for Photon; (c) Unity Catalog lineage adapter. |
| Where RVF does not fit    | Mosaic AI Vector Search is tightly coupled to Databricks compute. Replacing it with ruLake inside Databricks is fighting uphill. The win condition is a customer who wants **the same bundle** readable in Databricks AND somewhere else (BQ, edge, local DuckDB). |

**ruLake answer:** Tier 2. Prioritise via Iceberg compatibility, not
Delta directly (Delta 3.x reads Iceberg; we avoid a native Delta writer
in v1).

---

## Apache Iceberg

| Dimension                 | Where it stands (2026-04)                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| Native vector support     | None in the spec as of v2 / v3 draft. Iceberg is a table format, not an indexing format.                             |
| Ingest format             | Parquet / ORC / Avro, with manifest + snapshot metadata in its own format.                                           |
| Index format              | v3 draft introduces secondary indexes (verify current status; v3 is in draft). For v1 we treat Iceberg as indexless. |
| Governance                | Catalog-agnostic. Polaris (Snowflake), Nessie (open-source), AWS Glue, Dataplex, Unity Catalog all implement the Iceberg REST catalog API. |
| Region pinning            | Object-store level.                                                                                                  |
| UDF surface               | None. Iceberg does not execute queries — that is the compute engine's job (Spark, Trino, BQ, Snowflake, DuckDB, etc.). |
| What RVF has to expose    | An Iceberg v2 manifest entry + a convention: vector column is `BINARY` pointing at an RVF blob via `table_properties`. |
| Where RVF does not fit    | Iceberg does not index. Every compute engine that reads our Iceberg table still needs its own UDF wrapper. Iceberg is a distribution mechanism, not a query path. |

**ruLake answer:** **Iceberg is our lingua franca for Tier 2 lake
distribution.** We publish Iceberg v2 manifests; every engine that
speaks Iceberg (Trino, Spark, Snowflake via Polaris, BQ via BigLake)
can read the table. Query acceleration still needs a per-engine UDF.

---

## Trino / Presto

| Dimension                 | Where it stands (2026-04)                                                                                             |
|---------------------------|------------------------------------------------------------------------------------------------------------------------|
| Native vector support     | Trino added `vector` SQL type support + basic distance functions in 2024 (verify current scope). No native ANN index. |
| Ingest format             | Reads via connectors: Hive, Iceberg, Delta, Postgres, many others.                                                     |
| Index format              | Index lives in whichever underlying store the connector reads. Trino does not manage indexes.                          |
| Governance                | Via underlying catalog (Hive Metastore, Glue, Polaris, Unity).                                                          |
| Region pinning            | Per-coordinator / per-worker placement.                                                                                |
| UDF surface               | JVM plugin SPI (Java). Rust UDFs possible via sidecar process + gRPC (pattern used by several community plugins).     |
| GDPR deletion             | Inherits from the underlying store. Trino is query-only.                                                              |
| What RVF has to expose    | Trino connector plugin (JVM or Rust sidecar) that reads our Iceberg manifests and calls RaBitQ via gRPC.              |
| Where RVF does not fit    | The JVM ecosystem. We are 100% Rust. Shipping a Trino plugin means either a JVM wrapper or a sidecar subprocess — both are ops overhead. |

**ruLake answer:** Tier 2, deferred to v2. Trino users get ruLake via
the Iceberg path + a client-side distance filter, without the
kernel-in-UDF acceleration. We document the sidecar-plugin pattern
but do not implement it in v1.

---

## DuckDB

| Dimension                 | Where it stands (2026-04)                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| Native vector support     | `ARRAY<FLOAT>[D]` type; distance functions; HNSW extension (`vss`) community-maintained; production-grade for most workloads. |
| Ingest format             | Parquet, CSV, JSON, Arrow, SQLite, PostgreSQL wire, Iceberg (as of 2024), Delta (as of 2024; verify maturity).       |
| Index format              | Native `vss` HNSW, on-disk. Not portable outside DuckDB.                                                             |
| Governance                | DuckDB is embedded. Governance lives in the host app.                                                                |
| Region pinning            | Host app responsibility.                                                                                             |
| UDF surface               | Native C/C++ extensions (stable C ABI). Rust extensions via `duckdb-rs`. Python UDFs via PyDuckDB.                   |
| GDPR deletion             | `DELETE` DML; VACUUM. No cryptographic deletion in the core.                                                         |
| What RVF has to expose    | A DuckDB extension (`duckdb-rulake`) that registers: (a) a table function reading an RVF bundle as rows, (b) a scalar function for RaBitQ distance, (c) a table function for top-k ANN. |
| Where RVF does not fit    | On laptops / notebooks with < 1 M vectors, DuckDB `vss` is already faster than the network round-trip our UDF pattern adds. ruLake's value on DuckDB is **format portability** (same `.rvf` in the lake, on the laptop, at the edge), not raw perf. |

**ruLake answer:** Tier 2, **implemented in v1** because DuckDB is the
cheapest way to prove the portability story. See `07-implementation-plan.md`
for the week-10 slot.

---

## At-a-Glance Matrix

Columns: **Native vector ?** · **Reads Iceberg ?** · **UDF shape** ·
**Governance hook** · **v1 in scope ?** · **Where RVF does NOT fit**

| Target        | Native vec?      | Iceberg? | UDF shape                | Governance        | v1 ? | Where RVF doesn't fit |
|---------------|------------------|----------|--------------------------|-------------------|------|-----------------------|
| BigQuery      | Yes (Vec Search) | Via BigLake | Remote function (HTTP) | Dataplex          | Yes  | Customers only using BQ native path |
| Snowflake     | Yes (Cortex)     | Via Polaris | External function (HTTP)| Horizon / Polaris | No   | VPC peering ops overhead |
| Databricks    | Yes (Mosaic)     | Yes (3.x) | Spark / Photon UDF      | Unity Catalog     | No   | Inside-workspace competition |
| Iceberg       | No               | (native) | N/A (format only)        | Any Iceberg catalog | Yes | No query path on its own |
| Trino         | Partial          | Yes      | JVM plugin / sidecar    | Via connector     | No   | JVM ecosystem; sidecar ops cost |
| DuckDB        | Yes (`vss`)      | Yes      | Native C/Rust ext       | Host app          | Yes  | Local perf already good |

---

## Common Abstraction: The Bundle-Plus-UDF Pattern

Every target shares the same shape:

1. **Distribution**: Iceberg v2 manifest points at a Parquet file +
   an `.rvf` sidecar directory.
2. **Acceleration**: a UDF in the host's native shape reads the
   `.rvf` sidecar and runs RaBitQ + HNSW.
3. **Governance**: the host catalog (Dataplex / Polaris / Unity /
   Glue) consumes witness-chain edges from a ruLake lineage
   emitter.

The only thing that differs per target is (2) — the UDF ABI. That is
where most of the v1 engineering budget goes.

---

## Where RVF Does Not Fit Today (Consolidated)

A brief, honest consolidation of the "does not fit" columns above:

- **Inside a vendor's managed vector service** (BQ Vector Search,
  Cortex, Mosaic AI). If the customer is happy there, ruLake does not
  displace; it adds friction.
- **JVM-native stacks** (Spark, Trino, Flink). We will always be a
  second-class citizen without a JVM plugin team.
- **High-throughput OLTP vector write** workloads (>10k upserts/s per
  shard). RVF's append-only model + compaction is OLAP-shaped.
- **Arbitrary distance metrics.** RaBitQ+ is angular/cosine-optimised.
  Hamming / Manhattan / weighted-sum workloads are v2.
- **Sub-millisecond single-vector lookups.** RaBitQ+ at rerank×20 on
  1 M vectors is ~1 ms on a laptop with the whole index in RAM (see
  BENCHMARK.md). Over a BQ remote function, cold-path is ≥100 ms round
  trip. For sub-ms you want an embedded index in-process (DuckDB
  extension, `rvf-server` sidecar, WASM tile). That is supported, but
  it is not the BQ story.
