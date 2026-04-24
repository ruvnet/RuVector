# 00 — Master Plan

## Goal

Ship a vector-native datalake layer (`ruLake`) that lets BigQuery (Tier 1)
and Snowflake / Databricks / Trino / DuckDB / Iceberg (Tier 2) read RVF
vector data and run RaBitQ-accelerated similarity queries **without
standing up a parallel vector database**. Deliver the minimum viable shape
in 12 weeks. Validate the Tier-1 primary integration against a real
BigQuery project before M5.

## Non-Goals (v1)

- Distributed query planner. Tier-1 SQL engines do their own planning;
  ruLake provides the kernel they call.
- Writes through BigQuery / Trino / Snowflake. v1 is **read-optimised**;
  writes go through `rvf-import` or `rvf-server` ingest paths.
- A ruLake-branded UI. The Causal Atlas dashboard in `rvf-server` is the
  only UI surface this spike relies on.
- Multi-region active-active replication. v1 pins a `.rvf` bundle to one
  region and documents cross-region read latency honestly.
- Dense JOINs between vector columns and OLAP fact tables inside ruLake.
  That lives in the host SQL engine.
- Online learning / SONA federation integration. Already lives in
  `rvf-federation`; v1 documents the seam but does not wire it.

---

## Goal Tree

```
G0  ruLake: vector-native datalake layer
 ├── G1  BigQuery read path works end-to-end        (Tier 1, M3 exit)
 │    ├── G1.1  RVF → Parquet bridge for vector columns
 │    ├── G1.2  BigQuery remote function calling RaBitQ kernel
 │    └── G1.3  External table manifest with HNSW entry pointers
 ├── G2  Governance story is credible for enterprise (M4 exit)
 │    ├── G2.1  Lineage surfaced via Dataplex / Unity / Polaris
 │    ├── G2.2  GDPR delete path: logical + cryptographic
 │    ├── G2.3  RBAC mapping (row-level, column-level) to host catalog
 │    └── G2.4  Audit log: witness chain → host audit sink
 ├── G3  Tier-2 adapters: at least Iceberg + DuckDB             (M4)
 │    ├── G3.1  Iceberg v2 manifest entry for an RVF blob
 │    └── G3.2  DuckDB extension loading RVF + RaBitQ kernel
 ├── G4  Performance budget hits enterprise bar                 (M5)
 │    ├── G4.1  <50 ms p50 in-region cold-cache query (target)
 │    ├── G4.2  100% recall@10 with rerank×20 on 1 M scale (target)
 │    └── G4.3  Ingest: 250 k vectors/s from Parquet (target)
 └── G5  Honest positioning + operator docs                     (M5)
      ├── G5.1  "ruLake is NOT …" rubric shipped with the release
      └── G5.2  One runnable BQ + ruLake quickstart in `examples/`
```

---

## Milestones

| ID | Name                              | Exit criterion                                                                            | Week |
|----|-----------------------------------|-------------------------------------------------------------------------------------------|------|
| M1 | Format bridge spec                | Parquet + Iceberg wire shape for an RVF vector column is frozen; round-trips in tests    | 3    |
| M2 | Kernel-as-UDF prototype           | RaBitQ angular scan runs inside a BQ remote function against a fixture `.rvf` in GCS     | 6    |
| M3 | BigQuery end-to-end               | `SELECT VECTOR_SEARCH(...)` over an external table returns 100% recall@10 on 1 M vectors | 8    |
| M4 | Governance + Tier-2 adapters      | Dataplex lineage + Iceberg catalog entry + DuckDB extension demo all green               | 10   |
| M5 | Public spike + acceptance demo    | One-click quickstart; performance budget doc updated with measured numbers; ADR accepted | 12   |

---

## Dependency DAG

```
                        ┌──────────────────────────┐
                        │ M1  Format bridge spec   │
                        └──────┬───────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
 │ M2 Kernel-as-UDF │ │ G3.1 Iceberg v2  │ │ G3.2 DuckDB ext  │
 └──────┬───────────┘ └────────┬─────────┘ └────────┬─────────┘
        │                      │                    │
        ▼                      └───────┬────────────┘
 ┌──────────────────┐                  │
 │ M3 BQ end-to-end │                  │
 └──────┬───────────┘                  ▼
        │                    ┌────────────────────────────┐
        └───────────────────▶│ M4 Governance + Tier-2     │
                             └─────────────┬──────────────┘
                                           ▼
                              ┌────────────────────────────┐
                              │ M5 Public spike + demo     │
                              └────────────────────────────┘
```

M1 blocks everything. M2 and M3 are the Tier-1 critical path. Tier-2
adapters fan out from M1 and converge into M4. M4 blocks M5.

---

## 12-Week Timeline (Engineer-Weeks)

| Week | Milestone | Work                                                              | E-wks |
|------|-----------|-------------------------------------------------------------------|------:|
| 1    | M1        | Inventory RVF segments vs Parquet logical types; seed spec doc    | 1.0   |
| 2    | M1        | Iceberg v2 manifest mapping; metadata file shape                  | 1.5   |
| 3    | M1        | Round-trip tests (RVF → Parquet → RVF); freeze spec               | 1.5   |
| 4    | M2        | BQ remote function harness; GCS range read of `.rvf`              | 2.0   |
| 5    | M2        | RaBitQ kernel packaged as a cross-platform shared lib (Linux x86) | 2.0   |
| 6    | M2        | BQ UDF wrapper returns top-k IDs + distances; fixture validation  | 1.5   |
| 7    | M3        | External table manifest wiring; HNSW entry pointer extraction     | 2.0   |
| 8    | M3        | 1 M-vector acceptance run; recall@10 verification vs ground truth | 2.0   |
| 9    | M4        | Dataplex lineage adapter (Python-free: use REST from Rust)        | 2.0   |
| 10   | M4        | Unity Catalog / Polaris sketch; DuckDB extension crate            | 2.5   |
| 11   | M5        | Perf budget measured and written up; failure-mode docs            | 1.5   |
| 12   | M5        | Quickstart example; ADR flip to "Accepted"; human review gate     | 1.0   |

**Total: 20.5 engineer-weeks.** Assumes ~1.7 FTE average over 12 weeks.
Slack of ~4 E-wks is intentional — GCP quota, BQ remote-function IAM
quirks, and Iceberg spec ambiguities historically absorb that much.

---

## Risk Register

Each risk has: trigger, blast radius, mitigation, owner TBD.

| # | Risk                                                                 | Trigger                                                       | Blast radius            | Mitigation |
|---|----------------------------------------------------------------------|---------------------------------------------------------------|-------------------------|------------|
| R1| **BigQuery remote functions have a 1 MB payload cap** (verify before M2) | Query touches > ~8k vectors at D=128 in one request           | M2 slips; redesign UDF  | Chunked scan driven by BQ; push top-k merge into the client-side SQL. Verify cap in week 1. |
| R2| **Iceberg v2 manifest does not natively model "opaque sidecar blob"** | Iceberg catalog rejects our vector-column layout              | Tier-2 fallback only    | Treat vector column as a `binary` column referencing an external blob; rely on ADR-documented convention, not spec extension. |
| R3| **RaBitQ recall on real SIFT1M differs from our clustered-Gaussian numbers** | Measured recall on SIFT1M < 90% at rerank×20                  | Budget doc must soften  | Run SIFT1M and GIST1M in M2; publish numbers in `05-performance-budget.md` before M3 locks. `BENCHMARK.md §What's NOT benchmarked` already flags this. |
| R4| **GCS range-read latency tail dominates cold queries**               | p95 first-query > 500 ms on cold cache                        | Enterprise adoption bar | Use Layer-A hotset pointers (already in RVF) for 4 KB boot read; document warm-cache SLO separately. |
| R5| **Dataplex lineage API churns**                                      | Google changes lineage API between week 1 and week 10          | M4 slips by 1 week      | Use `gcloud data-catalog` REST, not a Python SDK wrapper; pin API version; write a contract test. |
| R6| **Cross-platform shared lib packaging (musl / glibc / macOS / Windows)** | BQ sandbox only accepts Linux glibc x86_64 + one ABI version  | M2 slips                | Ship glibc x86_64 only for v1 Tier-1. Document macOS/arm64 as "Tier-2 via DuckDB". |
| R7| **Post-quantum signature verification cost**                         | ML-DSA-65 verify adds > 50 ms per cold query                  | Perf budget miss        | Verify once at bundle open, not per query; cache manifest in the UDF warm state. Already the pattern in `rvf-manifest::boot`. |
| R8| **GDPR "hard delete" vs witness chain immutability**                 | Legal needs cryptographic deletion of a vector                | Governance story breaks | Two-phase delete: logical deletion via JOURNAL_SEG + tombstone, followed by compaction that re-seals the chain. Document the SLA and the window during which the witness still references the redacted ID. |
| R9| **Vendor lock on BQ remote function pricing**                        | Remote function execution dominates cost vs in-BQ scan         | TCO story weakens       | Provide a parallel DuckDB-embedded path; let customers pick. |
| R10| **Parquet spec does not allow nested fixed-size f32 arrays cleanly** | Type system rejects `ARRAY<FLOAT32>[D]` in some engines       | M1 wobble               | Use Arrow `FixedSizeList<Float32, D>`; verify BQ + Snowflake + DuckDB + Trino all accept it before M1 exit. |

---

## Go / No-Go Gates

- **End of M1:** Spec round-trips in tests OR stop and re-scope to "RVF
  side-by-side with Parquet, no bridge". No silent slip.
- **End of M3:** BQ end-to-end on 1 M vectors at 100% recall@10 OR drop
  BigQuery as Tier-1 and promote DuckDB. A non-working BQ integration is
  worse than no integration at this price point.
- **End of M4:** Dataplex lineage visible in the BQ console for a query
  that went through the ruLake UDF OR ship without the lineage story and
  mark it "deferred to v2" in `06-positioning.md`.
- **End of M5:** ADR flips from "Proposed" to "Accepted" OR document the
  open questions in the ADR and close the spike as "not ready".

---

## Deferred to v2

- Writes through SQL engines (`INSERT INTO ruvec_table SELECT …`)
- Active-active multi-region
- GPU kernels (CUDA / ROCm) for RaBitQ
- Delta / Hudi table format support (Iceberg is the v1 lingua franca)
- SONA federated learning integration
- Snowflake Cortex dense + ruLake hybrid search
- Azure Synapse / Fabric integration
