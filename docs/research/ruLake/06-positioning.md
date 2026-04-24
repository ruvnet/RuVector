# 06 — Positioning

The positioning rule for this spike: every time we are tempted to
write "ruLake is …," we write "ruLake is NOT …" first. Most of the
ways this kind of project fails are in the gap between what the
engineering team thinks they are shipping and what the GTM team sells.

This document is the hype-avoidance rubric for the spike. It ships
alongside the v1 release.

---

## What ruLake IS

A read-optimised, vector-native **format + catalog + kernel adapter
layer** that sits between object storage and the SQL engine the
enterprise already operates. It ships as:

- A Parquet / Iceberg extension for a vector column.
- A per-engine UDF (BigQuery remote function in v1; DuckDB extension
  in v1; others deferred).
- A catalog adapter emitting lineage into Dataplex / Unity / Polaris.
- The existing 22-crate RVF workspace, unchanged.

That is the whole product.

---

## What ruLake IS NOT

### Not a vector database

ruLake is not a competitor to Pinecone, Weaviate, Milvus, Qdrant, or
LanceDB. Those products are **systems of record** — they run their
own cluster, manage their own storage, expose their own API. ruLake
runs **no cluster of its own** and exposes **no API of its own**
beyond the UDF that the host warehouse calls.

If a prospect asks "does ruLake replace Pinecone?", the answer is
"only if you were using Pinecone because you wanted a vector column
inside your datalake — in which case, yes. If you were using Pinecone
because you wanted a managed serving tier with sub-ms p99, ruLake
does not displace it."

### Not a replacement for BigQuery / Snowflake / Databricks

The host warehouse remains the query planner, the RBAC boundary, the
billing boundary, the audit root, and the UI. ruLake adds a UDF and a
lineage edge. That is all.

If a prospect asks "should we rip out BigQuery?", the answer is "no,
we plug in."

### Not a storage system

ruLake has no storage service. Bytes live in S3 / GCS / Azure Blob
with the customer's existing bucket-level governance. We do not run
replicated storage, we do not manage durability, we do not charge for
storage. If the bucket burns, ruLake burns.

### Not a new table format

ruLake **rides on** existing table formats — Iceberg v2 primarily,
Delta via Iceberg interop. ruLake adds a convention (a `.rvf` sidecar
referenced by table properties), not a new open-format standard.
Talking about "the ruLake table format" is wrong; it is
"Iceberg + ruLake sidecars."

### Not an embedding / model / featureization service

ruLake does not produce embeddings. Customers bring their own model
(OpenAI, Cohere, Vertex AI, open-source, internal). ruLake stores and
serves vectors; it does not compute them.

### Not a real-time streaming system

ruLake's ingest path is batch-shaped. Append-only segments + daily
compaction is OLAP ergonomics. If a customer needs sub-second
ingest-to-query, we point them at `rvf-server`'s TCP streaming mode
(which is real-time) but that is not the ruLake product.

### Not quantum-safe out of the box

`rvf-crypto` supports ML-DSA-65 and SLH-DSA-128s. ruLake **optionally
enables them for bundle signing**. Per-row encryption at rest with
customer-managed PQ keys is a v2 line item. Today's ruLake bundle
is signed with PQ signatures but encrypted at rest with GCS-managed
(classical) keys. This is fine for 2026 but will need revisiting.

### Not a sub-millisecond query serving tier

See `05-performance-budget.md`. The BQ Tier-1 path is dominated by
HTTPS round-trip, ~30–80 ms warm. For sub-ms use cases, customers
embed ruLake (DuckDB extension, WASM tile, or direct `rvf-runtime`
use) — but that is not the BQ story and must be documented
separately.

### Not a data mesh

We track lineage. We do not build a mesh. Integration with
Starburst's data products, Atlan's metadata catalog, or Soda's data
contracts are all out of scope for v1.

### Not a MLOps platform

Model registry, feature store, experiment tracking, training
pipelines — all out of scope. ruLake plugs into whichever one the
customer runs. `rvf-federation` carries federated-learning primitives
today; their exposure as a product is a separate spike.

### Not production-ready as of v1 completion

v1 of this spike produces: a working BQ integration on a single
region with a single-instance UDF, DuckDB extension, Iceberg
manifests, and the governance story. It does **not** produce:
multi-region replication, active-active, HA, at-scale SRE runbooks,
or a support organisation. Those are post-spike.

---

## Hype-Avoidance Rubric

If any of these sentences shows up in a talk, a landing page, or a
sales deck, flag it:

| Suspect claim                                     | Why it is suspect / what to say instead |
|---------------------------------------------------|-----------------------------------------|
| "Fastest vector database."                        | We are not a database. Say "fastest **embedded** vector kernel we have measured on a laptop, 957 QPS at 100% recall@10 at n=100k — see BENCHMARK.md." |
| "Billion-scale vector search."                    | We have measured to n=100k. Billion-scale is a 2026-H2 acceptance target. Say "designed for billion-scale, measured at n=100k, SIFT1M benchmark is a tracked follow-up." |
| "Built-in GDPR compliance."                       | We provide the orchestration. Compliance is the customer's — and legal's — call. Say "GDPR orchestration primitives with a documented two-phase delete SLA." |
| "Zero-ops vector search inside BigQuery."         | The Cloud Run UDF is ops. Say "vector search inside BigQuery with one Cloud Run service per region." |
| "Quantum-resistant by default."                   | Only the signatures are PQ; encryption is classical GCS. Say "post-quantum signatures (ML-DSA-65); classical encryption at rest in v1." |
| "Provably correct query results."                 | Witness chain proves read integrity, not correctness. Say "witness-chain-backed audit trail for every query." |
| "AI-native data lake."                            | Say literally anything else. |
| "Eliminates your vector database."                | See "Not a vector database" above. Say "alternative to standing up a separate vector database when your requirements are datalake-shaped." |
| "100% recall."                                    | Only on our clustered Gaussian fixture. SIFT1M is unmeasured. Say "100% recall@10 at n=100k on BENCHMARK.md fixture; SIFT1M target, unmeasured." |
| "Drop-in replacement for Pinecone."               | See above. Say "complementary to Pinecone for OLAP-shaped vector workloads inside the datalake." |

When in doubt, the grounding test is: **can an engineer reproduce the
claim from a file in the repo in under 30 minutes?** If no, rewrite.

---

## Three Customer Shapes Where ruLake Wins

These are the shapes of prospect where the spike actually produces
value. If the prospect does not look like one of these, step away.

1. **"We want vector search but our security team said no new
   systems."** The warehouse is BQ or Snowflake, governance lives in
   Dataplex / Unity, and standing up a Pinecone cluster requires a
   6-month security review. ruLake is a Cloud Run service and a
   remote function — much smaller attack surface.

2. **"We need the same vector index readable from BQ and from
   laptops."** Data-science team runs notebooks with DuckDB against a
   GCS bucket; the production query path is BQ. ruLake's
   bundle-plus-UDF shape is the only design that makes that one
   bundle.

3. **"We have to prove to an auditor that vector X was retrieved by
   job Y on date Z."** Witness chains + lineage edges produce
   cryptographic provenance the auditor can replay offline. BQ's
   audit logs alone do not do this.

## Three Customer Shapes Where ruLake Loses

State these out loud. Walking away early is cheaper.

1. **"We need sub-millisecond p99."** The BQ path is fundamentally
   HTTPS-shaped. Point them at embedded ruLake (DuckDB, WASM) or a
   dedicated vector DB.
2. **"We need real-time feature store ingest at 100k rows/s."**
   Append-only + nightly compaction is wrong shape. Point them at a
   streaming vector store.
3. **"We only use BigQuery and BQ Vector Search meets our needs."**
   Let them. ruLake's portability argument is moot here.

---

## The One-Line Pitch

> ruLake is the adapter layer that makes a `.rvf` vector bundle read
> like a regular column inside BigQuery, DuckDB, and Iceberg-aware
> engines — so you do not have to stand up a second system of record
> to do vector search.

If the one-line pitch starts to grow, it is drifting.
