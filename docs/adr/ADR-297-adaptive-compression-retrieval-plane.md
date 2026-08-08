# ADR-297: Adaptive Compression & Retrieval Plane (ACRP)

- **Status**: Accepted
- **Date**: 2026-08-06
- **Extends**: ADR-296 (Turbo4 quantized vector datatype), ADR-254 (turbovec), ADR-026 (tiered routing)
- **Related crates**: `ruvector-turboquant`, `ruvector-core`, `ruvector-rabitq`, `ruvector-pq-search`, `ruvector-attn-mincut`, `ruvector-agent-memory`, `rvf`

## Context

ADR-296 gives RuVector a faithful Turbo4 datatype: uniform 4-bit storage with
direct packed scoring inside HNSW. That matches Qdrant's headline capability —
but stopping there means *copying* Turbo4. The defensible advantage is an
**adaptive compression and retrieval plane**: one system that chooses vector
representations by workload, because retrieval errors are not equally costly
across vectors, queries, or time.

Today's obstacles:

- Compression code is scattered across disconnected crates
  (`ruvector-core::quantization`, `ruvector-turboquant`, `ruvector-rabitq`,
  `ruvector-pq-search`, `ruvector-turbovec`, `ruvllm`), each with its own
  types; storage, HNSW, snapshots, WASM, and bindings cannot consume them
  interchangeably.
- Precision is a single global choice; queries that need more bits pay the
  same as queries that need fewer, and vice versa.
- No provenance: stored codes don't record which embedding model, codec
  version, or rotation seed produced them, so migrations and verification are
  not deterministic.

**Product claim to earn**: *RuVector automatically places each vector at the
cheapest precision that preserves its retrieval value.*

## Decision

### 1. One `EncodedVector` interface for every representation

`ruvector-core::encoding` defines the unified codec plane:

```rust
pub enum CodecKind { Fp32, Fp16, Int8, Turbo4, Pq, RaBitQ1 }

pub trait VectorCodec: Send + Sync {
    fn kind(&self) -> CodecKind;
    fn dim(&self) -> usize;
    fn encoded_len(&self) -> usize;                    // bytes per vector
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>>;
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>>; // approximation
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32>;       // symmetric
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>>;
}

pub trait EncodedQuery: Send + Sync {
    fn distance_to(&self, blob: &[u8]) -> f32;         // asymmetric
}
```

Storage, HNSW, DiskANN-style indexes, snapshots, WASM, and bindings consume
`&dyn VectorCodec` / blobs — never concrete codec types. Core ships `Fp32`,
`Fp16`, `Int8`, and `Turbo4` implementations; `Pq` and `RaBitQ1` implement the
same traits from their own crates (core reserves the `CodecKind` variants so
blob headers and provenance are stable). `ruvector-turboquant` becomes a
non-optional core dependency: it is dependency-free and WASM-safe, so the
codec plane exists on every build; only the HNSW index remains
feature-gated.

### 2. Storage precision ≠ search precision

The active search plane composes per role (~5 bits/dim effective):

| Role | Representation |
|------|----------------|
| Source storage | Turbo4 (`D/2 + 8` B) |
| Candidate index | RaBitQ 1-bit (`D/8` B) |
| Rescoring | Turbo4 exact-LUT |
| Critical verification | optional FP16 / FP32 |

Candidate generation traverses the graph on 1-bit codes (cheapest memory
bandwidth), rescoring uses the Turbo4 codes, and a configurable verification
tier re-checks results for critical policies.

### 3. Automatic precision selection (per query)

Measure query difficulty and spend bits only where needed:

- **Score margin** — relative gap between the last kept (k-th) and first
  dropped (k+1-th) rescored distances. Large margin ⇒ the Turbo4 result is
  already stable; return it.
- **Candidate stability** — if an escalated pass (higher `efSearch`, larger
  rescore pool) changes the top-k membership, keep escalating; if not, stop.
- **Critical policy** — callers can require exact (FP32 source or decode)
  verification.

Target: only ~5–15 % of queries take an escalation. The three-tier ladder is:
Turbo4 answer → widened traversal (2–3× ef, 2× rescore pool) → high-precision
verification.

### 4. Adaptive memory tiers (per vector, over time)

Promote by access frequency, demote by coldness — integrated with
`ruvector-agent-memory` and the temporal-coherence machinery:

| Tier | Representation |
|------|----------------|
| Hot | FP16 |
| Warm | Turbo4 |
| Cold | PQ |
| Archive | RaBitQ + source object reference |

### 5. Topology-aware precision allocation (per vector, by position)

Graph centrality and MinCut boundaries (via `ruvector-attn-mincut` /
`ruvector-graph`) drive bit allocation: bridge vectors, rare concepts, and
high-influence nodes get more bits; redundant vectors inside dense
communities get fewer. Retrieval errors on hubs and bridges poison many
queries; errors on redundant leaves poison almost none.

### 6. Streaming drift detection

Turbo4 is data-oblivious but its retrieval quality varies with model and
dimension; PQ codebooks go stale outright. Track per collection: recall
proxies (overlap between quantized and verified top-k on sampled queries),
score distortion, embedding-norm distribution, and embedding model identity.
Threshold failures trigger background re-encoding/migration.

### 7. Provenance (mandatory metadata)

Every stored vector carries a `VectorProvenance`:

```rust
pub struct VectorProvenance {
    pub model_id: Option<String>,   // embedding model identity
    pub codec: CodecKind,
    pub codec_version: u16,
    pub rotation_seed: Option<u64>,
    pub dim: usize,
    pub metric: DistanceMetric,
    pub source_hash: Option<String>, // hex sha-256 of source object
    pub lineage: Vec<String>,        // migration history entries
}
```

Snapshots remain deterministic and independently verifiable through RVF: the
same (source, provenance) always reproduces byte-identical codes (this is why
ADR-296 forbids `rand`-derived rotations).

### 8. Honest benchmarking

Corpora: SIFT1M, GIST1M, Deep1M, DBpedia embeddings, RuFlo memory vectors,
RuView spatial vectors. Metrics: Recall@1, Recall@10, NDCG, P50/P95/P99
latency, build time, ingest rate, disk, RSS, energy per million queries.
Harness lives in `ruvector-sota-bench`; results land in `bench_results/`.

### 9. Product surface: three policies

```rust
pub enum SearchPolicy { Quality, Balanced, MaxCompression }
```

Users pick an outcome, not an algorithm. Policies map to (rescore multiplier,
escalation threshold, escalation rounds, verification tier); `Balanced` is
the default. All lower-level knobs remain reachable for experts.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| A | ADR-296 phases 1–2 (Turbo4 codec + applied HNSW integration) | done (PR #802) |
| B | `EncodedVector`/`VectorCodec` plane in core (Fp32/Fp16/Int8/Turbo4); `VectorProvenance` schema; `SearchPolicy` + margin-based adaptive escalation v1 in the Turbo4 index | this ADR, first slice |
| C | RaBitQ1 candidate cascade (traverse 1-bit, rescore Turbo4); PQ/RaBitQ codecs implementing the plane traits from their crates | next |
| D | Provenance persisted in storage + RVF snapshot verification; drift monitors (recall proxy, distortion, norms, model id) | next |
| E | Memory tiers (hot/warm/cold/archive) with promotion/demotion driven by access stats | next |
| F | Topology-aware allocation (centrality/MinCut bit budgets) | next |
| G | `ruvector-sota-bench` acceptance runs + **ablation** (below); NEON/AVX-512/WASM kernels from ADR-296 phase 3 | next |

## Acceptance test (ablation-gated)

Across at least three real workloads (one public benchmark, RuFlo memory, one
embedding corpus):

- Adaptive mode must reduce total memory by **≥ 30 %** beyond uniform Turbo4,
- keep Recall@10 within **0.5 pp**, and
- keep P95 latency within **10 %** of the best fixed configuration.

The stated uncertainty — whether adaptive-precision complexity pays for
itself over uniform Turbo4 — is resolved by this ablation, not argued. If the
ablation fails, phases E/F stay experimental and the product surface ships
uniform Turbo4 with per-query escalation only (phase B), which is already
strictly better than a fixed pipeline.

## Consequences

- One interface ends the "N disconnected compression crates" problem; codecs
  become pluggable data, not architecture.
- Per-query adaptivity means headline P50 latency reflects 4-bit scoring
  while tail quality is protected by escalation — the recall/latency curve
  dominates any fixed-precision point.
- Complexity is contained: each phase lands behind the same trait plane, and
  the ablation gate prevents shipping adaptive machinery that doesn't pay.
- Provenance-first storage makes migrations (drift, tier moves, codec
  upgrades) auditable and reversible.
