# ADR-094: π.ruv.io Shared Web Memory on RuVector

**Status**: Proposed
**Date**: 2026-03-14
**Authors**: ruv
**Deciders**: ruv, RuVector Architecture Team
**Technical Area**: Web-scale ingestion / Shared agent memory / Cloud orchestration / Compression / Temporal storage
**Related**: ADR-030 (RVF Computational Container), ADR-040 (Cognitum Swarm), ADR-047 (Proof-Gated Mutation Protocol), ADR-058 (MCP Tool Groups), ADR-059 (Shared Brain — Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-077 (Midstream Platform Integration), ADR-091 (INT8 CNN Quantization), ADR-017 (Temporal Tensor Compression)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-03-14 | ruv | Initial proposal — architecture, storage model, ingestion pipeline |
| 0.2 | 2026-03-15 | RuVector Team | Added implementation phases, cost model, acceptance criteria |

---

## Decision Statement

**ADR-094 establishes π.ruv.io as a RuVector-native shared web memory platform that ingests large public corpora (starting with Common Crawl), compresses them into structured semantic memory objects, and exposes them as a queryable knowledge substrate for agents, models, and users.**

This decision prioritizes:
- Retrieval-first architecture over model-training-first pipelines
- Temporal compression as a first-class storage primitive (ADR-017)
- Proof-gated mutation for all agent writes (ADR-047)
- Hybrid local/cloud deployment for cost control
- Coherence and contrast as query primitives via mincut boundaries

**Acceptance Benchmark**: Ingest ≥1M Common Crawl pages, compress to ≤40% of raw embedding storage via temporal deduplication, achieve p95 semantic retrieval latency ≤50ms for 10M memory objects, with full provenance chains on every stored object.

---

## 1. Context and Problem Statement

### 1.1 Current State

π.ruv.io currently operates as a shared brain for Claude Code sessions (ADR-059, ADR-060). The `mcp-brain-server` crate provides:

| Component | Path | Current State |
|-----------|------|---------------|
| Memory store | `crates/mcp-brain-server/src/store.rs` | DashMap + Firestore, session-contributed memories |
| Knowledge graph | `crates/mcp-brain-server/src/graph.rs` | ruvector-mincut + ruvector-solver PPR search |
| Embeddings | `crates/mcp-brain-server/src/embeddings.rs` | HashEmbedder + RlmEmbedder, 128-dim |
| Cognitive engine | `crates/mcp-brain-server/src/cognitive.rs` | Hopfield + DentateGyrus + HDC |
| Midstream | `crates/mcp-brain-server/src/midstream.rs` | Lyapunov attractor + temporal solver + strange loop |
| RVF pipeline | `crates/mcp-brain-server/src/pipeline.rs` | VEC + META + WITNESS + DP segments |
| Temporal tracking | `types.rs` (DeltaStream) | VectorDelta per-memory, knowledge velocity |

### 1.2 Problem

The current system accepts only agent-contributed memories from Claude Code sessions. There is no mechanism to:

1. **Ingest web-scale corpora** — Common Crawl alone is ~3.5 billion pages per crawl
2. **Compress and retain only semantic structure** — raw HTML wastes storage; boilerplate templates repeat across millions of pages
3. **Track temporal evolution** — the same URL may change content across crawls; deltas must collapse when novelty is low
4. **Build graph structure from hyperlinks** — link topology encodes authority, relevance, and domain relationships
5. **Detect contradictions** — conflicting claims across sources are coherence signals, not noise
6. **Scale writes safely** — agent swarms reading from and writing to shared memory need proof-gated mutation at scale

Without this architecture, π.ruv.io risks becoming:
- An expensive web archive (storing raw data without compression)
- A conventional vector database (embeddings without graph or temporal structure)
- An opaque training pipeline (data in, model out, no shared retrieval)
- A static RAG backend (retrieval without shared learning or evolution tracking)

### 1.3 Opportunity

Treating the public web as a continuously evolving external memory layer where:
- Pages become structured **memory objects** (`WebMemory`)
- Semantic meaning becomes **vector state** (128-dim embeddings via ruvLLM)
- Links and references become **graph structure** (KnowledgeGraph edges with PPR)
- Change over time becomes **temporal deltas** (DeltaStream from ruvector-delta-core)
- Contradictions become **coherence signals** (mincut partition boundaries)
- Writes remain **proof-gated** (ADR-047 ProofGate<T>)

---

## 2. Decision

We will implement π.ruv.io as a RuVector-native shared web memory platform with the following core decisions:

### 2.1 RuVector is the System of Record for Semantic Memory

All cleaned web content, embeddings, graph relationships, temporal deltas, and coherence metadata are stored in RuVector's memory plane. The `mcp-brain-server` store is extended with web-specific types (`WebMemory`, `WebPageDelta`, `LinkEdge`) that compose with the existing `BrainMemory` infrastructure.

### 2.2 Retrieval-First, Not Model-Training-First

Initial implementation focuses on shared retrieval, reasoning, clustering, provenance, and agent memory. The existing search, partition, and transfer APIs (ADR-059) are extended for web-scale queries. Fine-tuning or model training is optional and outside the critical path — the LoRA federation (ADR-060) can optionally train on web-derived preference signals.

### 2.3 Temporal Compression is a First-Class Primitive

Memory is stored as stable state plus deltas over time (extending `DeltaStream<VectorDelta>` from ADR-017). Repeated, templated, or low-novelty content collapses aggressively:

| Content Signal | Compression Action |
|---|---|
| Boilerplate (nav, footer, cookie banners) | Strip before embedding; deduplicate via content hash |
| Template pages (product listings, directory entries) | Collapse to schema + per-instance delta |
| Low-novelty recrawls (< 5% content change) | Store as temporal delta, not full re-embedding |
| Near-duplicate pages (cosine > 0.98) | Merge into cluster centroid with provenance list |
| Contradictory content (mincut boundary crossing) | Preserve both sides with contradiction edge |

### 2.4 Hybrid Local/Cloud Deployment

| Layer | Hardware | Responsibility |
|---|---|---|
| Fetch + filter | Cloud (Cloud Run jobs) | WARC download, language detection, robots.txt compliance |
| Preprocess | Local (Mac Studio M2 Ultra / Mac mini M4) | HTML cleaning, deduplication, chunking, optional local embedding via ruvLLM |
| Ingest API | Cloud (Cloud Run service) | Validation, job dispatch, write orchestration, proof verification |
| Storage | Cloud (Firestore + GCS) | Persistent memory objects, RVF containers, temporal deltas |
| Query | Cloud (Cloud Run service) | Semantic search, graph traversal, coherence analysis |

### 2.5 Proof-Gated Mutation Governs All Writes

All agent-contributed memory updates must carry provenance, policy validation, and mutation proofs before becoming canonical (ADR-047). Web-ingested content uses a simplified proof path:

```
ProofRequirement::Composite(vec![
    ProofRequirement::TypeMatch { schema_id: WEB_MEMORY_SCHEMA },
    ProofRequirement::InvariantPreserved { invariant_id: PROVENANCE_CHAIN },
    ProofRequirement::CoherenceBound { min_coherence: 0.3 },
])
```

### 2.6 Coherence and Contrast are Query Primitives

Dynamic mincut (ruvector-mincut), contradiction edges, graph partition boundaries, and novelty scoring are exposed as first-class indexing and reasoning features:

- `GET /v1/web/contradictions?topic=X` — find conflicting claims across sources
- `GET /v1/web/novelty?since=2026-03-01` — detect newly emerging knowledge clusters
- `GET /v1/web/coherence?cluster_id=N` — measure internal consistency of a knowledge cluster
- `GET /v1/web/evolution?url=X` — temporal delta history for a specific source

---

## 3. Decision Drivers

| Driver | Requirement | Metric |
|---|---|---|
| Cost | Avoid full raw-cloud retention and excessive managed embedding spend | ≤$500/month for 10M memory objects |
| Auditability | Preserve source provenance and mutation history | 100% of objects have witness chains |
| Shared intelligence | Many agents contributing to and reading from one memory substrate | Support ≥100 concurrent agent sessions |
| Compression | Exploit RuVector temporal compression and structured retention | ≥60% storage reduction vs. raw embeddings |
| Performance | Semantic retrieval and graph traversal at production latency | p95 ≤50ms for search, p95 ≤100ms for graph traversal |
| Safety | Proof-gated write paths and scoped authority | Zero unprovenanced writes to canonical memory |
| Extensibility | Support future RVF packaging, agent swarms, and Cognitum edge nodes | Clean bounded contexts per DDD |

---

## 4. High-Level Architecture

### 4.1 Logical Topology

```text
Common Crawl / Public Corpora
        │
        ▼
┌─────────────────────┐
│  Fetch + Filter      │  Cloud Run Jobs
│  WARC download       │  Language detection
│  robots.txt check    │  Size/quality gates
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Local Preprocess    │  Mac Studio / Mac mini
│  HTML → text         │  Boilerplate strip
│  Deduplication       │  Content hashing
│  Chunking (512 tok)  │  ruvLLM embedding (opt)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cloud Ingest API    │  Cloud Run (π.ruv.io)
│  Validation          │  Proof verification
│  Job dispatch        │  Write orchestration
│  Rate limiting       │  Byzantine aggregation
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│  RuVector Shared Memory Plane            │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Vectors  │ │  Graph   │ │ Temporal │ │
│  │ 128-dim  │ │ MinCut   │ │ Deltas   │ │
│  │ HNSW idx │ │ PPR rank │ │ ADR-017  │ │
│  └──────────┘ └──────────┘ └──────────┘ │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Coherence │ │Provenance│ │ Midstream│ │
│  │ scoring  │ │ witness  │ │ attractor│ │
│  │ mincut   │ │ chains   │ │ solver   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────┬───────────────────┬──────────┘
           │                   │
           ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│ Agent Query API │  │ Admin/Analytics │
│ semantic search │  │ health, compact │
│ graph traverse  │  │ drift, audit    │
│ contrast query  │  │ cost tracking   │
└─────────────────┘  └─────────────────┘
```

### 4.2 Data Model

#### WebMemory (extends BrainMemory)

```rust
/// A web-sourced memory object in the shared memory plane
pub struct WebMemory {
    /// Core brain memory fields (embedding, quality, witness, etc.)
    pub base: BrainMemory,
    /// Source URL (canonical, after redirect resolution)
    pub source_url: String,
    /// Domain extracted from source_url
    pub domain: String,
    /// Content hash (SHAKE-256) for deduplication
    pub content_hash: String,
    /// Crawl timestamp (when the content was fetched)
    pub crawl_timestamp: DateTime<Utc>,
    /// Crawl source identifier (e.g., "cc-2026-09")
    pub crawl_source: String,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Outbound link URLs (for graph construction)
    pub outbound_links: Vec<String>,
    /// Temporal compression tier
    pub compression_tier: CompressionTier,
    /// Novelty score relative to existing memory (0.0 = duplicate, 1.0 = entirely new)
    pub novelty_score: f32,
}

/// Temporal compression tiers (ADR-017 alignment)
pub enum CompressionTier {
    /// Full embedding + content stored (high novelty, first seen)
    Full,
    /// Embedding stored, content as delta from nearest neighbor
    DeltaCompressed,
    /// Only centroid contribution stored (near-duplicate)
    CentroidMerged,
    /// Archived — retrievable from GCS but not in hot memory
    Archived,
}
```

#### WebPageDelta (temporal evolution)

```rust
/// Tracks how a web page changes across crawls
pub struct WebPageDelta {
    pub page_url: String,
    pub previous_memory_id: Uuid,
    pub current_memory_id: Uuid,
    /// Cosine similarity between previous and current embeddings
    pub embedding_drift: f32,
    /// Content diff summary (not raw diff — structured delta)
    pub content_delta: ContentDelta,
    /// Crawl interval
    pub time_delta: Duration,
    /// Whether this delta crossed a mincut boundary
    pub boundary_crossing: bool,
}

pub enum ContentDelta {
    /// Content unchanged (hash match)
    Unchanged,
    /// Minor update (< 5% token change)
    Minor { changed_tokens: usize, total_tokens: usize },
    /// Major revision (≥ 5% token change)
    Major { summary: String, changed_tokens: usize },
    /// Complete rewrite (cosine < 0.7)
    Rewrite,
}
```

#### LinkEdge (graph construction)

```rust
/// A directed edge from source page to target page
pub struct LinkEdge {
    pub source_memory_id: Uuid,
    pub target_memory_id: Uuid,
    /// Anchor text embedding (if meaningful anchor text exists)
    pub anchor_embedding: Option<Vec<f32>>,
    /// Link context: surrounding paragraph embedding
    pub context_embedding: Vec<f32>,
    /// Link type classification
    pub link_type: LinkType,
    /// Weight based on semantic relevance of anchor + context
    pub weight: f64,
}

pub enum LinkType {
    /// Informational reference
    Citation,
    /// Navigational (same-site)
    Navigation,
    /// Supporting evidence
    Evidence,
    /// Contradiction or rebuttal
    Contradiction,
    /// Unknown / unclassified
    Unknown,
}
```

### 4.3 Ingestion Pipeline

```text
Phase 1: Fetch
  WARC segment → HTTP response → (url, html, headers, timestamp)
  Gates: robots.txt compliance, language filter, size limit (1MB)

Phase 2: Clean
  HTML → readable text (via readability/trafilatura equivalent)
  Strip: nav, footer, ads, cookie banners, scripts
  Extract: title, main content, outbound links, meta description
  Output: CleanedPage { url, text, links, title, meta }

Phase 3: Deduplicate
  Content hash (SHAKE-256 of normalized text)
  Check against content_hash index in store
  If exact match → skip (or record temporal "unchanged" delta)
  If near-match (simhash within 3 bits) → flag for delta compression

Phase 4: Chunk + Embed
  Split text into 512-token chunks (with 64-token overlap)
  Generate 128-dim embeddings via ruvLLM HashEmbedder
  Optional: local ruvLLM RlmEmbedder for higher quality

Phase 5: Novelty Score
  Compare each chunk embedding against nearest neighbors in HNSW index
  Novelty = 1.0 - max_cosine_similarity(chunk, existing_memories)
  If novelty < 0.05 → CentroidMerged tier
  If novelty < 0.20 → DeltaCompressed tier
  Else → Full tier

Phase 6: Graph Construction
  For each outbound link in CleanedPage:
    Resolve target URL → target memory_id (if exists)
    Classify link type from anchor text + context
    Create LinkEdge with semantic weight

Phase 7: Proof + Store
  Construct ProofRequirement (Section 2.5)
  Build RVF container (pipeline.rs extension)
  Store WebMemory + LinkEdges + WebPageDeltas
  Update KnowledgeGraph (graph.rs)
  Record witness chain
```

### 4.4 Midstream Integration

The existing midstream crate (`crates/mcp-brain-server/src/midstream.rs`) provides three capabilities critical to web memory:

| Midstream Component | Web Memory Application |
|---|---|
| `temporal-attractor-studio` (Lyapunov) | Detect which knowledge domains are stable vs. chaotic. Stable domains (negative λ) compress more aggressively. Chaotic domains (positive λ) retain more temporal deltas. |
| `temporal-neural-solver` (Certified prediction) | Predict future content drift for scheduling recrawl priority. High-confidence stability predictions → lower crawl frequency. |
| `strange-loop` (Meta-cognition) | Evaluate query relevance × memory quality for web search results. The 5ms budget per query adds meta-cognitive scoring without latency impact. |
| `nanosecond-scheduler` | Schedule background compaction, recrawl triggers, and temporal delta aggregation with nanosecond precision. |

### 4.5 New API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/web/ingest` | Submit a batch of cleaned pages for ingestion |
| `GET` | `/v1/web/search` | Semantic search across web memory (extends `/v1/search`) |
| `GET` | `/v1/web/contradictions` | Find conflicting claims across sources |
| `GET` | `/v1/web/novelty` | Detect newly emerging knowledge clusters |
| `GET` | `/v1/web/coherence` | Measure internal consistency of a cluster |
| `GET` | `/v1/web/evolution` | Temporal delta history for a URL or topic |
| `GET` | `/v1/web/graph` | Subgraph extraction around a memory or topic |
| `POST` | `/v1/web/recrawl` | Request recrawl of specific URLs |
| `GET` | `/v1/web/status` | Web memory statistics and pipeline health |
| `GET` | `/v1/web/domains` | Domain-level statistics and authority scores |

---

## 5. Compression and Storage Model

### 5.1 Storage Budget

| Item | Per Object | At 10M Objects | At 100M Objects |
|---|---|---|---|
| Embedding (128 × f32) | 512 B | 4.8 GB | 48 GB |
| Metadata (JSON compressed) | ~200 B | 1.9 GB | 19 GB |
| Graph edges (avg 5/page) | ~120 B | 1.1 GB | 11 GB |
| Temporal deltas (avg 2/page) | ~80 B | 0.7 GB | 7.5 GB |
| Witness chain | ~82 B | 0.8 GB | 7.8 GB |
| **Total (hot)** | **~994 B** | **9.3 GB** | **93 GB** |

With temporal compression at 60% reduction: **3.7 GB for 10M objects** in hot memory.

### 5.2 Tiered Storage

| Tier | Location | Latency | Retention |
|---|---|---|---|
| Hot | DashMap (in-memory) | <1ms | Active + high-quality memories |
| Warm | Firestore | ~10ms | All canonical memories with recent access |
| Cold | GCS (RVF containers) | ~100ms | Full archive, low-access memories |
| Frozen | GCS Archive class | ~1s | Historical snapshots, compacted deltas |

### 5.3 Compaction Policy

Background compaction runs on the nanosecond-scheduler:

1. **Centroid merge**: Near-duplicates (cosine > 0.98) merge into cluster centroids every 6 hours
2. **Delta collapse**: Sequential minor deltas (< 5% change each) collapse into a single major delta daily
3. **Tier promotion/demotion**: Access-frequency-based movement between hot/warm/cold every hour
4. **Archive sweep**: Memories with quality_score < 0.2 after 30 days move to frozen tier

---

## 6. Emergent Capabilities

The combination of web-scale ingestion, RuVector's graph + temporal + coherence primitives, and the shared memory plane enables capabilities that go beyond conventional search or RAG:

### 6.1 Collective Intelligence for Agent Swarms

Agents contribute patterns, solutions, and observations into the shared substrate. Web memory provides the foundational knowledge layer that agent contributions build upon. Instead of every agent rediscovering known solutions, the web memory serves as a global baseline.

### 6.2 Autonomous Research Engine

Continuous research agents can:
1. Semantic retrieval across the web corpus
2. Pattern extraction via Hopfield recall
3. Contradiction detection via mincut boundaries
4. New hypothesis storage back into the graph

### 6.3 Knowledge Cartography

RuVector's combination of vectors + graphs + mincut boundaries + coherence scoring enables mapping knowledge topology: emerging research fields, scientific disagreements, technological trends, cross-disciplinary idea clusters.

### 6.4 Truth Verification Infrastructure

Claims stored with supporting sources, semantic similarity, contradiction edges, and confidence scoring enable evidence-backed reasoning. Agents retrieve not only answers but evidence graphs — aligned with proof-gated mutation (ADR-047).

### 6.5 Knowledge Evolution Tracking

Temporal delta tracking observes how knowledge evolves: consensus formation, misinformation propagation, idea propagation through research communities. The attractor analysis (midstream) identifies stable vs. chaotic knowledge domains.

### 6.6 Structural Search

Traditional search: "what pages mention this?"
π.ruv.io search: "what does humanity know about this problem?"

The shift from document retrieval to knowledge reasoning is enabled by graph traversal + coherence analysis + temporal context.

---

## 7. Cost Model

### 7.1 Preprocessing (Local)

| Resource | Specification | Cost |
|---|---|---|
| Mac Studio M2 Ultra | 192 GB unified memory, 24-core CPU | One-time ($3,999) |
| Storage (local NVMe) | 2 TB for WARC staging | Included |
| Power | ~50W average | ~$5/month |
| **Preprocessing throughput** | **~10K pages/hour (embed + dedupe)** | |

### 7.2 Cloud (Google Cloud)

| Service | Usage | Monthly Cost |
|---|---|---|
| Cloud Run (ingest API) | 2 vCPU, 4 GB, always-on | ~$65 |
| Cloud Run (query API) | Same as existing brain server | ~$65 |
| Firestore | 10M documents, 100K reads/day | ~$50 |
| GCS (RVF containers) | 50 GB Standard, 500 GB Archive | ~$15 |
| Cloud Tasks (job queue) | 1M tasks/month | ~$5 |
| **Total monthly** | | **~$200** |

### 7.3 Scaling Projection

| Scale | Hot Memory | Monthly Cloud | Preprocessing Time |
|---|---|---|---|
| 1M pages | ~370 MB | ~$100 | ~4 days (Mac Studio) |
| 10M pages | ~3.7 GB | ~$200 | ~6 weeks |
| 100M pages | ~37 GB | ~$500 | ~14 months |
| 1B pages | ~370 GB (tiered) | ~$2,000 | Distributed cluster |

---

## 8. Implementation Phases

### Phase 1: Data Model + Types (Week 1-2)

**Files**: `crates/mcp-brain-server/src/web_memory.rs` (new)

- Define `WebMemory`, `WebPageDelta`, `LinkEdge`, `CompressionTier`, `ContentDelta`, `LinkType`
- Extend `FirestoreClient` with web memory CRUD operations
- Add web-specific categories to `BrainCategory::Custom`
- Integration tests for type serialization round-trips

### Phase 2: Ingestion Pipeline (Week 3-5)

**Files**: `crates/mcp-brain-server/src/web_ingest.rs` (new)

- HTML cleaning (strip boilerplate, extract content)
- Content hashing (SHAKE-256 deduplication)
- Chunking (512 tokens, 64-token overlap)
- Novelty scoring against existing HNSW index
- Compression tier assignment
- Batch ingestion endpoint (`POST /v1/web/ingest`)

### Phase 3: Graph Construction (Week 5-7)

**Files**: extend `crates/mcp-brain-server/src/graph.rs`

- Link extraction and resolution
- LinkEdge creation with semantic weighting
- Contradiction detection via mincut boundary analysis
- PPR-ranked graph traversal for web subgraphs

### Phase 4: Temporal Compression (Week 7-9)

**Files**: extend `crates/mcp-brain-server/src/drift.rs`, new `web_temporal.rs`

- WebPageDelta tracking across recrawls
- Compaction policy implementation on nanosecond-scheduler
- Centroid merge for near-duplicates
- Delta collapse for sequential minor changes
- Tiered storage promotion/demotion

### Phase 5: Query APIs (Week 9-11)

**Files**: extend `crates/mcp-brain-server/src/routes.rs`

- `/v1/web/search` — semantic + graph-ranked web memory search
- `/v1/web/contradictions` — mincut boundary queries
- `/v1/web/novelty` — emerging cluster detection
- `/v1/web/evolution` — temporal delta history
- `/v1/web/coherence` — cluster consistency measurement
- Midstream scoring integration (attractor + solver + strange loop)

### Phase 6: Local Preprocessing CLI (Week 11-13)

**Files**: `crates/mcp-brain-server/src/preprocess.rs` (new), extend `npm/packages/ruvector/bin/cli.js`

- WARC reader for Common Crawl segments
- Local HTML cleaning pipeline
- Batch embedding via ruvLLM
- Upload to Cloud Ingest API
- CLI commands: `npx ruvector web ingest`, `npx ruvector web status`

### Phase 7: Hardening + Acceptance (Week 13-14)

- Load testing at 10M objects
- p95 latency validation (≤50ms search, ≤100ms graph)
- Compression ratio validation (≥60%)
- Proof chain integrity audit
- Cost model validation against projections

---

## 9. Invariants

| ID | Invariant | Enforcement |
|---|---|---|
| INV-1 | Every WebMemory has a non-empty provenance chain | `build_rvf_container` requires witness_chain |
| INV-2 | Content hash is unique per Full-tier memory | SHAKE-256 dedup check before store |
| INV-3 | Agent writes are proof-gated | ProofGate<WebMemory> wraps mutation path |
| INV-4 | Temporal deltas reference valid parent memories | Foreign key check on previous_memory_id |
| INV-5 | Compression tier assignment matches novelty score | Tier = f(novelty_score) is deterministic |
| INV-6 | LinkEdge weights are in [0.0, 1.0] | Clamped at construction |
| INV-7 | Hot memory fits within configured budget | Compaction triggers when hot tier exceeds threshold |
| INV-8 | Contradiction edges are symmetric | If A contradicts B, B contradicts A |

---

## 10. Security Considerations

### 10.1 Inherited from ADR-059

All seven security layers from the Shared Brain apply:
1. Input sanitization (PII strip)
2. Differential privacy (ε=1.0, δ=1e-5)
3. Ed25519 signatures
4. Witness chains (SHAKE-256)
5. Byzantine-tolerant aggregation
6. Rate limiting (BudgetTokenBucket)
7. Reputation-gated writes

### 10.2 Web-Specific Threats

| Threat | Mitigation |
|---|---|
| SEO spam injection | Quality gating: novelty < 0.05 + low authority domain → reject |
| Link manipulation | PageRank-style authority scoring via ruvector-solver PPR |
| Content poisoning | Contradiction detection flags conflicting claims for review |
| Copyright claims | robots.txt compliance; only store embeddings + structured metadata, not raw content |
| Crawl budget abuse | Rate limiting on ingest API; batch size caps |

---

## 11. Alternatives Considered

### 11.1 Use Common Crawl as LLM Training Data

**Rejected**: Training produces a static model that cannot be queried structurally, lacks provenance, and requires expensive retraining. The retrieval-first approach preserves auditability and supports continuous evolution.

### 11.2 Use a Managed Vector Database (Pinecone, Weaviate)

**Rejected**: External managed services add latency, cost, and vendor lock-in. They lack graph structure, temporal compression, mincut coherence, and proof-gated mutation. RuVector provides all of these natively.

### 11.3 Store Raw HTML in Object Storage

**Rejected**: Raw HTML storage is expensive at scale and provides no semantic structure. The cleaning + embedding + graph construction pipeline is essential for the platform to be useful as a knowledge substrate rather than a web archive.

### 11.4 Build a Separate Ingestion Service

**Rejected**: The ingestion pipeline shares types, embedding infrastructure, graph construction, and proof verification with the existing brain server. A separate service would duplicate these dependencies. Instead, web memory is implemented as an extension module within `mcp-brain-server`.

---

## 12. Future Directions

1. **Developer Platform API**: Expose π.ruv.io as a shared memory service for any agent framework, not just Claude Code sessions
2. **Automatic Contradiction Detection**: Use dynamic mincut + coherence scoring to detect contradictions and emerging ideas automatically
3. **Cognitum Edge Nodes**: Deploy compressed web memory subsets to edge devices (ADR-040) for offline agent reasoning
4. **RVF Knowledge Export**: Package web memory clusters as RVF cognitive containers (ADR-056) for transfer between deployments
5. **Cross-Domain Transfer Learning**: Use web memory as a foundation for domain expansion (ADR-068) across specialized agent populations

---

## 13. Acceptance Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Ingestion throughput | ≥1K pages/second (cloud batch) | Load test with 100K page batch |
| Storage compression | ≥60% reduction vs. raw embeddings | Compare raw vs. compressed storage at 1M objects |
| Search latency (p95) | ≤50ms for semantic search | Benchmark with 10M objects, 100 concurrent queries |
| Graph traversal (p95) | ≤100ms for 2-hop subgraph | Benchmark with 10M nodes, 50M edges |
| Provenance coverage | 100% of objects have witness chains | Audit query: count objects without witness_chain |
| Contradiction detection | ≥80% precision on labeled test set | Manual evaluation of 500 flagged contradictions |
| Cost | ≤$500/month at 10M objects | Monthly billing review |
| Proof-gate coverage | Zero unprovenanced writes | Audit log analysis |

---

## 14. References

- ADR-017: Temporal Tensor Compression with Tiered Quantization
- ADR-030: RVF Computational Container
- ADR-040: Cognitum Swarm
- ADR-047: Proof-Gated Mutation Protocol
- ADR-058: MCP Tool Groups
- ADR-059: Shared Brain — Google Cloud Deployment
- ADR-060: Shared Brain Capabilities — Federated MicroLoRA Intelligence Substrate
- ADR-077: Midstream Platform Integration
- ADR-091: INT8 CNN Quantization
- Common Crawl: https://commoncrawl.org/
- RuVector compression and temporal tiering architecture
- Contrastive AI framing and proof-gated mutation principles
