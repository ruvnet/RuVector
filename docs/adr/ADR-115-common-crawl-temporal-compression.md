# ADR-115: Common Crawl Integration with Semantic Compression

**Status**: Proposed
**Date**: 2026-03-16
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: None
**Related**: ADR-096 (Cloud Pipeline), ADR-059 (Shared Brain), ADR-060 (Brain Capabilities), ADR-077 (Midstream Platform)

## 1. Executive Summary

**Core proposition**: Turn the open web into a compact, queryable, time-aware semantic memory layer for agents—with enough compression to move from expensive archive analytics to cheap always-on retrieval.

**Not**: "The whole web fits in 56 MB." That is a research hypothesis, not an established result.

**What we're building**: A compressed web memory service that provides:
- Queryable vector memory over Common Crawl
- Semantic cluster IDs and prototype exemplars
- Monthly deltas with provenance links
- Sub-50ms retrieval latency

## 2. Context

### 2.1 Common Crawl Scale

Common Crawl represents the largest public web archive:

| Metric | Value | Source |
|--------|-------|--------|
| Monthly crawl pages | 2.1-2.3 billion | [Feb 2026 release](https://commoncrawl.org/blog/february-2026-crawl-archive-now-available) |
| Monthly uncompressed size | 363-398 TiB | Common Crawl statistics |
| Total corpus (2008-present) | 300+ billion pages | Historical archives |
| Host-level graph edges | Billions | [Graph releases](https://commoncrawl.org/blog/host--and-domain-level-web-graphs-november-december-2025-and-january-2026) |

The challenge: this scale makes naive storage prohibitively expensive (~$5,000+/month for embeddings alone).

### 2.2 The Opportunity

RuVector's compression stack—PiQ quantization, MinCut clustering, SONA attractors—can potentially reduce this to manageable size. But compression claims must be validated empirically.

## 3. Three-Tier Value Framework

### 3.1 Tier 1: Practical Now (High Confidence)

Immediately useful as a **compressed semantic memory fabric**:

| Application | Description | Value |
|-------------|-------------|-------|
| **Domain memory for agents** | Store compressed embeddings, canonical clusters, temporal snapshots, attractor summaries | Retrieval over huge corpus without repeated frontier model calls |
| **Change detection & topic drift** | Bucket by crawl month, track cluster transitions | Detect when topics stabilize, domains shift stance, concepts fork |
| **Near real-time knowledge distillation** | Keep compressed attractor per semantic family + witness provenance + recency cache | Web-scale memory for summarization, routing, RAG |
| **Cheap multi-tenant retrieval** | Cloud Run's granular pricing (vCPU-second, GiB-second) | Small hot retrieval service vs giant search cluster |

### 3.2 Tier 2: High Value If Compression Works (Medium Confidence)

Requires empirical validation of compression ratios:

**Conservative Path** (established techniques):
1. PiQ-style quantization → meaningful first-order reduction
2. Semantic dedup → reduce near-duplicate pages
3. HNSW indexing → fast recall on remaining set
4. Temporal bucketing → reduce repeated storage across snapshots

**Aggressive Research Path** (exotic upside):
1. Cluster to prototypes
2. Distill clusters into attractors
3. Represent time as transitions between attractors
4. Reconstruct details on demand from exemplars

### 3.3 Tier 3: Exotic But Interesting (Research Hypothesis)

**A. Web-Scale Semantic Nervous System**

Model the web not as documents but as evolving attractor fields:
- Pages are observations
- Clusters are local semantic basins
- Attractors are stable concept states
- Temporal compression captures state transitions
- MinCut marks semantic fault lines

**Practical outputs**: Early controversy detection, narrative fracture maps, emerging concept birth detection, regime shift alerts.

**B. Memory Substrate for Swarm Reasoning**

Compressed attractors become shared memory for agent swarms:
- Cluster representatives
- Attractor deltas
- Witness-linked updates
- MinCut-based anomaly boundaries

**C. Historical Web Archaeology**

Time-indexed analysis enables:
- Topic lineage graphs
- Domain evolution traces
- Language drift maps
- "What changed when" semantic replay

**D. World Model Built from Contrast**

Treat the web structurally:
- Dense clusters = consensus regions
- Sparse bridges = weak agreements
- MinCuts = fault lines
- Temporal attractor jumps = worldview transitions

This is far more interesting than ordinary vector search.

## 4. Use Case Prioritization

| Use Case | Value | Technical Risk | Compression Tolerance | Near-Term Fit |
|----------|------:|---------------:|----------------------:|--------------:|
| Competitive intelligence | 9 | 4 | 8 | **9** |
| Trend and drift monitoring | 9 | 5 | 8 | **9** |
| Agent shared memory | 10 | 6 | 7 | **8** |
| Temporal web archaeology | 8 | 5 | 7 | **8** |
| General frontier knowledge store | 10 | 9 | 3 | 4 |
| Narrative fault line detection | 9 | 7 | 9 | 7 |
| Autonomous world model substrate | 10 | 10 | 5 | 3 |

**Recommendation**: Start with the top four, not the bottom three.

## 5. Decision

Build a **phased compressed web memory service**, starting with conservative techniques and validating exotic compression empirically.

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Common Crawl Ingestion Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Common Crawl S3          CDX Index Cache          π.ruv.io (Cloud Run)
  ─────────────────        ────────────────         ─────────────────────
  │                        │                        │
  │  WARC Archives         │  URL → (offset,len)    │  ┌──────────────────┐
  │  s3://commoncrawl/     │  Redis/Memorystore     │  │ CommonCrawlAdapter│
  │  crawl-data/           │  (~$8/mo)              │  │                  │
  │                        │                        │  │ • CDX queries    │
  └────────────┬───────────┘                        │  │ • WARC range-GET │
               │                                    │  │ • URL dedup      │
               │  Range GET (only needed bytes)     │  │ • Content dedup  │
               ▼                                    │  └────────┬─────────┘
  ┌────────────────────────┐                        │           │
  │    Extraction Layer    │                        │           ▼
  │    ─────────────────   │                        │  ┌──────────────────┐
  │    • HTML → text       │ ───────────────────────┼─►│  7-Phase Pipeline │
  │    • Boilerplate strip │    Streaming inject    │  │                  │
  │    • Language detect   │                        │  │ 1. Validate      │
  └────────────────────────┘                        │  │ 2. Dedupe (URL)  │
                                                    │  │ 3. Chunk         │
                                                    │  │ 4. Embed         │
                                                    │  │ 5. Novelty Score │
                                                    │  │ 6. Compress      │
                                                    │  │ 7. Store         │
                                                    │  └────────┬─────────┘
                                                    │           │
                                                    │           ▼
                                                    │  ┌──────────────────┐
                                                    │  │ Compression Stack│
                                                    │  │ (validated)      │
                                                    │  │ • PiQ3 (10.7x)   │
                                                    │  │ • SimHash dedup  │
                                                    │  │ • HNSW index     │
                                                    │  └────────┬─────────┘
                                                    │           │
                                                    │           ▼
                                                    │  ┌──────────────────┐
                                                    │  │ Exemplar Store   │
                                                    │  │                  │
                                                    │  │ • Cluster centroids
                                                    │  │ • Raw exemplars  │
                                                    │  │ • Witness chain  │
                                                    │  └──────────────────┘
                                                    └──────────────────────
```

### 5.2 Component Summary

| Component | Technology | Purpose | Cost |
|-----------|------------|---------|------|
| CDX Cache | Cloud Memorystore (Redis) | Cache Common Crawl CDX index queries | ~$8/mo |
| WARC Fetcher | reqwest + Range headers | Fetch only needed bytes from S3 | $0 (public bucket) |
| URL Deduplication | DashMap<hash, ()> | Skip previously seen URLs | ~2 GB RAM |
| Content Deduplication | SimHash/MinHash | Skip near-duplicate content | ~500 MB RAM |
| PiQ3 Quantizer | ruvector-solver | 3-bit embedding quantization | CPU |
| HNSW Index | ruvector-hnsw | Fast approximate nearest neighbor | CPU/RAM |
| Exemplar Store | GCS + Firestore | Raw exemplars per cluster | Storage |
| Scheduler | Cloud Scheduler | Periodic crawl ingestion | ~$0.50/mo |

## 6. Compression Stack (Conservative Claims)

### 6.1 Validated Compression: PiQ3 Quantization

PiQ (Pi Quantization) reduces embedding precision while preserving semantic relationships:

```rust
enum PiQLevel {
    PiQ2,  // 2-bit: 16x compression, ~0.92 recall
    PiQ3,  // 3-bit: 10.7x compression, ~0.96 recall (recommended)
    PiQ4,  // 4-bit: 8x compression, ~0.98 recall
}

// Example: 384-dim float32 embedding
// Original: 384 × 4 bytes = 1,536 bytes
// PiQ3: 384 × 3 bits / 8 = 144 bytes
// Compression: 1,536 / 144 = 10.67x
```

**Status**: Implemented in ruvector-solver. Recall validated on MTEB benchmarks.

### 6.2 Validated Compression: Semantic Deduplication

Near-duplicate detection using SimHash:

```rust
// Conservative dedup: cosine > 0.95 threshold
// Reduces near-identical pages (syndicated news, mirror sites)
// Typical reduction: 3-5x on news domains, 1.5-2x on diverse content
```

**Status**: Implemented. Reduction ratio varies heavily by domain.

### 6.3 Indexing (Not Compression): HNSW

HNSW is an indexing structure, not storage compression:

```
HNSW provides:
✓ Fast approximate nearest neighbor search
✓ Sub-linear query time
✗ Storage reduction (adds graph overhead)
```

**Clarification**: HNSW trades memory for speed. It's essential for retrieval but doesn't reduce total storage.

### 6.4 Research Compression: Attractor Distillation

**Hypothesis**: SONA attractors can compress 10,000 clusters → 100 stable attractors (100x).

**Status**: Not validated. This is the "exotic upside" that requires empirical measurement of:
1. Recall@k after compression
2. Nearest neighbor fidelity
3. Downstream task accuracy
4. Temporal reconstruction error
5. Provenance retention quality

### 6.5 Compression Estimates (Conservative vs Aggressive)

| Stage | Conservative | Aggressive (Hypothesis) |
|-------|-------------|-------------------------|
| Text extraction | 15 PB → 4.6 TB | Same |
| PiQ3 quantization | 4.6 TB → 430 GB | Same |
| Semantic dedup | 430 GB → 150 GB (3x) | 430 GB → 43 GB (10x) |
| HNSW + exemplars | 150 GB total | — |
| Attractor distillation | — | 43 GB → 430 MB (100x) |
| Temporal compression | — | 430 MB → 56 MB (8x) |

**Conservative target**: ~150 GB working set (fits in RAM for fast retrieval)
**Aggressive hypothesis**: ~56 MB (requires validation)

## 7. Implementation Phases

### Phase 1: Compressed Web Memory Service (Weeks 1-3)

**Goal**: Queryable vector memory over Common Crawl with validated compression.

**Deliverables**:
- CommonCrawlAdapter with CDX queries and WARC range-GET
- PiQ3 quantization layer
- SimHash deduplication
- HNSW index for retrieval
- Monthly crawl bucket ingestion

**Inputs**:
- Common Crawl WET text
- Embeddings (all-MiniLM-L6-v2)
- Monthly crawl bucket
- Domain metadata

**Outputs**:
- Queryable vector memory
- Semantic cluster IDs
- Prototype exemplars
- Monthly deltas
- Provenance links

**Success Criteria**:
- Retrieval latency < 50ms
- Recall ≥ 90% of uncompressed baseline
- Storage ≥ 5-10x reduction vs naive embedding-only

### Phase 2: Semantic Drift & Fracture Engine (Weeks 4-6)

**Goal**: Detect topic evolution and structural changes.

**Additions**:
- MinCut on cluster graph
- Temporal cluster transition graph
- "Fault line" score
- Alerting for concept bifurcation

**Success Criteria**:
- Detects known topic splits before manual analysts
- Low false positive rate on stable topics

### Phase 3: Shared Memory Brain for Swarms (Weeks 7-10)

**Goal**: Multi-agent coordination via compressed memory.

**Additions**:
- Attractor compression (validate research hypothesis)
- Witness-linked updates
- Per-agent working set cache
- Route by cost/latency/privacy/quality

**Success Criteria**:
- Lower token spend per task
- Fewer repeated retrievals
- Better multi-agent consistency

## 8. Critical Validation Requirements

### 8.1 Acceptance Test

Before claiming aggressive compression ratios:

1. Take 3 monthly Common Crawl slices (CC-MAIN-2026-08, 09, 10)
2. Embed full text (all-MiniLM-L6-v2)
3. Apply PiQ3 quantization
4. Apply semantic deduplication (SimHash)
5. Build HNSW index
6. **Measure**: Recall@10 vs uncompressed baseline on fixed benchmark
7. **Target**: ≥90% recall with ≥10x storage reduction

### 8.2 Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| `recall_at_10` | Retrieval accuracy vs uncompressed | ≥ 0.90 |
| `nn_fidelity` | Nearest neighbor distance preservation | ≥ 0.95 |
| `task_accuracy` | Downstream QA accuracy | ≥ 0.85 |
| `temporal_error` | Reconstruction error across time | ≤ 0.10 |
| `provenance_retention` | % of sources traceable | ≥ 0.99 |

## 9. Failure Modes & Mitigations

### 9.1 Compression Destroys Edge Cases

**Risk**: Exotic compression preserves the average and kills rare-but-valuable content.

**Mitigation**:
- Retain raw exemplar pages per cluster
- Preserve long-tail pockets (high novelty score)
- Measure recall separately for common vs rare concepts

### 9.2 HNSW Complexity

**Risk**: HNSW adds graph structure and tuning complexity without storage reduction.

**Mitigation**:
- Use HNSW for speed, not compression claims
- Tune ef_construction and M parameters empirically
- Consider IVF-PQ for truly massive scale

### 9.3 Temporal Compression Hallucinates Continuity

**Risk**: Merging months into attractors can accidentally erase sharp changes.

**Mitigation**:
- Keep raw monthly witnesses
- Detect and preserve change points
- Flag high-magnitude attractor jumps

### 9.4 Provenance Loss

**Risk**: Aggressive compression without source anchors makes system hard to audit.

**Mitigation**:
- Every cluster retains exemplar citations
- Time buckets preserved
- Cluster lineage graph maintained

## 10. API Endpoints

### 10.1 Discovery Endpoint

```
POST /v1/pipeline/crawl/discover
Authorization: Bearer <token>

{
  "query": "*.arxiv.org/abs/*",
  "crawl": "CC-MAIN-2026-10",
  "limit": 1000,
  "filters": {"language": "en", "min_length": 1000}
}

Response:
{
  "total": 15234,
  "returned": 1000,
  "records": [{"url": "...", "timestamp": "...", "length": 45000}]
}
```

### 10.2 Ingest Endpoint

```
POST /v1/pipeline/crawl/ingest
Authorization: Bearer <token>

{
  "urls": ["https://arxiv.org/abs/2603.12345"],
  "crawl": "CC-MAIN-2026-10",
  "options": {"skip_duplicates": true, "compute_novelty": true}
}

Response:
{
  "ingested": 1,
  "skipped_duplicates": 0,
  "compression_ratio": 10.7,
  "novelty_score": 0.82,
  "cluster_id": "arxiv-quantum-ec"
}
```

### 10.3 Search Endpoint

```
POST /v1/pipeline/crawl/search
Authorization: Bearer <token>

{
  "query": "quantum error correction surface codes",
  "limit": 10,
  "include_exemplars": true
}

Response:
{
  "results": [
    {
      "cluster_id": "arxiv-quantum-ec",
      "score": 0.92,
      "exemplar_url": "https://arxiv.org/abs/2603.12345",
      "observation_count": 1234
    }
  ],
  "latency_ms": 23
}
```

### 10.4 Drift Endpoint

```
GET /v1/pipeline/crawl/drift?topic=machine+learning&months=6

Response:
{
  "topic": "machine learning",
  "drift_score": 0.34,
  "transitions": [
    {"from": "deep-learning", "to": "llm-agents", "month": "2026-01", "magnitude": 0.12}
  ],
  "fault_lines": [
    {"boundary": "symbolic-vs-neural", "stability": 0.23}
  ]
}
```

## 11. Cost Analysis

### 11.1 Conservative Estimate (Validated Compression)

| Component | Monthly Cost |
|-----------|--------------|
| Cloud Memorystore (CDX cache) | $8 |
| GCS storage (150 GB compressed) | $3 |
| Firestore (metadata) | $10 |
| Cloud Run (4 vCPU, 16 GB RAM) | $100 |
| Cloud Scheduler (8 jobs) | $0.50 |
| Egress | $20 |
| **Total** | **~$150/month** |

### 11.2 Aggressive Estimate (If Research Compression Validates)

| Component | Monthly Cost |
|-----------|--------------|
| Cloud Memorystore (CDX cache) | $8 |
| GCS storage (56 MB compressed) | $0.01 |
| Firestore (attractor metadata) | $5 |
| Cloud Run (2 vCPU, 8 GB RAM) | $50 |
| Cloud Scheduler (8 jobs) | $0.50 |
| Egress | $10 |
| **Total** | **~$75/month** |

## 12. Success Metrics

### 12.1 Phase 1 Success (Conservative)

| Metric | Target |
|--------|--------|
| Compression ratio (vs naive embeddings) | ≥ 10x |
| Retrieval latency (p99) | < 50ms |
| Recall@10 | ≥ 0.90 |
| Monthly operating cost | < $200 |

### 12.2 Phase 3 Success (Aggressive)

| Metric | Target |
|--------|--------|
| Compression ratio | ≥ 1000x |
| Retrieval latency (p99) | < 50ms |
| Recall@10 | ≥ 0.90 |
| Monthly operating cost | < $100 |
| Agent token savings | ≥ 30% |

## 13. Open Questions

1. **Attractor validation**: What recall@k does SONA attractor compression actually achieve?
2. **Long-tail preservation**: How do we ensure rare concepts aren't crushed?
3. **Multi-language**: Should attractors be language-specific or cross-lingual?
4. **Real-time**: Can we process new pages before monthly crawl release?
5. **Legal**: What are the implications of derived knowledge vs raw content storage?

## 14. References

- [Common Crawl February 2026 Archive](https://commoncrawl.org/blog/february-2026-crawl-archive-now-available)
- [Common Crawl Graph Statistics](https://commoncrawl.github.io/cc-crawl-statistics/)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [ADR-096: Cloud Pipeline](./ADR-096-cloud-pipeline-realtime-optimization.md)
- [ADR-077: Midstream Platform](./ADR-077-midstream-ruvector-platform.md)

---

## 15. Decision Summary

**What we're building**: A compressed web memory service for agents, not "the whole web in 56 MB."

**Conservative framing**: Turn the open web into a compact, queryable, time-aware semantic memory layer—with enough compression to move from expensive archive analytics to cheap always-on retrieval.

**Exotic framing**: We're not compressing pages. We're compressing the web's evolving conceptual structure.

**Starting point**: Phase 1 with validated compression (10x minimum), then validate research hypotheses for exotic compression.
