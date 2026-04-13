# ADR-148: Brain Hypothesis Engine — Self-Improving Knowledge System with Gemini, DiskANN, and Auto-Experimentation

## Status

Proposed

## Date

2026-04-13

## Context

The pi.ruv.io brain (10,300+ memories, 38M graph edges, LoRA epoch 41) stores and retrieves knowledge but cannot:
1. Generate hypotheses from cross-domain connections
2. Evaluate quality beyond embedding similarity (quality scores mostly 0.0)
3. Filter noise from curated knowledge (random IEEE events alongside real patterns)
4. Measure whether LoRA training actually improves search quality

The brain runs on Google Cloud Run (`ruvbrain` service, us-central1) backed by `crates/mcp-brain-server/` (Rust/Axum). Current embedding: `ruvllm::RlmEmbedder` at 128-dim. Current index: flat HNSW.

## Decision

Add four capabilities as **additive layers** — no changes to the running brain's core path. All new code is behind feature flags or in separate Cloud Run services.

### Architecture: Three New Components

```
┌─────────────────────────────────────────────────────────┐
│  EXISTING (untouched)                                    │
│  mcp-brain-server: store, search, graph, drift, LoRA    │
│  Embedder: ruvllm::RlmEmbedder (128-dim)                │
│  Index: flat HNSW                                        │
└──────────────┬──────────────────────────────────────────┘
               │ (reads from, writes back to)
               v
┌─────────────────────────────────────────────────────────┐
│  NEW: Hypothesis Engine (separate Cloud Run service)     │
│                                                          │
│  1. HYPOTHESIS GENERATOR                                 │
│     - Watches for new cross-domain graph edges           │
│     - Templates: "If X works in domain A,                │
│       then X should work in domain B"                    │
│     - Uses Gemini 2.5 Flash for hypothesis formulation   │
│       and experiment design                              │
│     - Stores hypotheses as "untested" memories           │
│                                                          │
│  2. QUALITY SCORER                                       │
│     - DiskANN index over all 10K+ memory embeddings      │
│     - PageRank via ruvector-solver ForwardPush            │
│     - Multi-signal: centrality + citations + verdicts     │
│       + contributor rep + temporal + surprise             │
│     - Updates quality field via brain API                 │
│                                                          │
│  3. NOISE FILTER                                         │
│     - Ingestion gate: regex + embedding dedup             │
│     - Weekly cleanup: archive orphan low-quality          │
│     - Meta-mincut: ruvector-mincut on knowledge graph     │
│       to find noise partition                             │
│                                                          │
│  4. BENCHMARK SUITE                                      │
│     - 50 curated test queries with known-good answers     │
│     - Runs before/after each LoRA epoch                   │
│     - Tracks MRR, precision@5, cross-domain recall        │
│     - Auto-rollback if MRR drops > 5%                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Component Details

#### Gemini 2.5 Flash for Hypothesis Generation

**Why Gemini, not local LLM:**
- Hypothesis generation is infrequent (triggered by new cross-domain edges, ~10/day)
- Requires reasoning about domain transfer ("if mincut detects seizures, could it detect X?")
- Gemini 2.5 Flash: fast, cheap (~$0.15/1M input tokens), 1M context window
- Local RLM embedder stays for indexing (it's tuned to the corpus) — Gemini is for reasoning only

**API integration:**
```rust
// New module: crates/mcp-brain-server/src/hypothesis.rs
// Feature-gated: #[cfg(feature = "hypothesis")]

use google_generativeai::Client; // or raw REST via reqwest

async fn generate_hypothesis(edge: &CrossDomainEdge) -> Hypothesis {
    let prompt = format!(
        "Given this cross-domain connection:\n\
         Domain A: {}\nDomain B: {}\nBridge concept: {}\n\n\
         Generate a testable hypothesis: if the pattern from domain A \
         works, what specific prediction does it make in domain B? \
         Include: hypothesis statement, test method, expected outcome, \
         null hypothesis, required data.",
        edge.domain_a, edge.domain_b, edge.bridge_concept
    );
    // Call Gemini 2.5 Flash
    let response = gemini_client.generate(&prompt).await?;
    parse_hypothesis(response)
}
```

**Cost estimate:** ~10 hypotheses/day × ~500 tokens each = ~5K tokens/day = ~$0.001/day. Negligible.

#### DiskANN for Scalable Quality Scoring

**Why DiskANN, not current flat HNSW:**
- Current HNSW is in-memory, fine for 10K memories
- At 100K+ memories (projected within months), memory pressure becomes real
- DiskANN stores the graph on SSD, loads only neighbors on demand
- Product Quantization (PQ) compresses vectors 4-8x for candidate filtering
- `ruvector-diskann` already implements Vamana graph + PQ (ADR-146)

**Integration plan:**
```rust
// New module: crates/mcp-brain-server/src/diskann_index.rs
// Feature-gated: #[cfg(feature = "diskann")]

use ruvector_diskann::{DiskAnnIndex, DiskAnnConfig};

pub struct HybridIndex {
    hnsw: HnswIndex,      // Existing, stays as primary for <50K
    diskann: DiskAnnIndex, // New, activates at >50K memories
    threshold: usize,      // Switch point (default: 50_000)
}

impl HybridIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.hnsw.len() < self.threshold {
            self.hnsw.search(query, k)
        } else {
            self.diskann.search(query, k)
        }
    }
}
```

**Benchmark plan:** Run both HNSW and DiskANN on the current 10K corpus, measure:
- Recall@10 (should be >95% for both)
- Query latency (HNSW: ~1ms, DiskANN: ~5-10ms expected)
- Memory usage (HNSW: ~50MB, DiskANN: ~5MB + SSD)
- Index build time

#### Quality Scorer with ForwardPush PageRank

```rust
// crates/mcp-brain-server/src/quality.rs

pub fn compute_quality_scores(brain: &Brain) -> Vec<(MemoryId, f64)> {
    // 1. Build CSR graph from memory edges
    let graph = brain.graph_to_csr();
    
    // 2. Run ForwardPush PageRank (sublinear, O(1/epsilon))
    let pr = ForwardPushSolver::new(0.85, 0.001);
    let pagerank = pr.solve(&graph)?;
    
    // 3. Compute multi-signal quality
    brain.memories().map(|m| {
        let centrality = pagerank[m.id];
        let citations = m.inbound_edge_count as f64 / max_citations;
        let verdict = match m.verdict {
            Confirmed => 1.0,
            Refuted => -0.5,
            Untested => 0.0,
        };
        let surprise = 1.0 - m.max_similarity_to_existing;
        let temporal = recency_weight(m.created_at);
        let bridge = if m.crosses_domains { 0.3 } else { 0.0 };
        
        let quality = 0.25 * centrality
                    + 0.20 * citations
                    + 0.20 * verdict
                    + 0.15 * surprise
                    + 0.10 * temporal
                    + 0.10 * bridge;
        
        (m.id, quality.clamp(0.0, 1.0))
    }).collect()
}
```

### Safety Constraints (don't break the running system)

1. **All new code is feature-gated.** The existing `mcp-brain-server` binary is unchanged unless `--features hypothesis,diskann,benchmark` is explicitly enabled.

2. **Hypothesis engine runs as a SEPARATE Cloud Run service.** It calls the brain's API; it doesn't modify the brain's process. If it crashes, the brain keeps running.

3. **DiskANN is a fallback, not a replacement.** HNSW stays as primary for <50K memories. DiskANN only activates when memory count exceeds the threshold. Both can be queried in parallel for benchmark comparison.

4. **Quality scores are written to a NEW field (`quality_v2`).** The existing `quality` field is untouched until v2 scores are validated.

5. **Noise filtering is archive-only.** Memories are archived (moved to cold storage), never deleted. Full rollback possible.

6. **Benchmark auto-rollback.** If LoRA epoch N+1 degrades MRR by >5%, the epoch is discarded and the EWC checkpoint is restored automatically.

7. **Gemini API key stored in gcloud secrets.** Already available as `GEMINI_API_KEY`. Rate-limited to 10 calls/hour to avoid cost surprises.

### Implementation Phases

| Phase | What | Risk | Timeline |
|-------|------|------|----------|
| **P0: ADR + Branch** | This document + feature branch | None | Done |
| **P1: Benchmark suite** | 50 test queries, MRR tracking | None (read-only) | 3 days |
| **P2: Quality scorer** | PageRank + multi-signal scoring | Low (writes to new field) | 1 week |
| **P3: Noise filter** | Ingestion gate + weekly cleanup | Low (archive-only) | 3 days |
| **P4: DiskANN integration** | Hybrid index behind feature flag | Low (fallback only) | 1 week |
| **P5: Hypothesis engine** | Gemini integration + auto-test | Medium (new service) | 2 weeks |

**Total: ~5 weeks, phased. P1-P3 can run in parallel.**

## Consequences

### Positive
- Brain evolves from "smart database" to "scientific reasoner"
- Quality scores become meaningful (currently all 0.0)
- Noise filtering reduces graph pollution
- LoRA training becomes measurable and rollback-safe
- DiskANN prepares for 100K+ memory scale
- Gemini hypothesis generation is the first step toward autonomous discovery

### Negative
- New dependency: Google Gemini API (adds cost, ~$0.03/day estimated)
- DiskANN adds complexity to the index path
- Hypothesis engine needs curation — false hypotheses could pollute if not filtered
- More Cloud Run services to monitor

### Risks
- Gemini may generate low-quality hypotheses → mitigated by verdict system (untested until confirmed)
- DiskANN recall may be lower than HNSW at small corpus → mitigated by hybrid approach with threshold
- Quality scoring may be gamed by circular citations → mitigated by PageRank dampening

## References

- ADR-146: DiskANN Vamana Implementation
- ADR-131: Consciousness Metrics Crate
- ADR-048: Sublinear Graph Attention
- Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search" (NeurIPS 2019)
- Google Gemini API: https://ai.google.dev/gemini-api
- ForwardPush PPR: Andersen, Chung, Lang 2006
