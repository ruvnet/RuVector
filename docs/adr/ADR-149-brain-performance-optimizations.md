# ADR-149: Brain Performance Optimizations — SIMD Search, Batch Graph, Incremental LoRA, Quality Gating

## Status

Accepted

## Date

2026-04-13

## Context

The pi.ruv.io brain (10,290 memories, 38M graph edges) performs well at current scale but has four optimization opportunities that compound:

| Current Bottleneck | Observed | Root Cause |
|---|---|---|
| Search | ~0.5ms per query | Scalar cosine similarity over all memories |
| Graph rebuild | ~5-10 min on cold start | Rebuilds all 38M edges from scratch |
| LoRA training | Full corpus every cycle | Retrains on all 10K memories even when only 5 are new |
| Search noise | Returns debug/noise entries | No quality filtering in the search path |

DiskANN was evaluated (ADR-148 P4) but is not cost-effective at 10K memories — brute-force is 10x faster. The crossover is ~100K memories (~6 months at accelerated ingestion). These four optimizations deliver immediate value at current scale.

## Decision

Implement four optimizations in priority order. Each is independent and can ship separately.

### P1: SIMD Cosine in Search (2-3x speedup, 1 hour)

**Problem:** `crates/mcp-brain-server/src/graph.rs` uses a scalar cosine similarity:
```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

**Fix:** Replace with `ruvector-core`'s SIMD-accelerated version which already has NEON (Apple Silicon), AVX2, and AVX-512 implementations:

```rust
// In graph.rs or store.rs:
use ruvector_core::simd_intrinsics::cosine_similarity_simd;

// In search_memories():
let sim = cosine_similarity_simd(&query_embedding, &m.embedding);
```

The `ruvector-core` SIMD cosine at 128 dimensions processes 4 floats/cycle (NEON) or 8 floats/cycle (AVX2), with 4x loop unrolling. Expected speedup: **2-3x on x86 (Cloud Run), 2x on ARM**.

**Dependency:** Add `ruvector-core` to `mcp-brain-server/Cargo.toml` (it's already in the workspace but not a direct dependency of the brain server).

**Risk:** None. Same math, faster execution. The SIMD functions have 1,486 lines of tests in `simd_intrinsics.rs`.

### P2: Quality-Gated Search (immediate relevance improvement, 30 minutes)

**Problem:** Search returns all memories regardless of quality. With ~40% of memories being noise (`solution` category scraped from random websites), the top-k results are polluted.

**Fix:** Add quality floor to the search path:

```rust
// In search_memories():
let quality_floor = 0.05; // Skip bottom-tier noise
for m in memories {
    if m.quality_score.mean() < quality_floor {
        continue; // Skip known noise
    }
    let sim = cosine_similarity_simd(&query_embedding, &m.embedding);
    // ...
}
```

Also add optional `min_quality` parameter to the search API:
```
GET /v1/memories/search?q=seizure+detection&limit=10&min_quality=0.1
```

**Side effect:** Reduces the number of memories to scan, giving an additional 30-40% search speedup (skip ~4K noise memories).

**Risk:** Very low. Quality floor defaults to 0.05 (only skip the absolute bottom). The API parameter is optional — existing calls are unaffected.

### P3: Batch Graph Operations (10x faster rebuild, 1 day)

**Problem:** The 38M-edge graph rebuilds by inserting edges one at a time after Firestore hydration. Each insertion triggers adjacency list updates.

**Fix:** Batch the graph construction:

```rust
// Current (slow): 
for memory in memories {
    for neighbor in find_neighbors(memory) {
        graph.add_edge(memory.id, neighbor.id, similarity);
    }
}

// Optimized (fast):
// 1. Compute all similarities in a single pass (SIMD-accelerated)
// 2. Sort edges by source node
// 3. Build adjacency lists in one allocation
let mut edges: Vec<(NodeId, NodeId, f32)> = Vec::with_capacity(estimated_edges);
for (i, m1) in memories.iter().enumerate() {
    for m2 in &memories[i+1..] {
        let sim = cosine_similarity_simd(&m1.embedding, &m2.embedding);
        if sim > threshold {
            edges.push((m1.id, m2.id, sim));
        }
    }
}
edges.sort_by_key(|e| e.0);
graph.build_from_sorted_edges(&edges);
```

**Additional optimization:** Use `rayon` parallel iterator for the similarity computation:
```rust
let edges: Vec<_> = (0..n).into_par_iter()
    .flat_map(|i| {
        (i+1..n).filter_map(move |j| {
            let sim = cosine_similarity_simd(&memories[i].embedding, &memories[j].embedding);
            if sim > threshold { Some((i, j, sim)) } else { None }
        })
    }).collect();
```

At 10K memories × 128 dims with SIMD, this processes ~50M similarity computations in ~5 seconds on 4 cores (vs minutes for sequential single-edge insertion).

**Risk:** Low. Graph is rebuilt from scratch — no incremental state to corrupt. If the batch builder fails, fall back to the existing sequential builder.

### P4: Incremental LoRA (5x faster training, 1 week)

**Problem:** Every 5-minute training cycle processes ALL 10K+ memories to extract propositions and compute LoRA updates. Most memories haven't changed — only the last ~5 are new.

**Fix:** Track a `last_trained_at` watermark and only process new memories:

```rust
struct IncrementalTrainer {
    last_trained_at: chrono::DateTime<chrono::Utc>,
    cumulative_lora: LoraWeights,
}

impl IncrementalTrainer {
    fn train_cycle(&mut self, store: &MemoryStore) {
        // Only process memories created since last training
        let new_memories: Vec<_> = store.all_memories()
            .into_iter()
            .filter(|m| m.created_at > self.last_trained_at)
            .collect();
        
        if new_memories.is_empty() {
            return; // Nothing new — skip entirely
        }
        
        // Extract propositions from new memories only
        let new_propositions = extract_propositions(&new_memories);
        
        // Compute incremental LoRA update
        let delta_lora = compute_lora_delta(&new_propositions, &self.cumulative_lora);
        
        // Apply with EWC to prevent catastrophic forgetting
        self.cumulative_lora = ewc_merge(&self.cumulative_lora, &delta_lora);
        
        self.last_trained_at = chrono::Utc::now();
    }
}
```

**Savings:**
- Current: process 10K memories every 5 min = 2M memories/day
- Incremental: process ~50 new memories per cycle = 14K memories/day
- **143x reduction in computation**

**Risk:** Medium. Incremental training may drift from what a full retrain would produce. Mitigate by running a full retrain every 24 hours (the existing nightly job) and using the incremental updates only between full retrains. EWC (already implemented) prevents catastrophic forgetting.

## Implementation Order

```
Week 1:
  Day 1: P1 (SIMD cosine) — wire ruvector-core into brain server
  Day 1: P2 (quality gate) — add quality floor to search path
  Day 2-3: P3 (batch graph) — parallel batch graph builder

Week 2-3:
  P4 (incremental LoRA) — watermark tracking + delta training
```

P1 and P2 can ship in the same deploy. P3 ships independently. P4 is a separate PR.

## Expected Impact

| Optimization | Before | After | Speedup |
|---|---|---|---|
| **P1: SIMD search** | 0.5ms/query | **0.2ms/query** | **2.5x** |
| **P2: Quality gate** | Scan 10.3K memories | **Scan ~6K memories** | **1.7x** |
| **P1+P2 combined** | 0.5ms/query | **~0.1ms/query** | **5x** |
| **P3: Graph rebuild** | 5-10 min cold start | **~30 seconds** | **10-20x** |
| **P4: LoRA training** | 10K memories/cycle | **~50 memories/cycle** | **143x** |

Combined search improvement: **5x faster with cleaner results.**
Combined startup improvement: **10-20x faster cold boot.**
Combined training improvement: **143x less computation per cycle.**

## Consequences

### Positive
- Search drops from 0.5ms to ~0.1ms (5x) — enables real-time use cases
- Cold start drops from 5-10 min to ~30 seconds — faster deploys, less downtime
- Training cycles go from processing 10K memories to ~50 — 143x less compute, lower Cloud Run costs
- Search quality improves immediately from quality gating (skip noise)
- All optimizations are backward-compatible — no API changes, no data migration

### Negative
- `ruvector-core` added as brain server dependency (increases binary size ~2MB)
- Quality gate may hide memories that become relevant later (mitigated by low 0.05 threshold)
- Incremental LoRA may diverge from full retrain (mitigated by nightly full retrain + EWC)

### Not Addressed
- DiskANN integration deferred to 100K+ memories (ADR-148 P4)
- Embedding dimension upgrade (128 → 384) deferred to next major version
- Multi-region replication not in scope

## References

- ADR-148: Brain Hypothesis Engine (DiskANN at 100K+)
- ADR-146: DiskANN Vamana Implementation
- `crates/ruvector-core/src/simd_intrinsics.rs` — NEON/AVX2/AVX-512 cosine at lines 42-904
- `crates/mcp-brain-server/src/graph.rs` — current scalar cosine
- `crates/mcp-brain-server/src/store.rs:580-615` — current search implementation
