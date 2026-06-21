# ADR-252: Coherence-Weighted Agent Memory Compaction

**Status**: Proposed  
**Date**: 2026-06-14  
**Author**: ruvnet / claude-flow nightly  
**Crate**: `crates/ruvector-agent-memory`  
**Branch**: `research/nightly/2026-06-14-agent-memory-compaction`

---

## Context

RuVector positions itself as a *Rust-native cognition substrate for agents*.  As
agents run continuously they accumulate memory embeddings at a rate that exceeds
the capacity of efficient search.  Without a principled compaction strategy:

1. Brute-force search latency grows as O(n · d) per query.
2. Stale memories crowd out relevant neighbors, reducing Recall@K.
3. Edge deployments (Cognitum Seed, Pi Zero) run out of SRAM.

Existing memory management in other systems relies on token budgets (MemGPT),
LLM-rated importance scores (Generative Agents, Park et al. 2023), or explicit
DELETE calls (Mem0).  None provide a continuous, vector-native, LLM-free
importance score that incorporates *semantic coherence with the current agent
context window*.

The 2026 survey "From Storage to Experience" (arXiv:2605.06716) explicitly
confirms "adaptive pruning of working memory" as an open research gap.

---

## Decision

We add `crates/ruvector-agent-memory` to the workspace, implementing three
compaction policies and establishing the **CoherencePolicy** as the recommended
default:

```
I(m) = α·recency(m) + β·frequency(m) + γ·coherence(m, context_window)
```

with defaults `α=0.25, β=0.35, γ=0.40`.

`coherence(m, context_window)` is the maximum cosine similarity between `m.vector`
and any embedding in the rolling context window — i.e., the agent's recent queries.

The policy is implemented as a `CompactionPolicy` trait, allowing the default to
be swapped without changing call sites.

---

## Consequences

### Positive

- **Recall improvement**: CoherencePolicy achieves +29.0pp recall@10 over LRU
  and +13.4pp over LFU at 50% compaction on the benchmark dataset.
- **LLM-free**: No LLM call required; scoring is O(n·W·d) arithmetic where W
  is the context window size (typically 20).
- **Zero dependencies**: The library crate has no external deps, enabling WASM
  and embedded deployment.
- **Auditable**: Compaction decisions are deterministic and can be logged to the
  `ruvector-verified` witness chain.
- **Composable**: The `CompactionPolicy` trait allows custom policies without
  modifying core code.

### Negative / Trade-offs

- **CoW compaction latency**: 3,123 µs for 2,000 × 64-dim entries (vs 127 µs
  for LFU).  This is acceptable for background compaction but not for
  on-query-path usage.
- **Context-monopolisation risk**: An agent fixated on one topic will retain
  only memories from that topic.  Future work should add a cluster-diversity
  constraint.
- **Cold-start gap**: When context_window is empty (first N turns), CoherencePolicy
  degrades to frequency-only scoring (γ term drops to 0.0).

---

## Alternatives Considered

### A: LRU only

Simple, low-overhead (127 µs).  Benchmark shows 71.0% recall — unacceptable for
agents where missing 29% of true neighbors leads to wrong responses.  Rejected
as default.

### B: LFU only

Better than LRU (86.6% recall).  Simple to implement.  But does not exploit
semantic alignment with the current reasoning context.  LFU is kept as a
built-in fallback for cold-start scenarios.

### C: Ebbinghaus decay (MemoryBank style)

Would require tracking per-entry decay curves and time deltas.  Adds floating-
point state per entry with no clear benefit over CoherencePolicy in high-access-
rate agent scenarios where the frequency signal is already strong.  Deferred to
future work; could be added as `EbbinghausPolicy`.

### D: LLM-rated importance (Generative Agents style)

Requires an LLM call at write time; prohibitively expensive for high-throughput
agents (e.g., coding agents with 100+ turns/minute).  Introduces a prompt
injection surface.  Rejected.

### E: Graph-cut coherence (ruvector-mincut)

Using `ruvector-mincut` to score memories by their centrality in the retrieval
graph would be stronger but requires a live graph index.  This ADR establishes
the flat compaction primitive; graph-coherence is the natural next step (future
ADR).

---

## Implementation Plan

1. ✅ `crates/ruvector-agent-memory` added to workspace.
2. ✅ `MemoryEntry`, `MemoryStore`, `CompactionPolicy` implemented.
3. ✅ `LruPolicy`, `LfuPolicy`, `CoherencePolicy` implemented.
4. ✅ 11 unit tests + 1 acceptance test pass.
5. ✅ Benchmark binary produces real measured results.
6. [ ] Add `feature = "hnsw"` gate wrapping `MemoryStore` over HNSW index.
7. [ ] Add `feature = "mcp"` MCP tool handler in `crates/mcp-gate`.
8. [ ] Add `feature = "rvf"` RVF snapshot serialisation.
9. [ ] Add online coherence tracking (incremental update per turn).
10. [ ] Evaluate on real agent conversation logs.

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-agent-memory` on:
- **Hardware**: Intel Celeron N4020, x86-64
- **OS**: Linux 6.18.5
- **Rust**: rustc 1.94.1 (release)

| Policy | Recall@10 (after 50% compaction) | Compaction latency | vs LRU |
|--------|----------------------------------|-------------------|--------|
| LRU | 71.0% | 210 µs | — |
| LFU | 86.6% | 127 µs | +15.6 pp |
| CoherenceWeighted | 100.0% | 3,123 µs | +29.0 pp |

Dataset: 2,000 vectors, D=64, 20 clusters, 5 hot, 50 test queries, seed=42.

**Acceptance**: CoW recall > LRU + 2pp → **PASS** (actual delta: +29.0pp).

---

## Failure Modes

| Mode | Condition | Mitigation |
|------|-----------|-----------|
| Recall collapse on cold start | Context window empty | Fall back to LFU |
| Context monopolisation | Agent fixated on one topic | Future: cluster-diversity constraint |
| Compaction latency on hot path | Called synchronously per turn | Move to background task; trigger async |
| Float instability | Very long sessions with large access counts | Saturating cast to f64 for frequency ratio |

---

## Security Considerations

- No LLM calls: zero prompt injection surface in compaction path.
- Compaction is deterministic: given identical inputs, identical output.
- Compaction events SHOULD be logged to `ruvector-verified` witness chain for
  audit trails in safety-critical agent deployments.
- The crate MUST NOT store raw text content; only embeddings and metadata.

---

## Migration Path

`ruvector-agent-memory` is a new crate.  No existing code is modified.  To adopt:

1. Replace raw `Vec<Vec<f32>>` memory buffers with `MemoryStore::new(dims)`.
2. Call `compact(store, &CoherencePolicy::default(), target, ctx)` when
   `store.len() >= capacity`.
3. Pass the last N query embeddings as the `context_window`.

Existing users of `ruvector-delta-index` are unaffected; that crate handles
incremental updates to the HNSW graph, while this crate handles coarse-grained
eviction at the application layer.

---

## Open Questions

1. **Optimal weights**: Are `α=0.25, β=0.35, γ=0.40` the best defaults across
   agent workload types?  A self-tuning variant should be explored.
2. **Online coherence**: Can we maintain coherence scores incrementally rather
   than recomputing at compaction time?
3. **Real corpus validation**: How does recall differ on real agent memory
   (vs synthetic Gaussian clusters)?
4. **Cluster diversity**: Should the policy guarantee ≥1 survivor per cluster?
5. **Graph extension**: Can `coherence(m)` be replaced by graph-centrality scores
   from `ruvector-mincut` for graph-RAG use cases?
