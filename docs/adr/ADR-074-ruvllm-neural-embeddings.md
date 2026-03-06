# ADR-074: RuvLLM Neural Embedding Integration

**Status:** Implemented (Phase 2 — RlmEmbedder Active)
**Date:** 2026-03-01 (Phase 1), 2026-03-03 (Phase 2)
**Author:** ml-engineer, platform-eng

## Context

The π.ruv.io shared brain server previously relied on client-side embedding generation (SHA-256 hash or token-averaged hashes) which produced poor-quality embeddings that failed cosine similarity search. A keyword search fallback was added as a stopgap, but vector-native search is essential for scaling beyond trivial corpus sizes.

The ruvllm crate provides a pure-Rust embedding pipeline with three tiers:
1. **HashEmbedder** — FNV-1a hash with character bigrams, L2-normalized (no model required)
2. **RlmEmbedder** — Recursive context-aware embeddings conditioned on a neighbor corpus
3. **Candle sentence transformer** — Neural sentence embeddings (all-MiniLM-L6-v2 or similar)

## Decision

Integrate ruvllm into mcp-brain-server with a phased approach:

### Phase 1 (Implemented): HashEmbedder
- Add `ruvllm = { path = "../ruvllm", default-features = false, features = ["minimal"] }` dependency
- Create `src/embeddings.rs` wrapping `ruvllm::bitnet::rlm_embedder::HashEmbedder`
- Server auto-generates 128-dim L2-normalized embeddings when clients send empty `embedding: []`
- Both storage and search use the same embedding dimension
- No model download, no cold-start penalty, deterministic output

### Phase 2 (Implemented 2026-03-03): RlmEmbedder
- `FlatNeighborStore` populated from all stored memories on startup
- `RlmEmbedder<HashEmbedder, FlatNeighborStore>` active at **50+ corpus documents** (was 1000)
- Storage uses **CorpusConditioned** variant (base=0.7, context=0.25, anti=0.05)
- Search uses **QueryConditioned** variant (base=0.6, context=0.3, anti=0.1)
- **Re-embedding on startup**: When RLM activates, all persisted memories are re-embedded with CorpusConditioned RLM for embedding space consistency (stored embeddings may have been HashEmbedder-generated)
- Graph similarity threshold raised from 0.30 → 0.55 for RLM (contextual gravity makes embeddings more similar)
- Clone derives added upstream to `HashEmbedder` and `FlatNeighborStore`

### Phase 3 (Future): Candle Sentence Transformer
- Enable `candle` feature for ruvllm
- Load all-MiniLM-L6-v2 (~90MB) or gte-small (~30MB) model
- 384-dim sentence embeddings with true semantic understanding
- Trade-off: model download time vs. embedding quality
- Mitigate cold-start with model pre-loading in Cloud Run min-instances

## Architecture

```
Client Request (empty embedding)
       │
       ▼
┌──────────────────────────┐
│  routes.rs: share_memory │
│  ┌────────────────────┐  │
│  │ Auto-embed check:  │  │
│  │ empty or dim≠128?  │──── Yes ──▶ EmbeddingEngine::embed_for_storage()
│  │                    │  │              │
│  └────────────────────┘  │              ▼
│         │ No             │     ruvllm::HashEmbedder::embed()
│         ▼                │     FNV-1a + char bigrams + L2 norm
│  Use client embedding    │              │
│         │                │              ▼
│         ▼                │     128-dim Vec<f32>
│  Verifier::verify_share  │◀─────────────┘
│  (on final embedding)    │
└──────────────────────────┘

Client Search (text query)
       │
       ▼
┌──────────────────────────┐
│ routes.rs: search_memories│
│  ┌────────────────────┐  │
│  │ Has text query q?  │──── Yes ──▶ EmbeddingEngine::embed()
│  │                    │  │              │
│  └────────────────────┘  │              ▼
│         │ No             │     Same HashEmbedder pipeline
│         ▼                │              │
│  Return empty            │              ▼
│                          │     cosine_similarity(query_emb, stored_emb)
│                          │     → reputation-weighted ranking
└──────────────────────────┘
```

## Key Design Decisions

1. **Server-side embedding**: Clients send empty `embedding: []` and the server generates. This ensures:
   - Consistent dimension (128) across all memories
   - No client-side embedding logic needed
   - Future backend upgrades transparent to clients
   - Backward compatible: clients can still send pre-computed embeddings

2. **`minimal` feature**: Avoids pulling in candle-core (~50 crates). HashEmbedder is pure Rust with zero external dependencies.

3. **128-dim**: Matches existing SONA engine, cognitive engine, and ranking engine dimensions. Lower than typical sentence transformer (384) but sufficient for hash-based embeddings.

4. **Embedding verification after auto-generation**: The share handler generates embeddings before calling `Verifier::verify_share()`, so the verification validates the server-generated embedding (not an empty array).

5. **Corpus tracking**: `EmbeddingEngine::add_to_corpus()` tracks corpus size for future RlmEmbedder integration. Status endpoint reports `embedding_corpus` count.

## Dependencies Added

```toml
ruvllm = { path = "../ruvllm", default-features = false, features = ["minimal"] }
```

Transitive: `ruvector-core`, `ruvector-sona` (already in tree)

## Files Changed

| File | Change |
|------|--------|
| `crates/mcp-brain-server/Cargo.toml` | Added ruvllm dependency |
| `crates/mcp-brain-server/src/embeddings.rs` | New: EmbeddingEngine wrapping HashEmbedder |
| `crates/mcp-brain-server/src/lib.rs` | Added `pub mod embeddings` |
| `crates/mcp-brain-server/src/types.rs` | Added `embedding_engine` to AppState and StatusResponse |
| `crates/mcp-brain-server/src/routes.rs` | Auto-embed in share, embed-based search, status fields |

## Consequences

### Positive
- Vector similarity search works with consistent 128-dim embeddings
- No model download or external service required
- Deterministic: same text always produces same embedding
- Zero cold-start penalty (HashEmbedder is <1ms)
- Clients simplified: no embedding logic needed

### Negative
- RLM contextual gravity reduces discriminative power on homogeneous corpora — keyword matching must remain dominant signal
- 128-dim is lower fidelity than 384-dim sentence transformers
- Re-embedding on startup adds ~2-3s to cold start with 237 memories
- FNV-1a hash collisions possible for very similar token patterns (base embedder)

### Neutral
- Keyword search still primary ranking signal (keyword floor +1.0 always outranks embedding-only)
- Future upgrade to candle sentence transformer is backward-compatible (same dimension)

## Metrics (Phase 1 Deployment — 2026-03-01)

- **Seeded:** 37 memories, 19 contributors
- **Search hit rate:** 10/10 queries return results
- **Graph:** 37 nodes, 200 edges
- **Clusters:** 7 (category-based partition)
- **Avg quality:** 0.838
- **Embedding corpus:** 37 entries
- **Build size:** No significant increase (ruvllm minimal is pure Rust)

## Metrics (Phase 2 Deployment — 2026-03-03)

- **Memories:** 237, **Contributors:** 17
- **Embedding engine:** `ruvllm::RlmEmbedder` (context-aware, activated at 50+ docs)
- **Search P@1:** 100% (30/30 benchmark queries)
- **Search P@3:** 100% (30/30)
- **Graph:** 237 nodes, 827 edges (threshold 0.55)
- **Clusters:** 20 (meaningful MinCut partitions)
- **Avg quality:** 0.73
- **Votes:** 608
- **LoRA epoch:** 2

### Search Intelligence Stack (Phase 2)

| Layer | Signal | Weight (keyword path) | Weight (no keyword) |
|-------|--------|----------------------|---------------------|
| Keyword matching | Word-boundary title/tag/category/content | 0.85 × boost + 1.0 floor | — |
| RLM embedding similarity | QueryConditioned cosine | 0.05 | 0.45 |
| Graph PPR (ForwardPushSolver) | PageRank over knowledge graph | 0.04 | 0.25 |
| Vote quality (Bayesian Beta) | Learning-to-rank from 608 votes | 0.03 | 0.15 |
| Reputation | Multi-factor contributor trust | 0.03 | 0.15 |
| Query expansion | 32 synonym rules (abbreviations) | implicit | implicit |
| Attention ranking | TopologyGatedAttention post-processing | post-score | post-score |
