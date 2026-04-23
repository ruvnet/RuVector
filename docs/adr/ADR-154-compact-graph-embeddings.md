# ADR-154: Compact Graph Embeddings — Anchor-Based Node Tokenization

**Status:** Accepted  
**Date:** 2026-04-23  
**Deciders:** ruvector core team  
**Relates to:** ADR-003 (HNSW indexing), ADR-006 (Unified Memory Service)

---

## Context

The ruvector graph-embedding stack currently stores one f32 vector of dimension D per
node, occupying N × D × 4 bytes.  For a production knowledge graph with N = 10M nodes
and D = 256, this is **10 GB** — exceeding the memory budget of many inference hosts.

Existing mitigations (DiskANN, HNSW sharding) address *search* latency but not the
*storage* cost of the embedding table itself.  We need a representation that:

- Reduces per-node embedding RAM by ≥ 10× without retraining the downstream model.
- Keeps single-node reconstruct latency below 10 μs at p50 (hot cache, release build).
- Degrades cosine recall by < 3% compared with the dense f32 baseline.
- Supports INT8 quantization for a further 4× table-size reduction.
- Remains inductive: new nodes can be embedded without table retraining.

The NodePiece paper (Galkin et al., ICLR 2022, arXiv:2106.12144) demonstrated that
representing nodes as compositions of a small anchor vocabulary achieves 10× parameter
reduction with < 2% quality drop on knowledge-graph link prediction benchmarks.

---

## Decision

We introduce a new crate, **`crates/compact-graph-embed`**, implementing anchor-based
compact node tokenization:

### Architecture

```
Node v → [tokens: Vec<(anchor_u8, dist_u8)>]
        ↓  embed()
        mean( token_table[anchor, dist]  for each token )
```

### Components

| Component | Type | Description |
|-----------|------|-------------|
| `AnchorTokenizer` | `impl NodeTokenizer` | BFS from k high-degree anchors; stores flat `Vec<(u8,u8)>` + `Vec<u32>` offsets |
| `EmbeddingTableF32` | `impl TokenStorage` | k × max_dist × dim f32 entries |
| `EmbeddingTableI8` | `impl TokenStorage` | Row-quantized INT8 + per-row f32 scale |
| `MeanEmbedder<T, S>` | `impl NodeEmbedder` | Mean-pools token embeddings; specialised zero-alloc path for `AnchorTokenizer` |

### Anchor selection

1. Sort nodes by degree descending.
2. Break ties with a seeded Fisher-Yates shuffle (reproducible).
3. Take top-k.

### Token storage layout

Flat packed `Vec<(u8, u8)>` with CSR-style `Vec<u32>` offsets.  Two bytes per token
(vs 8 bytes for `Token { anchor_idx: u32, dist: u8 }` with padding).

### Hot path

`MeanEmbedder<AnchorTokenizer, S>::embed_fast()` reads `tokens_packed()` directly
and calls `TokenStorage::accumulate_token(&mut [f32])` — one heap allocation per call
(the output Vec<f32>), no intermediate allocations.

---

## Consequences

### Positive

- **14.2× RAM compression** on N=50K, D=128 graph (25.6 MB → 1.8 MB; measured).
- **Zero cosine-recall drop** with INT8 row-quantization (cosine is invariant to
  row-wise scale; measured 1.0000 on 1000-node sample).
- **1.96 μs p50** reconstruct latency (release build, optimized+debuginfo profile;
  p99 = 4.34 μs).
- Inductive: unseen nodes can be tokenized at inference time given the anchor BFS
  distances; no embedding table retraining required.
- Trait-based (`NodeTokenizer`, `TokenStorage`, `NodeEmbedder`) — any component can
  be swapped (e.g., trained vs random init, mmap vs in-memory, f32 vs i8).

### Negative / Trade-offs

- **Approximation**: token-mean pooling loses per-node uniqueness; nodes at the same
  BFS distances from all anchors receive identical embeddings.  Impact grows as k↓ or
  max_dist↓.
- **BFS pre-computation cost**: O(k × (N + E)) one-time cost at graph load.  For
  k=16 and E=500K this is sub-second; for billion-edge graphs it requires parallelism
  (not yet implemented).
- **Disconnected graph**: nodes in isolated components receive all-fallback tokens.
  Embeddings may collide; downstream quality depends on how the model handles this.
- **k ≤ 255**: anchor_idx stored as u8; larger k requires a breaking API change.
- **Random init** (PoC): production use requires self-supervised pretraining of the
  token embedding table for meaningful representations.

---

## Alternatives Considered

### A. Dense embedding + DiskANN offload

Keep the full f32 table but page infrequently-accessed rows from NVMe.  Pros: zero
quality loss.  Cons: p50 latency dominated by disk I/O (10–100 ms); not suitable for
real-time inference.

### B. Product Quantization (FAISS PQ)

Compress each embedding vector into M sub-codebook indices.  Pros: tunable compression
ratio; battle-tested (FAISS).  Cons: requires training; asymmetric distance calculation
adds complexity; not inductive without re-encoding.

### C. Row-quantized INT8 of the full table

Apply INT8 quantization to the N × D table directly (same approach as EmbeddingTableI8
but applied to the full per-node table).  Pros: 4× RAM reduction with 0% cosine loss.
Cons: only 4× vs 14× for anchor tokenization; still requires O(N × D) storage.

### D. Node clustering (centroid lookup)

Cluster nodes into C clusters; store one embedding per centroid.  Pros: O(C × D)
storage.  Cons: requires k-means training; quality loss proportional to cluster purity;
not inductive for new nodes not assigned to a cluster.

**Decision**: adopt anchor tokenization (this ADR) as the primary PoC.  Alternatives A
and C can be layered on top (DiskANN for graph traversal, INT8 for table reduction).
Alternative B is planned for a subsequent ADR when production training infra is ready.

---

## References

- Galkin et al. NodePiece (ICLR 2022) — arXiv:2106.12144  
- Wang et al. GQT (ICLR 2025) — arXiv:2410.13798  
- Jégou et al. Product Quantization (TPAMI 2011) — DOI:10.1109/TPAMI.2010.57  
- Implementation: `crates/compact-graph-embed/`  
- Research doc: `docs/research/nightly/2026-04-23-compact-graph-embeddings/README.md`
