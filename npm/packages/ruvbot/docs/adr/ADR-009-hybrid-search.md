# ADR-009: Hybrid Search Architecture

## Status
Accepted

## Date
2026-01-27

## Context

Clawdbot uses basic vector search with external embedding APIs. RuvBot improves on this with:
- Local WASM embeddings (75x faster)
- HNSW indexing (150x-12,500x faster)
- Need for hybrid search combining vector + keyword (BM25)

## Decision

### Hybrid Search Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RuvBot Hybrid Search                          │
├─────────────────────────────────────────────────────────────────┤
│  Query Input                                                     │
│    └─ Text normalization                                        │
│    └─ Query embedding (WASM, <3ms)                              │
├─────────────────────────────────────────────────────────────────┤
│  Parallel Search                                                 │
│    ├─ Vector Search (HNSW)          ├─ Keyword Search (BM25)   │
│    │    └─ Cosine similarity        │    └─ Full-text index    │
│    │    └─ Top-K candidates         │    └─ TF-IDF scoring     │
├─────────────────────────────────────────────────────────────────┤
│  Result Fusion                                                   │
│    └─ Reciprocal Rank Fusion (RRF)                              │
│    └─ Configurable weights: α·vector + β·keyword               │
│    └─ Deduplication and ranking                                 │
├─────────────────────────────────────────────────────────────────┤
│  Post-Processing                                                 │
│    └─ Score normalization                                       │
│    └─ Snippet extraction                                        │
│    └─ Source attribution                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```typescript
interface HybridSearchConfig {
  vector: {
    enabled: boolean;
    weight: number;  // 0.0-1.0
    dimensions: number;
    efSearch: number;
  };
  keyword: {
    enabled: boolean;
    weight: number;  // 0.0-1.0
    ftsTable: string;
    bm25k1: number;
    bm25b: number;
  };
  fusion: {
    method: 'rrf' | 'linear' | 'learned';
    candidateMultiplier: number;
    minScore: number;
  };
}
```

### Performance Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Query embedding | <5ms | 2.7ms |
| Vector search (100K) | <10ms | <5ms |
| Keyword search | <20ms | <15ms |
| Fusion | <5ms | <2ms |
| Total hybrid | <40ms | <25ms |

## Consequences

### Positive
- Better recall than vector-only search
- Handles exact matches and semantic similarity
- Maintains keyword search for debugging

### Negative
- Slightly higher latency than vector-only
- Requires maintaining FTS index
- More complex tuning

### Trade-offs
- Weight tuning requires experimentation
- Memory overhead for dual indices
