# ADR-143: Implement Missing Capabilities in ruvector

## Status
Accepted

## Date
2026-04-06

## Context

A comprehensive audit of the `ruvector` npm package (v0.2.22) identified 3 gaps where claimed capabilities were either stubs or trivially implemented:

1. **Speculative Embedding (parallel-workers.ts)** - The `speculativeEmbed` worker returned `{ embedding: [], confidence: 0.5 }` for all files. No actual embedding computation occurred.

2. **RAG Retrieval (parallel-workers.ts)** - The `ragRetrieve` and `contextRank` workers used keyword-matching (`string.includes()`) instead of semantic similarity on embeddings, despite the module claiming "Parallel RAG chunking and retrieval" and "Semantic deduplication."

3. **DiskANN / Vamana (README, package.json)** - Claimed in README ("billion-scale SSD-backed ANN with <10ms latency") and package.json description/keywords, but no implementation exists anywhere in the codebase.

All other 14 modules were verified as real implementations (see release v2.1.1 audit).

## Decision

### 1. Speculative Embedding - Implement real hash-based embedding

Replace the stub with the same multi-hash embedding approach used in `intelligence-engine.ts` (FNV-1a + positional encoding). This produces deterministic, consistent embeddings from file content without requiring ONNX or native modules. The worker already has access to `fs` for reading file content.

Embedding dimension: 128 (sufficient for co-edit prediction, avoids overhead of 384-dim).

### 2. RAG Retrieval - Implement cosine similarity on embeddings

When chunks include embeddings, use cosine similarity for ranking. Fall back to keyword matching only when embeddings are absent. This makes the existing `embedding?` field on `ContextChunk` actually functional.

Also upgrade `contextRank` to use TF-IDF weighting instead of raw keyword matching.

### 3. DiskANN - Remove false claims, add roadmap note

DiskANN/Vamana requires SSD-backed graph storage with PQ compression — a significant implementation effort that should be a dedicated Rust crate. Rather than ship a stub, remove the claim from README/package.json and add it to a roadmap section. The existing HNSW index (backed by `hnsw_rs`) already provides fast ANN search for in-memory datasets.

## Consequences

- Speculative embedding becomes functional for co-edit prediction use cases
- RAG retrieval produces semantically meaningful results when embeddings are available
- README accurately reflects capabilities (no DiskANN claim without implementation)
- No new dependencies required (all implementations use existing math primitives)
