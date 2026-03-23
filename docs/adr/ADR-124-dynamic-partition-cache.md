# ADR-124: Dynamic Partition Cache with Large-Graph Guard

## Status
Accepted

## Context
The pi.ruv.io brain server's `/v1/partition` endpoint runs exact Stoer-Wagner MinCut on the knowledge graph. With 2,090+ nodes and 971K+ edges, this computation exceeds Cloud Run's 300s timeout and the MCP SSE transport's 60s tool call timeout.

PR #287 (ADR-117) introduced a canonical source-anchored MinCut with deterministic hashing, but it shares the same O(V*E) complexity for the initial computation.

## Decision

### 1. Partition Caching
Add `cached_partition: Arc<RwLock<Option<PartitionResult>>>` to `AppState`. The cache is populated during training cycles and served instantly via:
- **MCP `brain_partition`**: Returns cached compact partition (sub-millisecond)
- **REST `/v1/partition`**: Returns cache by default; `?force=true` recomputes

### 2. Large-Graph Guard
During training cycles, only compute exact MinCut if `edge_count <= 100,000`. For larger graphs, the cache remains unpopulated until the scheduled `rebuild_graph` job runs asynchronously without timeout pressure.

### 3. Tier Roadmap (from ADR-117)
- **Tier 1 (shipped)**: Exact Stoer-Wagner + source-anchored canonical cut
- **Tier 2 (next)**: Tree packing for O(V^2 log V) fast path
- **Tier 3 (future)**: Incremental maintenance for evolving graphs (dynamic MinCut)

## Consequences
- MCP `brain_partition` returns instantly from cache instead of timing out
- Enhanced training cycle no longer blocks on MinCut for large graphs
- Fresh partition data depends on scheduled background jobs for graphs >100K edges
- REST `?force=true` still allows on-demand recomputation (may timeout for large graphs)

## Benchmark Results (Canonical MinCut - ADR-117)

| Graph Type | Nodes | Time |
|-----------|-------|------|
| Cycle | 50 | 3.09 us |
| Complete | 10 | 2.61 us |
| Hash stability | 100 | 1.39 us |
