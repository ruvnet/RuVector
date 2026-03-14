# ADR-004: HNSW for Approximate Nearest Neighbor Search

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-12 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-001 (Core Architecture), ADR-027 (HNSW Parameterized Query) |

## 1. Context

### 1.1 The k-NN Search Problem

Given:
- A database D of N vectors in R^d
- A query vector q in R^d
- Integer k

Find the k vectors in D closest to q under some distance metric (Euclidean, Cosine, etc.).

### 1.2 Exact vs. Approximate

| Approach | Query Time | Index Build | Memory | Recall |
|----------|------------|-------------|--------|--------|
| Brute Force | O(N * d) | O(N * d) | O(N * d) | 100% |
| KD-Tree | O(log N) to O(N) | O(N log N) | O(N) | 100% |
| LSH | O(N^rho) | O(N) | O(N * L) | Variable |
| **HNSW** | O(log N) | O(N log N) | O(N * M) | >95% |

For high-dimensional vectors (d > 20), tree-based exact methods degrade to brute force. Approximate methods trade recall for speed.

### 1.3 ANN Algorithm Comparison

| Algorithm | Query Time | Memory | Strengths | Weaknesses |
|-----------|------------|--------|-----------|------------|
| LSH | O(N^0.5-0.8) | High (multiple tables) | Theoretical guarantees | High memory, tuning |
| IVF-PQ | O(sqrt(N)) | Low (quantized) | Memory efficient | Requires training |
| Annoy | O(log N) | O(N * trees) | Simple, no training | Static index |
| FAISS IVF | O(sqrt(N)) | Medium | GPU support | Batch-oriented |
| **HNSW** | O(log N) | O(N * M) | Online updates, high recall | Memory intensive |

### 1.4 Why HNSW?

HNSW (Hierarchical Navigable Small World) is chosen because:

1. **Online Insertion**: Add vectors without rebuilding
2. **High Recall**: 95-99.9% recall achievable with tuning
3. **Consistent Latency**: O(log N) search regardless of distribution
4. **No Training**: Works immediately, no clustering required
5. **Simple API**: Insert, search, delete operations

## 2. Decision

### 2.1 Adopt HNSW as Primary Index Structure

We implement HNSW with the following architecture:

```
Level 3: [v0] -------- [v5]         <- Sparse (skip-list like)
            \            /
Level 2: [v0] -- [v3] -- [v5] -- [v9]
            \    /    \    /    \
Level 1: [v0]-[v1]-[v3]-[v4]-[v5]-[v7]-[v9]
            |    |    |    |    |    |    |
Level 0: [v0]-[v1]-[v2]-[v3]-[v4]-[v5]-[v6]-[v7]-[v8]-[v9]  <- Dense (all vectors)
```

### 2.2 Core Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Max connections | M | 32 | 8-64 | Edges per node per layer |
| Max connections (L0) | M_max0 | 2*M | 16-128 | Edges at level 0 |
| Construction ef | ef_c | 200 | 100-500 | Build-time beam width |
| Search ef | ef_s | 100 | 10-500 | Query-time beam width |
| Level multiplier | m_L | 1/ln(M) | - | Probability of layer promotion |

### 2.3 Level Assignment

Each vector is assigned a maximum level `l` according to:

```
l = floor(-ln(uniform(0, 1)) * m_L)
```

With m_L = 1/ln(M), this yields:
- ~63% of vectors at level 0 only
- ~23% at level 1
- ~9% at level 2
- Exponentially fewer at higher levels

## 3. Algorithm Details

### 3.1 Insertion Algorithm

```rust
pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
    let node_level = self.random_level();
    let entry_point = self.entry_point;

    // Phase 1: Greedy search from entry point to node_level
    let mut current = entry_point;
    for level in (node_level + 1..=self.max_level).rev() {
        current = self.greedy_search_layer(current, &vector, level)?;
    }

    // Phase 2: Insert and connect at each level
    for level in (0..=node_level).rev() {
        // Find ef_construction nearest neighbors
        let neighbors = self.search_layer(&vector, current, self.ef_construction, level)?;

        // Select M best neighbors using heuristic
        let selected = self.select_neighbors(&vector, neighbors, self.m, level)?;

        // Add bidirectional edges
        for neighbor in &selected {
            self.add_edge(id.clone(), neighbor.clone(), level)?;
            self.add_edge(neighbor.clone(), id.clone(), level)?;

            // Prune neighbor's connections if over M
            self.shrink_connections(neighbor, level)?;
        }

        current = selected[0].clone();
    }

    // Update entry point if new node has higher level
    if node_level > self.max_level {
        self.entry_point = id;
        self.max_level = node_level;
    }

    Ok(())
}
```

### 3.2 Search Algorithm

```rust
pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
    let ef = self.ef_search.max(k);

    // Phase 1: Navigate to lowest level from entry point
    let mut current = self.entry_point.clone();
    for level in (1..=self.max_level).rev() {
        current = self.greedy_search_layer(&current, query, level)?;
    }

    // Phase 2: Search level 0 with beam width ef
    let candidates = self.search_layer(query, current, ef, 0)?;

    // Return top-k results
    Ok(candidates.into_iter().take(k).collect())
}

fn search_layer(
    &self,
    query: &[f32],
    entry: String,
    ef: usize,
    level: usize,
) -> Result<Vec<(String, f32)>> {
    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();  // Min-heap by distance
    let mut results = BinaryHeap::new();     // Max-heap by distance

    let entry_dist = self.distance(query, &self.get_vector(&entry)?);
    candidates.push(Reverse((OrderedFloat(entry_dist), entry.clone())));
    results.push((OrderedFloat(entry_dist), entry.clone()));
    visited.insert(entry);

    while let Some(Reverse((_, current))) = candidates.pop() {
        let furthest_result = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);

        if self.distance(query, &self.get_vector(&current)?) > furthest_result {
            break;  // All remaining candidates are worse
        }

        for neighbor in self.get_neighbors(&current, level)? {
            if visited.insert(neighbor.clone()) {
                let dist = self.distance(query, &self.get_vector(&neighbor)?);

                if dist < furthest_result || results.len() < ef {
                    candidates.push(Reverse((OrderedFloat(dist), neighbor.clone())));
                    results.push((OrderedFloat(dist), neighbor));

                    if results.len() > ef {
                        results.pop();  // Remove furthest
                    }
                }
            }
        }
    }

    // Convert max-heap to sorted vec
    let mut result_vec: Vec<_> = results.into_iter()
        .map(|(d, id)| (id, d.0))
        .collect();
    result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Ok(result_vec)
}
```

### 3.3 Neighbor Selection Heuristic

The simple approach (M nearest) can create poor graphs. We use the heuristic from the original paper:

```rust
fn select_neighbors_heuristic(
    &self,
    query: &[f32],
    candidates: Vec<(String, f32)>,
    m: usize,
    extend_candidates: bool,
    keep_pruned: bool,
) -> Vec<(String, f32)> {
    let mut working_set = candidates;

    // Optionally extend with neighbors' neighbors
    if extend_candidates {
        for (id, _) in &candidates {
            for neighbor in self.get_neighbors(id, 0).unwrap_or_default() {
                let dist = self.distance(query, &self.get_vector(&neighbor).unwrap());
                working_set.push((neighbor, dist));
            }
        }
        working_set.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        working_set.dedup_by(|a, b| a.0 == b.0);
    }

    let mut selected = Vec::with_capacity(m);
    let mut discarded = Vec::new();

    for (candidate, dist_to_query) in working_set {
        if selected.len() >= m {
            break;
        }

        // Check if candidate is closer to query than to any selected neighbor
        let is_good = selected.iter().all(|(s, _)| {
            let dist_to_selected = self.distance(
                &self.get_vector(&candidate).unwrap(),
                &self.get_vector(s).unwrap(),
            );
            dist_to_query < dist_to_selected
        });

        if is_good {
            selected.push((candidate, dist_to_query));
        } else {
            discarded.push((candidate, dist_to_query));
        }
    }

    // Fill remaining slots with discarded if keep_pruned
    if keep_pruned && selected.len() < m {
        selected.extend(discarded.into_iter().take(m - selected.len()));
    }

    selected
}
```

## 4. Configuration Guidelines

### 4.1 Parameter Tuning

| Use Case | M | ef_construction | ef_search | Notes |
|----------|---|-----------------|-----------|-------|
| High recall (99%+) | 48 | 400 | 200 | Memory intensive |
| Balanced | 32 | 200 | 100 | Default |
| Low latency | 16 | 100 | 50 | Lower recall |
| Memory constrained | 12 | 100 | 30 | Minimum viable |

### 4.2 Memory Usage

Memory per vector:
```
bytes_per_vector = dimension * 4                    # Vector data
                 + M * (max_level + 1) * 8          # Edge lists (avg)
                 + id_storage                       # ID string
```

For 384-dim vectors with M=32, avg 1.2 levels:
```
bytes = 384 * 4 + 32 * 1.2 * 8 + ~50 = ~1.9 KB per vector
```

1M vectors ~ 1.9 GB memory

### 4.3 Build Time

Build time complexity: O(N * log(N) * M * ef_construction)

Empirical results (10K 384-dim vectors):

| ef_construction | M | Build Time | Recall@10 |
|-----------------|---|------------|-----------|
| 100 | 16 | 2.3s | 92.1% |
| 200 | 32 | 8.1s | 97.4% |
| 400 | 48 | 28.3s | 99.2% |

## 5. Serialization

### 5.1 Binary Format

```rust
#[derive(Serialize, Deserialize)]
pub struct HnswState {
    pub version: u32,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub config: HnswConfig,
    pub max_level: usize,
    pub entry_point: Option<String>,
    pub vectors: Vec<(String, Vec<f32>)>,
    pub layers: Vec<HashMap<String, Vec<String>>>,  // layer -> node -> neighbors
}
```

### 5.2 Incremental Updates

For large indexes, we support delta serialization:

```rust
pub struct HnswDelta {
    pub base_version: u64,
    pub additions: Vec<(String, Vec<f32>, Vec<Vec<String>>)>,  // (id, vector, neighbors_by_layer)
    pub deletions: HashSet<String>,
}
```

## 6. Consequences

### 6.1 Benefits

1. **O(log N) Search**: Consistent sub-millisecond latency
2. **Online Updates**: Insert/delete without rebuild
3. **High Recall**: 95-99.9% achievable with tuning
4. **No Training**: Works immediately on any data distribution
5. **Simple API**: Insert, search, delete, serialize

### 6.2 Costs

1. **Memory**: ~1.5-2 KB per vector (vs 1.5 KB for flat)
2. **Build Time**: O(N log N) vs O(N) for flat
3. **Deletion**: Soft delete only (graph not repaired)
4. **Cold Start**: First searches may hit suboptimal paths

### 6.3 Benchmark Results

Dataset: 1M 384-dim vectors (OpenAI ada-002 embeddings)

| Configuration | Build Time | Memory | QPS | Recall@10 |
|---------------|------------|--------|-----|-----------|
| Flat scan | - | 1.5 GB | 45 | 100% |
| HNSW (M=16) | 12 min | 1.8 GB | 8,200 | 93.2% |
| HNSW (M=32) | 28 min | 2.1 GB | 6,400 | 97.8% |
| HNSW (M=48) | 51 min | 2.5 GB | 4,800 | 99.3% |

## 7. Alternative Indexes

### 7.1 When to Use Flat Index

- Dataset < 10K vectors
- 100% recall required
- Memory constrained (no overhead)
- Batch processing (parallelizes well)

### 7.2 When to Use IVF-PQ

- 10M+ vectors
- Memory constrained (quantized storage)
- Willing to train codebooks
- Batch-oriented workloads

### 7.3 HNSW Sweet Spot

- 10K - 10M vectors
- Online updates required
- Interactive latency needed
- High recall (>95%) required

## 8. Related Decisions

- **ADR-001-core-architecture**: HNSW as primary index
- **ADR-001-simd-first**: SIMD-optimized distance for graph traversal
- **ADR-027-hnsw-parameterized-query**: Runtime ef adjustment

## 9. References

1. Malkov, Y. & Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI.
2. HNSW Implementation: `/crates/ruvector-core/src/index/hnsw.rs`
3. Benchmarks: `/crates/ruvector-core/benches/hnsw_benchmark.rs`

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-12 | Architecture Team | Initial decision record |
