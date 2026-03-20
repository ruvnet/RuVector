# Companion Systems: Drift Detection & Attributed ANN

**Document ID**: spectral-sparsification/04-companion-systems
**Date**: 2026-03-19
**Status**: Research Complete
**Classification**: Companion System Design
**Series**: [00](./00-executive-summary.md) | [01](./01-algorithms-sota.md) | [02](./02-rust-crate-design.md) | [03](./03-ruvector-integration.md) | **04**

---

## 1. Online Drift Detection with Conformal Monitoring

### 1.1 The Problem

RuVector operates in non-stationary environments where:
- Embedding distributions shift over time
- Query patterns evolve
- Memory clusters go stale
- Agent working states move out of distribution

Traditional detection relies on batch analysis — too slow for real-time systems.

### 1.2 Conformal and Martingale-Based Drift

**Conformal prediction** provides distribution-free uncertainty quantification. Combined with **martingale-based change detection**, it gives:

- **Online**: processes one observation at a time, O(1) per update
- **Distribution-free**: no parametric assumptions
- **Provably calibrated**: false positive rate controlled at any desired level
- **Adaptive**: detects both gradual drift and sudden shifts

The key statistic is the **conformal martingale**:

```
M_t = ∏ᵢ₌₁ᵗ ε · p_i^{ε-1}
```

where p_i is the conformal p-value of observation i. When M_t exceeds a threshold, drift is detected.

### 1.3 Design: Lightweight Per-Shard Witness

```rust
/// Online drift witness for a single shard/agent/memory region.
pub struct DriftWitness {
    /// Non-conformity scores from calibration set.
    calibration_scores: Vec<f64>,
    /// Running conformal martingale value.
    martingale: f64,
    /// Detection threshold (e.g., 100 for ~1% false positive rate).
    threshold: f64,
    /// Observations since last reset.
    observation_count: u64,
}

impl DriftWitness {
    /// Process a new observation and return whether drift is detected.
    pub fn observe(&mut self, non_conformity_score: f64) -> bool {
        let p_value = self.conformal_p_value(non_conformity_score);
        // Update martingale (power martingale with ε=0.92)
        let epsilon = 0.92;
        self.martingale *= epsilon * p_value.powf(epsilon - 1.0);
        self.observation_count += 1;
        self.martingale > self.threshold
    }
}
```

### 1.4 Compound Evidence: Structural + Statistical

The key architectural insight: **structural alarms** (min-cut, sparsifier audit) and **statistical alarms** (conformal drift) should fire together before taking action.

```
Min-cut alarm       → structure is changing
Conformal drift     → stream is statistically untrustworthy
Both fire together  → trigger reindex / retrain / reroute / human review
```

This avoids:
- False positives from noise (only structural change, no statistical drift)
- Missed changes (statistical drift in a structurally stable region)

```rust
pub fn should_act(
    mincut_alarm: bool,
    sparsifier_audit_failed: bool,
    drift_detected: bool,
) -> Action {
    match (mincut_alarm || sparsifier_audit_failed, drift_detected) {
        (true, true)   => Action::Reindex,       // Both: strong signal
        (true, false)  => Action::Monitor,        // Structural only: watch
        (false, true)  => Action::IncreaseSampling, // Statistical only: gather evidence
        (false, false) => Action::Continue,       // All clear
    }
}
```

---

## 2. Dynamic Graph Routing for Filtered/Attributed ANN

### 2.1 The Problem

Modern vector systems need nearest neighbors under constraints:
- **Time**: only documents from the last 24 hours
- **Tenant**: only this user's data
- **Source**: only from specific data sources
- **Modality**: only text, only images
- **Security scope**: only documents the user can access
- **Metadata filters**: arbitrary attribute predicates

The naive approach (separate index per filter combination) explodes exponentially.

### 2.2 Attributed Graph Overlay

Build a single graph with attributed edges:

```rust
pub struct AttributedEdge {
    pub u: usize,
    pub v: usize,
    pub weight: f64,
    pub attributes: AttributeSet,
}

pub struct AttributeSet {
    pub tenant_id: Option<u64>,
    pub time_range: Option<(u64, u64)>,
    pub source_tag: Option<u32>,
    pub security_level: u8,
}
```

Store **common routing edges once**, then materialize filtered search paths cheaply:

```
Full attributed graph
    ↓ filter(tenant=X, time>T)
Filtered view (virtual, not materialized)
    ↓ search
Nearest neighbors under constraints
```

### 2.3 Sparsifier Connection

The sparsifier can operate on the attributed graph:
- Maintain a **structural sparsifier** of the full unfiltered graph
- For each filter query, project the sparsifier to the relevant subgraph
- Use the projected sparsifier for cheap routing/diagnostic decisions
- Verify final results on the full filtered graph

### 2.4 Implementation Sketch

```rust
pub struct AttributedGraph {
    /// Base graph (all edges).
    base: SparseGraph,
    /// Edge attributes indexed by canonical edge key.
    attributes: HashMap<(usize, usize), AttributeSet>,
    /// Sparsifier of the base graph.
    sparsifier: AdaptiveGeoSpar,
}

impl AttributedGraph {
    /// Get a filtered view of the sparsifier.
    pub fn filtered_sparsifier(
        &self,
        predicate: &dyn Fn(&AttributeSet) -> bool,
    ) -> SparseGraph {
        let mut filtered = SparseGraph::with_capacity(
            self.sparsifier.sparsifier().num_vertices()
        );
        for (u, v, w) in self.sparsifier.sparsifier().edges() {
            let key = if u <= v { (u, v) } else { (v, u) };
            if let Some(attrs) = self.attributes.get(&key) {
                if predicate(attrs) {
                    let _ = filtered.insert_edge(u, v, w);
                }
            }
        }
        filtered
    }
}
```

### 2.5 Why This Matters

This turns RuVector from "vector database" into "general purpose memory substrate":
- Single graph with many filtered views
- Multi-tenant search without index explosion
- Hybrid memory where symbolic filters and vector similarity coexist naturally

---

## 3. Kernel Graph Sparsification for Embeddings

### 3.1 The Embedding-Graph Connection

RuVector's similarity graphs are **kernel graphs**: each vector embedding p_i ∈ ℝᵈ is a node, and edge weights are kernel evaluations:

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Cosine | p·q / (‖p‖‖q‖) | Text embeddings |
| Gaussian RBF | exp(-‖p-q‖²/2h²) | Smooth similarity |
| Dot product | p·q | Raw inner product |
| Hybrid | α·cos + β·RBF | Multi-signal |

### 3.2 Embedding Update = Point Move

When a vector embedding is updated (reindexed, fine-tuned, or corrected):
1. The point p_i moves in ℝᵈ
2. All incident edge weights change (because kernel evaluations change)
3. This is exactly a **point update** in the kernel graph

Recent dynamic kernel graph sparsification results show this can be handled in n^{o(1)} time per point update, maintaining:
- A spectral sparsifier with near-linear edges
- Dynamic sketches for Laplacian multiplication
- Dynamic sketches for Laplacian solving

### 3.3 RuVector Integration

```
Embedding update (vector reindex)
    ↓ triggers
Kernel graph point move
    ↓ updates
Dynamic sparsifier (incremental)
    ↓ enables
Cheap Laplacian analytics (spectral coherence, clustering, diffusion)
    ↓ informs
Control decisions (reindex, reroute, alert)
```

The `update_embedding(node, old_neighbors, new_neighbors)` method in `ruvector-sparsifier` directly implements this flow:

```rust
// When vector 42 is reindexed:
let old_nbrs: Vec<(usize, f64)> = old_similarity_scores;
let new_nbrs: Vec<(usize, f64)> = new_similarity_scores;
spar.update_embedding(42, &old_nbrs, &new_nbrs)?;

// Immediately available: cheap analytics on updated sparsifier
let audit = spar.audit();
let coherence = spectral_coherence(spar.sparsifier());
```

### 3.4 Dynamic Laplacian Operations

The kernel graph sparsifier enables:

| Operation | Description | Use Case |
|-----------|-------------|----------|
| L_H × v | Laplacian-vector multiply on sparsifier | Graph signal filtering |
| L_H⁻¹ × v | Laplacian solve on sparsifier | Label propagation |
| λ₁(L_H) | Fiedler value on sparsifier | Connectivity monitoring |
| spectral_cluster(L_H) | Spectral clustering on sparsifier | Community detection |

All at 10-17x reduced cost compared to the full graph.

---

## 4. Synthesis: The Complete Control Stack

```
data → structure → continuous reasoning → action

Full graph     = memory (ground truth)
Sparsifier     = reasoning layer (compressed world model)
Min-cut        = stability detector (fragility alarm)
Conformal drift = statistical witness (distribution monitor)
Attributed ANN = filtered perception (constrained retrieval)
ruQu           = evolution (structural mutation)
```

Without sparsification:
- Heavy, slow, reactive, local view

With sparsification:
- Fast, aware, proactive, global understanding

**The system moves from lookup → reasoning system.**

---

## Sources

- [Vovk et al., "Algorithmic Learning in a Random World" (2005)](https://link.springer.com/book/10.1007/b106715) — Conformal prediction foundations
- [Fedorova et al., "Plug-in martingales for testing exchangeability on-line" (ICML 2012)](https://proceedings.mlr.press/v25/fedorova12.html) — Online drift detection
- [Recent VLDB work on dynamic range filtering ANN](https://www.vldb.org/) — Attributed ANN
- [Abraham et al., "On Fully Dynamic Graph Sparsifiers" (FOCS 2016)](https://arxiv.org/abs/1604.02094)
- [Spielman & Srivastava, "Graph Sparsification by Effective Resistances" (SIAM 2011)](https://arxiv.org/abs/0803.0929)

---

## Document Navigation

- **Previous**: [03 - RuVector Integration](./03-ruvector-integration.md)
- **Index**: [00 - Executive Summary](./00-executive-summary.md)
