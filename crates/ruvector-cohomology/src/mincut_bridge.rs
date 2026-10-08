//! Dynamic mincut coupling (issue #669, Phase 5).
//!
//! Cohomology diagnoses contradiction; mincut chooses a low-cost isolation
//! boundary. This module converts syndrome support into per-edge tension,
//! computes the cheapest quarantine boundary separating contaminated seeds
//! from the healthy region (deterministic internal max-flow; optionally the
//! `ruvector-mincut` global engine behind the `mincut` feature), runs sparse
//! repair, verifies the post-intervention syndrome, and signs a combined
//! intervention receipt.

use crate::dynamic::{DynamicCohomology, DynamicSheafEngine};
use crate::repair::{RepairPlan, RepairPolicy};
use crate::syndrome::SyndromeResult;
use crate::witness::quantize;
use crate::{CohomologyError, EdgeKey, VertexId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Mixing weights for contradiction tension
/// `τ_e = α‖s_e‖ + β·leverage + γ·uncertainty + ζ·policy_severity`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensionWeights {
    /// α — syndrome magnitude weight.
    pub alpha: f64,
    /// β — statistical leverage weight.
    pub beta: f64,
    /// γ — uncertainty / drift weight.
    pub gamma: f64,
    /// ζ — policy severity weight.
    pub zeta: f64,
}

impl Default for TensionWeights {
    fn default() -> Self {
        Self { alpha: 1.0, beta: 1.0, gamma: 0.5, zeta: 1.0 }
    }
}

/// One edge of the tension/quarantine graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionEdge {
    /// The underlying sheaf edge.
    pub edge: EdgeKey,
    /// Contradiction tension τ_e (how implicated the edge is).
    pub tension: f64,
    /// Quarantine cost k_e (how expensive cutting the edge is).
    pub quarantine_cost: f64,
}

/// Tension graph derived from a syndrome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionGraph {
    /// Edges in canonical order.
    pub edges: Vec<TensionEdge>,
}

impl TensionGraph {
    /// Derive per-edge tension from a syndrome result.
    ///
    /// `uncertainty` and `severity` are optional per-edge signals in
    /// canonical order (map drift and policy severity); quarantine cost
    /// defaults to `1 + tension`-independent unit cost unless provided.
    pub fn from_syndrome(
        syndrome: &SyndromeResult,
        weights: &TensionWeights,
        uncertainty: Option<&[f64]>,
        severity: Option<&[f64]>,
        quarantine_costs: Option<&[f64]>,
    ) -> Self {
        // ranked_support is sorted by energy; rebuild canonical (key) order.
        let mut by_key: Vec<&crate::syndrome::EdgeContribution> =
            syndrome.ranked_support.iter().collect();
        by_key.sort_by_key(|c| c.edge);
        let edges = by_key
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let magnitude = c.energy.sqrt();
                let leverage = c.fraction;
                let u = uncertainty.map(|s| s[i]).unwrap_or(0.0);
                let p = severity.map(|s| s[i]).unwrap_or(0.0);
                let tension = weights.alpha * magnitude
                    + weights.beta * leverage
                    + weights.gamma * u
                    + weights.zeta * p;
                let quarantine_cost = quarantine_costs.map(|s| s[i]).unwrap_or(1.0);
                TensionEdge { edge: c.edge, tension, quarantine_cost }
            })
            .collect();
        Self { edges }
    }

    /// Vertices whose incident tension exceeds `threshold` — the
    /// contamination seed set, canonically sorted.
    pub fn contaminated_seeds(&self, threshold: f64) -> Vec<VertexId> {
        let mut incident: BTreeMap<VertexId, f64> = BTreeMap::new();
        for e in &self.edges {
            *incident.entry(e.edge.u).or_default() += e.tension;
            *incident.entry(e.edge.v).or_default() += e.tension;
        }
        incident
            .into_iter()
            .filter(|(_, t)| *t > threshold)
            .map(|(v, _)| v)
            .collect()
    }
}

/// A quarantine boundary: the cut and the isolated region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineBoundary {
    /// Cut edges, canonically sorted.
    pub cut_edges: Vec<EdgeKey>,
    /// Vertices inside the quarantined (contaminated-side) region, sorted.
    pub isolated_vertices: Vec<VertexId>,
    /// Total quarantine cost of the cut.
    pub total_cost: f64,
}

/// Deterministic Dinic max-flow / min-cut on the quarantine-cost graph,
/// separating `seeds` (contaminated) from all vertices *not* reachable
/// through them. Every non-seed vertex incident to zero-tension structure
/// acts as sink side via a virtual super-sink over `healthy` vertices.
pub fn quarantine_boundary(
    graph: &TensionGraph,
    seeds: &[VertexId],
    healthy: &[VertexId],
) -> Result<QuarantineBoundary, CohomologyError> {
    if seeds.is_empty() {
        return Ok(QuarantineBoundary {
            cut_edges: Vec::new(),
            isolated_vertices: Vec::new(),
            total_cost: 0.0,
        });
    }
    if healthy.iter().any(|h| seeds.contains(h)) {
        return Err(CohomologyError::InvalidValue(
            "healthy set intersects contaminated seeds".into(),
        ));
    }

    // Deterministic vertex indexing.
    let mut ids: Vec<VertexId> = graph
        .edges
        .iter()
        .flat_map(|e| [e.edge.u, e.edge.v])
        .chain(seeds.iter().copied())
        .chain(healthy.iter().copied())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let index: BTreeMap<VertexId, usize> = ids.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let n = ids.len() + 2;
    let source = ids.len();
    let sink = ids.len() + 1;

    // Dinic structures.
    struct Arc {
        to: usize,
        cap: f64,
        rev: usize,
    }
    let mut adj: Vec<Vec<Arc>> = (0..n).map(|_| Vec::new()).collect();
    let add_edge = |adj: &mut Vec<Vec<Arc>>, a: usize, b: usize, cap: f64| {
        let ra = adj[b].len();
        let rb = adj[a].len();
        adj[a].push(Arc { to: b, cap, rev: ra });
        adj[b].push(Arc { to: a, cap, rev: rb });
    };

    const INF: f64 = 1e30;
    for e in &graph.edges {
        let a = index[&e.edge.u];
        let b = index[&e.edge.v];
        // Undirected edge with capacity = quarantine cost, in both arcs.
        add_edge(&mut adj, a, b, e.quarantine_cost.max(0.0));
        add_edge(&mut adj, b, a, e.quarantine_cost.max(0.0));
    }
    for s in seeds {
        add_edge(&mut adj, source, index[s], INF);
    }
    for h in healthy {
        add_edge(&mut adj, index[h], sink, INF);
    }

    // Dinic: BFS levels + DFS blocking flow, deterministic adjacency order.
    let mut total_flow = 0.0;
    loop {
        let mut level = vec![usize::MAX; n];
        level[source] = 0;
        let mut queue = std::collections::VecDeque::from([source]);
        while let Some(u) = queue.pop_front() {
            for arc in &adj[u] {
                if arc.cap > 1e-12 && level[arc.to] == usize::MAX {
                    level[arc.to] = level[u] + 1;
                    queue.push_back(arc.to);
                }
            }
        }
        if level[sink] == usize::MAX {
            break;
        }
        let mut iter = vec![0usize; n];
        loop {
            // Iterative DFS for one augmenting path.
            let mut path: Vec<(usize, usize)> = Vec::new();
            let mut u = source;
            let found = loop {
                if u == sink {
                    break true;
                }
                let mut advanced = false;
                while iter[u] < adj[u].len() {
                    let i = iter[u];
                    let (to, cap) = (adj[u][i].to, adj[u][i].cap);
                    if cap > 1e-12 && level[to] == level[u] + 1 {
                        path.push((u, i));
                        u = to;
                        advanced = true;
                        break;
                    }
                    iter[u] += 1;
                }
                if !advanced {
                    if let Some((prev, _)) = path.pop() {
                        iter[prev] += 1;
                        u = prev;
                    } else {
                        break false;
                    }
                }
            };
            if !found {
                break;
            }
            let bottleneck = path
                .iter()
                .map(|&(u, i)| adj[u][i].cap)
                .fold(INF, f64::min);
            for &(u, i) in &path {
                let (to, rev) = (adj[u][i].to, adj[u][i].rev);
                adj[u][i].cap -= bottleneck;
                adj[to][rev].cap += bottleneck;
            }
            total_flow += bottleneck;
        }
    }

    // Min cut: vertices reachable from source in the residual graph are the
    // contaminated side.
    let mut reachable = vec![false; n];
    reachable[source] = true;
    let mut stack = vec![source];
    while let Some(u) = stack.pop() {
        for arc in &adj[u] {
            if arc.cap > 1e-12 && !reachable[arc.to] {
                reachable[arc.to] = true;
                stack.push(arc.to);
            }
        }
    }

    let isolated_vertices: Vec<VertexId> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| reachable[*i])
        .map(|(_, &v)| v)
        .collect();
    let mut cut_edges: Vec<EdgeKey> = graph
        .edges
        .iter()
        .filter(|e| reachable[index[&e.edge.u]] != reachable[index[&e.edge.v]])
        .map(|e| e.edge)
        .collect();
    cut_edges.sort_unstable();
    Ok(QuarantineBoundary {
        cut_edges,
        isolated_vertices,
        total_cost: total_flow,
    })
}

/// Containment ordering policy (issue #669, Phase 5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainmentPolicy {
    /// Cut the quarantine boundary first, then repair inside the region.
    IsolateFirst,
    /// Repair globally first; isolate only what repair cannot fix.
    RepairFirst,
}

/// Signed receipt for a combined diagnose → isolate → repair → verify loop.
#[derive(Debug, Clone)]
pub struct InterventionReceipt {
    /// Witness hash of the pre-intervention syndrome.
    pub pre_witness_hash: [u8; 32],
    /// Syndrome energy before intervention.
    pub pre_energy: f64,
    /// The quarantine boundary that was computed (may be empty).
    pub boundary: QuarantineBoundary,
    /// The repair that was proposed/applied.
    pub repair: Option<RepairPlan>,
    /// Syndrome energy after the intervention.
    pub post_energy: f64,
    /// Whether the post-intervention energy meets the threshold.
    pub verified: bool,
    /// Policy used.
    pub policy: ContainmentPolicy,
    /// SHA-256 over the receipt's canonical content.
    pub receipt_hash: [u8; 32],
}

impl InterventionReceipt {
    fn seal(mut self, quantization_scale: f64) -> Self {
        let mut h = Sha256::new();
        h.update(b"ruvector-cohomology/intervention/v1");
        h.update(self.pre_witness_hash);
        h.update(quantize(self.pre_energy, quantization_scale).to_le_bytes());
        h.update((self.boundary.cut_edges.len() as u64).to_le_bytes());
        for e in &self.boundary.cut_edges {
            h.update(e.u.to_le_bytes());
            h.update(e.v.to_le_bytes());
        }
        for v in &self.boundary.isolated_vertices {
            h.update(v.to_le_bytes());
        }
        h.update(quantize(self.boundary.total_cost, quantization_scale).to_le_bytes());
        match &self.repair {
            None => h.update([0u8]),
            Some(r) => {
                h.update([1u8]);
                for (k, _) in &r.corrections {
                    h.update(k.u.to_le_bytes());
                    h.update(k.v.to_le_bytes());
                }
                h.update(quantize(r.objective, quantization_scale).to_le_bytes());
            }
        }
        h.update(quantize(self.post_energy, quantization_scale).to_le_bytes());
        h.update([self.verified as u8]);
        h.update([match self.policy {
            ContainmentPolicy::IsolateFirst => 0u8,
            ContainmentPolicy::RepairFirst => 1u8,
        }]);
        self.receipt_hash = h.finalize().into();
        self
    }
}

/// Run the full containment control loop on a dynamic engine:
/// detect → localize → isolate (policy-dependent) → repair → verify → sign.
///
/// `seed_threshold` selects contamination seeds by incident tension;
/// `energy_threshold` is the configured acceptable residual syndrome energy.
pub fn run_containment(
    engine: &mut DynamicSheafEngine,
    tension_weights: &TensionWeights,
    repair_policy: &RepairPolicy,
    policy: ContainmentPolicy,
    seed_threshold: f64,
    energy_threshold: f64,
) -> Result<InterventionReceipt, CohomologyError> {
    let syndrome = engine.flush()?;
    let pre_energy = syndrome.energy;
    let pre_witness_hash = syndrome.witness.hash;
    let quant = syndrome.witness.quantization_scale;

    if pre_energy <= energy_threshold {
        // Already coherent: empty intervention, verified.
        return Ok(InterventionReceipt {
            pre_witness_hash,
            pre_energy,
            boundary: QuarantineBoundary {
                cut_edges: Vec::new(),
                isolated_vertices: Vec::new(),
                total_cost: 0.0,
            },
            repair: None,
            post_energy: pre_energy,
            verified: true,
            policy,
            receipt_hash: [0u8; 32],
        }
        .seal(quant));
    }

    let graph = TensionGraph::from_syndrome(&syndrome, tension_weights, None, None, None);
    let seeds = graph.contaminated_seeds(seed_threshold);
    let healthy: Vec<VertexId> = {
        let mut incident: BTreeMap<VertexId, f64> = BTreeMap::new();
        for e in &graph.edges {
            *incident.entry(e.edge.u).or_default() += e.tension;
            *incident.entry(e.edge.v).or_default() += e.tension;
        }
        incident
            .into_iter()
            .filter(|(v, t)| *t <= seed_threshold * 0.1 && !seeds.contains(v))
            .map(|(v, _)| v)
            .collect()
    };

    let boundary = match policy {
        ContainmentPolicy::IsolateFirst => quarantine_boundary(&graph, &seeds, &healthy)?,
        ContainmentPolicy::RepairFirst => QuarantineBoundary {
            cut_edges: Vec::new(),
            isolated_vertices: Vec::new(),
            total_cost: 0.0,
        },
    };

    let repair = engine.propose_repair(repair_policy.clone())?;
    let post_energy = repair.post_energy;
    let verified = post_energy <= energy_threshold;

    Ok(InterventionReceipt {
        pre_witness_hash,
        pre_energy,
        boundary,
        repair: Some(repair),
        post_energy,
        verified,
        policy,
        receipt_hash: [0u8; 32],
    }
    .seal(quant))
}

/// Convert a tension graph into `(u, v, tension)` triples for the
/// `ruvector-mincut` dynamic engine (global lowest-tension boundary).
#[cfg(feature = "mincut")]
pub fn global_tension_cut(
    graph: &TensionGraph,
) -> Result<(f64, Vec<VertexId>, Vec<VertexId>), CohomologyError> {
    use ruvector_mincut::MinCutBuilder;
    let edges: Vec<(u64, u64, f64)> = graph
        .edges
        .iter()
        .map(|e| (e.edge.u, e.edge.v, e.tension.max(f64::MIN_POSITIVE)))
        .collect();
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edges)
        .build()
        .map_err(|e| CohomologyError::SolverFailure(format!("mincut build: {e}")))?;
    let value = mincut.min_cut_value();
    let (s, t) = mincut.partition();
    Ok((value, s, t))
}
