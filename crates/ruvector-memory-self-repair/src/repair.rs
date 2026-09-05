//! Ties the reused `ruvector-coherence` spectral health monitor to a
//! bounded, witness-logged repair action on the memory graph.
//!
//! The spectral composite score (Fiedler value, spectral gap, effective
//! resistance, degree regularity — all computed by
//! `ruvector_coherence::spectral::HnswHealthMonitor`, unmodified) decides
//! *whether* to repair. The Fiedler eigenvector — nodes near zero sit on or
//! near a weak graph cut — decides *where*: repair targets are the alive
//! nodes with the smallest `|fiedler_vec[i]|`, i.e. the ones structurally
//! closest to falling into a disconnected fragment. Repair itself only adds
//! edges to a target's current top-k semantic neighbors (recomputed from
//! the embeddings that never decay), bounded to a fixed node/edge budget
//! per event so it can never degrade into "just rebuild everything".

use crate::graph::MemoryGraph;
use crate::witness::WitnessLog;
use ruvector_coherence::spectral::{
    estimate_fiedler, CsrMatrixView, HnswHealthMonitor, SpectralConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    NoRepair,
    MonitorOnly,
    SpectralRepair,
}

#[derive(Debug, Clone)]
pub struct RepairConfig {
    pub max_nodes_per_repair: usize,
    pub semantic_k_reconnect: usize,
    /// Repair triggers once `check_health()` returns at least this many
    /// simultaneous alerts (2 is also where `HnswHealthMonitor` itself
    /// emits `RebuildRecommended`, so this matches its own judgment of
    /// "seriously degraded", not an arbitrarily looser bar).
    pub alert_repair_min_issues: usize,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_nodes_per_repair: 12,
            semantic_k_reconnect: 3,
            alert_repair_min_issues: 2,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub scs_history: Vec<(u64, f64)>,
    pub alert_steps: Vec<u64>,
    pub repairs: Vec<(u64, usize, usize)>, // (step, nodes_touched, edges_added)
    pub cumulative_edges_added: usize,
    pub monitor_calls: usize,
}

pub struct Runner {
    monitor: HnswHealthMonitor,
    cfg: RepairConfig,
    pub witness: WitnessLog,
    pub stats: RunStats,
}

impl Runner {
    pub fn new(cfg: RepairConfig) -> Self {
        Self {
            monitor: HnswHealthMonitor::new(SpectralConfig::default()),
            cfg,
            witness: WitnessLog::new(),
            stats: RunStats::default(),
        }
    }

    /// Runs a full spectral health check against the graph's current alive
    /// subgraph and, for `Variant::SpectralRepair` only, performs a bounded
    /// targeted repair if the alert count crosses the configured threshold.
    /// No-ops entirely for `Variant::NoRepair` (the baseline never pays
    /// monitoring cost); `Variant::MonitorOnly` checks and records but never
    /// mutates the graph, which is what makes it useful as a detector
    /// sanity-check independent of repair effects.
    pub fn check_and_repair(&mut self, graph: &mut MemoryGraph, step: u64, variant: Variant) {
        if variant == Variant::NoRepair {
            return;
        }
        let (n, edges, alive_map) = graph.to_laplacian_edges();
        if n < 3 {
            return;
        }
        let lap = CsrMatrixView::build_laplacian(n, &edges);
        self.monitor.update(&lap, None);
        self.stats.monitor_calls += 1;
        let alerts = self.monitor.check_health();
        let scs_before = self.monitor.score().composite;
        self.stats.scs_history.push((step, scs_before));
        if !alerts.is_empty() {
            self.stats.alert_steps.push(step);
        }

        if variant == Variant::SpectralRepair && alerts.len() >= self.cfg.alert_repair_min_issues {
            self.perform_repair(graph, &lap, &alive_map, step, scs_before);
        }
    }

    fn perform_repair(
        &mut self,
        graph: &mut MemoryGraph,
        lap: &CsrMatrixView,
        alive_map: &[usize],
        step: u64,
        scs_before: f64,
    ) {
        let (_eigenvalue, fiedler_vec) = estimate_fiedler(lap, 50, 1e-6);
        let mut order: Vec<usize> = (0..fiedler_vec.len()).collect();
        order.sort_by(|&a, &b| fiedler_vec[a].abs().total_cmp(&fiedler_vec[b].abs()));

        let mut nodes_touched = Vec::new();
        let mut edges_added = 0usize;
        for &local in order.iter().take(self.cfg.max_nodes_per_repair) {
            let id = alive_map[local];
            let neighbors = graph.semantic_topk(id, self.cfg.semantic_k_reconnect);
            let mut touched = false;
            for (nbr, sim) in neighbors {
                if sim <= 0.0 || graph.adjacency[id].contains_key(&nbr) {
                    continue;
                }
                graph.add_edge(id, nbr, sim);
                edges_added += 1;
                touched = true;
            }
            if touched {
                nodes_touched.push(id);
            }
        }

        let (n2, edges2, _) = graph.to_laplacian_edges();
        let lap2 = CsrMatrixView::build_laplacian(n2, &edges2);
        self.monitor.update(&lap2, None);
        let scs_after = self.monitor.score().composite;

        self.witness.record(
            step,
            nodes_touched.clone(),
            edges_added,
            0,
            scs_before,
            scs_after,
        );
        self.stats
            .repairs
            .push((step, nodes_touched.len(), edges_added));
        self.stats.cumulative_edges_added += edges_added;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drift::{apply_arrival, build_arrivals, SimConfig};

    #[test]
    fn repair_budget_never_exceeds_configured_cap_per_event() {
        let cfg = SimConfig {
            total_steps: 200,
            ..Default::default()
        };
        let (_, arrivals) = build_arrivals(&cfg);
        let mut graph = MemoryGraph::new();
        for (step, e) in arrivals.into_iter().enumerate() {
            apply_arrival(&mut graph, e, step as u64, &cfg);
        }
        let rcfg = RepairConfig {
            max_nodes_per_repair: 5,
            semantic_k_reconnect: 2,
            ..Default::default()
        };
        let cap = rcfg.max_nodes_per_repair * rcfg.semantic_k_reconnect;
        let mut runner = Runner::new(rcfg);
        runner.check_and_repair(&mut graph, 200, Variant::SpectralRepair);
        for &(_, nodes, edges) in &runner.stats.repairs {
            assert!(nodes <= 5);
            assert!(edges <= cap);
        }
    }

    #[test]
    fn no_repair_variant_never_mutates_graph_edge_count() {
        let cfg = SimConfig {
            total_steps: 100,
            ..Default::default()
        };
        let (_, arrivals) = build_arrivals(&cfg);
        let mut graph = MemoryGraph::new();
        for (step, e) in arrivals.into_iter().enumerate() {
            apply_arrival(&mut graph, e, step as u64, &cfg);
        }
        let before = graph.edge_count();
        let mut runner = Runner::new(RepairConfig::default());
        runner.check_and_repair(&mut graph, 100, Variant::NoRepair);
        assert_eq!(before, graph.edge_count());
        assert_eq!(runner.stats.monitor_calls, 0);
    }
}
