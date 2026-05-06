//! `StructuralAudit` — one-call orchestrator that runs every analysis
//! primitive Connectome OS ships and returns a single
//! `StructuralAuditReport`.
//!
//! Application #13 from [`Connectome-OS/README.md`](../../README.md#part-3--exotic-needs-phase-2-or-phase-3-scaffolding)
//! ("Connectome-grounded AI safety auditing"). The shipped analysis
//! primitives (`Fiedler` coherence, structural / functional mincut,
//! SDPA motif retrieval, AC-5-shaped causal perturbation) answer
//! different questions individually. For a safety-auditing workflow
//! you want all four rolled up into one report that a reviewer can
//! read top-to-bottom without rebuilding the plumbing.
//!
//! That's what this module is.
//!
//! **What this is NOT:** a new analysis primitive. Every number in
//! the report comes from the existing `Analysis`, `Observer`, and
//! `LesionStudy` APIs; this module is the glue that runs them in
//! one go and formats the result. "Safety auditing" is the framing;
//! the deliverable is a reproducible report, not a safety guarantee.
//!
//! **Determinism contract:** given the same `(conn, stimulus, config)`,
//! the report is bit-identical across runs — inherited from the
//! determinism of each underlying primitive.

use crate::analysis::{Analysis, AnalysisConfig, FunctionalPartition};
use crate::connectome::Connectome;
use crate::lesion::{boundary_edges, interior_edges, CandidateCut, LesionReport, LesionStudy};
use crate::lif::{Engine, EngineConfig, Spike};
use crate::observer::{CoherenceEvent, Observer};
use crate::stimulus::Stimulus;

/// Everything a structural-audit reviewer needs in one report.
#[derive(Clone, Debug)]
pub struct StructuralAuditReport {
    /// Number of neurons in the audited connectome.
    pub n_neurons: usize,
    /// Number of synapses.
    pub n_synapses: usize,
    /// Total spikes produced by the baseline (unperturbed) run.
    pub total_spikes: u64,
    /// Coherence-collapse events emitted during the baseline run.
    pub coherence_events: Vec<CoherenceEvent>,
    /// Static-graph mincut result (AC-3a path).
    pub structural_partition: FunctionalPartition,
    /// Coactivation-weighted mincut result (AC-3b path).
    pub functional_partition: FunctionalPartition,
    /// Number of indexed motif windows in the baseline run (SDPA
    /// embedding over 20 ms rasters).
    pub motif_corpus_size: usize,
    /// Causal-perturbation summary — one measurement per candidate
    /// cut passed to `AuditConfig.candidate_cuts` (or auto-generated
    /// boundary vs interior pair if none supplied).
    pub causal: LesionReport,
}

impl StructuralAuditReport {
    /// Best-effort single-line summary for logging.
    pub fn one_line_summary(&self) -> String {
        let cut = self
            .causal
            .cuts
            .iter()
            .find(|m| m.label != self.causal.reference_label);
        let z = cut
            .and_then(|c| c.z_vs_reference)
            .map(|z| format!("{:.2}σ", z))
            .unwrap_or_else(|| "—".to_string());
        format!(
            "audit: n={} syn={} spikes={} events={} |a|={} |b|={} motifs={} z_targeted={}",
            self.n_neurons,
            self.n_synapses,
            self.total_spikes,
            self.coherence_events.len(),
            self.functional_partition.side_a.len(),
            self.functional_partition.side_b.len(),
            self.motif_corpus_size,
            z,
        )
    }
}

/// Knobs for the audit run. Defaults mirror the Tier-1 demo.
#[derive(Clone, Debug)]
pub struct AuditConfig {
    /// End of simulation in ms. Default 400.
    pub t_end_ms: f32,
    /// Maximum K boundary edges to consider for the causal cut.
    /// Default 100. Caps the scope of the perturbation so the σ
    /// measurement is repeatable.
    pub max_boundary_k: usize,
    /// Paired-trial count for the causal perturbation. Default 5
    /// (matches AC-5).
    pub trials: u32,
    /// If `Some`, use these custom cuts for the causal perturbation
    /// instead of auto-generating the boundary-vs-interior pair. The
    /// first cut whose label equals `reference_label` below becomes
    /// the σ reference.
    pub candidate_cuts: Option<Vec<CandidateCut>>,
    /// Reference-cut label. Default `"interior"` for the auto-generated
    /// boundary-vs-interior pair.
    pub reference_label: String,
    /// Analysis config for mincut + motif retrieval.
    pub analysis: AnalysisConfig,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            t_end_ms: 400.0,
            max_boundary_k: 100,
            trials: 5,
            candidate_cuts: None,
            reference_label: "interior".into(),
            analysis: AnalysisConfig::default(),
        }
    }
}

/// One-call audit runner.
///
/// ```ignore
/// use connectome_fly::*;
/// let conn = Connectome::generate(&ConnectomeConfig::default());
/// let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 250.0, 85.0, 120.0);
/// let report = StructuralAudit::new(&conn, stim).run();
/// println!("{}", report.one_line_summary());
/// ```
pub struct StructuralAudit<'a> {
    conn: &'a Connectome,
    stim: Stimulus,
    cfg: AuditConfig,
}

impl<'a> StructuralAudit<'a> {
    /// New audit with default knobs.
    pub fn new(conn: &'a Connectome, stim: Stimulus) -> Self {
        Self {
            conn,
            stim,
            cfg: AuditConfig::default(),
        }
    }

    /// Override knobs.
    pub fn with_config(mut self, cfg: AuditConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Run every primitive and build the report.
    pub fn run(&self) -> StructuralAuditReport {
        // ---- Baseline run — spikes + coherence events + motifs ----
        let mut eng = Engine::new(self.conn, EngineConfig::default());
        let mut obs = Observer::new(self.conn.num_neurons());
        eng.run_with(&self.stim, &mut obs, self.cfg.t_end_ms);
        let spikes: Vec<Spike> = obs.spikes().to_vec();
        let baseline_report = obs.finalize();
        let total_spikes = baseline_report.total_spikes;
        let coherence_events = baseline_report.coherence_events.clone();

        // ---- Analysis layer — partition + motif retrieval ----
        let an = Analysis::new(self.cfg.analysis.clone());
        let structural = an.structural_partition(self.conn);
        let functional = an.functional_partition(self.conn, &spikes);
        let (motif_index, _motif_hits) = an.retrieve_motifs(self.conn, &spikes, 5);
        let motif_corpus_size = motif_index.len();

        // ---- Causal perturbation (AC-5-shaped, reusable via LesionStudy) ----
        let candidate_cuts = match &self.cfg.candidate_cuts {
            Some(cuts) => cuts.clone(),
            None => auto_cuts(self.conn, &functional, self.cfg.max_boundary_k),
        };
        let causal = LesionStudy::new(self.conn, self.stim.clone())
            .with_trials(self.cfg.trials)
            .with_window(self.cfg.t_end_ms, self.cfg.t_end_ms - 100.0, self.cfg.t_end_ms)
            .with_reference_label(self.cfg.reference_label.clone())
            .run(&candidate_cuts);

        StructuralAuditReport {
            n_neurons: self.conn.num_neurons(),
            n_synapses: self.conn.synapses().len(),
            total_spikes,
            coherence_events,
            structural_partition: structural,
            functional_partition: functional,
            motif_corpus_size,
            causal,
        }
    }
}

/// Auto-generate a boundary-vs-interior candidate-cut pair from a
/// functional partition. Both cuts are size-matched to
/// `min(max_k, |boundary|, |interior|)` so the σ distribution has
/// comparable footprint on both arms.
fn auto_cuts(conn: &Connectome, part: &FunctionalPartition, max_k: usize) -> Vec<CandidateCut> {
    let b = boundary_edges(conn, &part.side_a);
    let i = interior_edges(conn, &part.side_a);
    let k = max_k.min(b.len()).min(i.len());
    // If there aren't enough edges on one side, we still emit both
    // cuts at whatever k is available. A degenerate k=0 is honest
    // and will show up as zero divergence in the report.
    let b_edges: Vec<usize> = b.into_iter().take(k).collect();
    let i_edges: Vec<usize> = i.into_iter().take(k).collect();
    vec![
        CandidateCut {
            label: "interior".into(),
            edges: i_edges,
        },
        CandidateCut {
            label: "boundary".into(),
            edges: b_edges,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{Connectome, ConnectomeConfig};
    use crate::stimulus::Stimulus;

    fn small_conn() -> Connectome {
        Connectome::generate(&ConnectomeConfig {
            num_neurons: 128,
            avg_out_degree: 12.0,
            ..ConnectomeConfig::default()
        })
    }

    #[test]
    fn structural_audit_populates_every_field() {
        let conn = small_conn();
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 120.0, 85.0, 120.0);
        let report = StructuralAudit::new(&conn, stim)
            .with_config(AuditConfig {
                t_end_ms: 200.0,
                trials: 2,
                ..AuditConfig::default()
            })
            .run();
        assert_eq!(report.n_neurons, conn.num_neurons());
        assert_eq!(report.n_synapses, conn.synapses().len());
        // At least one of the partitions must be non-degenerate — the
        // mincut primitive is mature enough that a small SBM can't
        // produce two totally-empty sides.
        let f_ok = !report.functional_partition.side_a.is_empty()
            && !report.functional_partition.side_b.is_empty();
        assert!(f_ok, "functional partition was degenerate on a 128-neuron SBM");
        // Causal report must have BOTH auto-generated cuts, and the
        // reference cut's z_vs_reference must be None.
        assert_eq!(report.causal.cuts.len(), 2);
        let ref_cut = report
            .causal
            .cuts
            .iter()
            .find(|m| m.label == report.causal.reference_label)
            .expect("reference cut present");
        assert!(ref_cut.z_vs_reference.is_none());
        // one_line_summary must be non-empty and contain the spike count.
        let summary = report.one_line_summary();
        assert!(summary.contains(&report.total_spikes.to_string()));
    }

    #[test]
    fn structural_audit_is_deterministic() {
        let conn = small_conn();
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 120.0, 85.0, 120.0);
        let cfg = AuditConfig {
            t_end_ms: 200.0,
            trials: 2,
            ..AuditConfig::default()
        };
        let a = StructuralAudit::new(&conn, stim.clone())
            .with_config(cfg.clone())
            .run();
        let b = StructuralAudit::new(&conn, stim).with_config(cfg).run();
        assert_eq!(a.total_spikes, b.total_spikes);
        assert_eq!(a.n_neurons, b.n_neurons);
        assert_eq!(a.n_synapses, b.n_synapses);
        assert_eq!(a.motif_corpus_size, b.motif_corpus_size);
        assert_eq!(a.coherence_events.len(), b.coherence_events.len());
        assert_eq!(a.structural_partition.side_a, b.structural_partition.side_a);
        assert_eq!(a.functional_partition.side_a, b.functional_partition.side_a);
        for (x, y) in a.causal.cuts.iter().zip(b.causal.cuts.iter()) {
            assert_eq!(x.label, y.label);
            assert_eq!(
                x.mean_divergence_hz.to_bits(),
                y.mean_divergence_hz.to_bits()
            );
        }
    }
}
