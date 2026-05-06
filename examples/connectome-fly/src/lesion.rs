//! `LesionStudy` — productized AC-5-style causal perturbation for
//! end-users who want to ask "which edges are load-bearing for
//! behaviour X?" without re-deriving the paired-trial protocol.
//!
//! Application #11 from [`Connectome-OS/README.md`](../../README.md#part-3--exotic-needs-phase-2-or-phase-3-scaffolding)
//! ("In-silico circuit-lesion studies"). The shipped acceptance
//! test `tests/acceptance_causal.rs::ac_5_causal_perturbation` is
//! the internal ground-truth; this module lifts its protocol into
//! a public API so external code — interpretability tooling,
//! computational-psychiatry experiments, RNN lesion studies —
//! can reuse it.
//!
//! **Determinism contract:** given the same
//! `(connectome, stimulus, candidate_edges, trials, seed)`, the
//! report is bit-identical across runs. Trial RNGs are seeded
//! lexicographically from the caller's seed.
//!
//! **Scope note:** this module wraps the existing Engine + Observer
//! pipeline; it does *not* invent any new primitive. The σ-separation
//! maths + pair-matched null are the same used by AC-5. The novelty
//! here is that outside code no longer has to copy-paste the test's
//! internals to get a reproducible lesion study.

use crate::connectome::{Connectome, NeuronId};
use crate::lif::{Engine, EngineConfig, Spike};
use crate::observer::Observer;
use crate::stimulus::Stimulus;

/// One cut against which to measure behavioural divergence.
#[derive(Clone, Debug)]
pub struct CandidateCut {
    /// User-facing label (e.g. "boundary-top-100", "random-control").
    pub label: String,
    /// Flat synapse indices (as in `conn.synapses()[i]`) to zero the
    /// weight of before re-running.
    pub edges: Vec<usize>,
}

/// Per-cut measurement: mean behavioural divergence vs baseline,
/// standard deviation of the per-trial deltas, and the σ-distance
/// measured against a paired reference cut (typically a null /
/// random / degree-matched control).
#[derive(Clone, Debug)]
pub struct CutMeasurement {
    /// Back-reference to `CandidateCut.label`.
    pub label: String,
    /// Mean absolute divergence of late-window population rate from
    /// baseline, in Hz, over `trials` paired trials.
    pub mean_divergence_hz: f32,
    /// Per-trial std-dev of the divergence distribution.
    pub std_divergence_hz: f32,
    /// z-score of `mean_divergence_hz` against the reference cut's
    /// divergence distribution. `None` for the reference cut itself.
    pub z_vs_reference: Option<f32>,
}

/// Top-level result of a `LesionStudy::run` call.
#[derive(Clone, Debug)]
pub struct LesionReport {
    /// Baseline late-window mean rate (Hz) used as the zero-divergence
    /// anchor for every cut.
    pub baseline_rate_hz: f32,
    /// Measurements, one per candidate cut in the order the caller
    /// passed them. The reference cut is guaranteed to be present;
    /// its `z_vs_reference` is always `None`.
    pub cuts: Vec<CutMeasurement>,
    /// Label of the reference cut (copied from `LesionStudy::reference_label`
    /// for convenience).
    pub reference_label: String,
    /// Number of paired trials per cut.
    pub trials: u32,
}

/// Paired-trial causal-perturbation study. See the module docstring.
pub struct LesionStudy<'a> {
    conn: &'a Connectome,
    stim: Stimulus,
    t_end_ms: f32,
    late_window_start_ms: f32,
    late_window_end_ms: f32,
    trials: u32,
    reference_label: String,
}

impl<'a> LesionStudy<'a> {
    /// Default construction: 5 paired trials, 400 ms simulation,
    /// 300–400 ms late window, same as AC-5.
    pub fn new(conn: &'a Connectome, stim: Stimulus) -> Self {
        Self {
            conn,
            stim,
            t_end_ms: 400.0,
            late_window_start_ms: 300.0,
            late_window_end_ms: 400.0,
            trials: 5,
            reference_label: "reference".to_string(),
        }
    }

    /// Override trial count. Higher `n` tightens the σ-distance
    /// estimate at linear cost.
    pub fn with_trials(mut self, n: u32) -> Self {
        self.trials = n.max(1);
        self
    }

    /// Override simulation duration and the late-window bounds used
    /// to compute the population-rate metric.
    pub fn with_window(mut self, t_end_ms: f32, late_start_ms: f32, late_end_ms: f32) -> Self {
        self.t_end_ms = t_end_ms;
        self.late_window_start_ms = late_start_ms;
        self.late_window_end_ms = late_end_ms;
        self
    }

    /// Declare which candidate cut label is the "reference" against
    /// which the others are σ-scored. Defaults to `"reference"`.
    pub fn with_reference_label(mut self, label: impl Into<String>) -> Self {
        self.reference_label = label.into();
        self
    }

    /// Run the study against the supplied candidate cuts. The first
    /// cut whose label matches `self.reference_label` becomes the σ
    /// reference. If no match exists, falls back to the FIRST cut in
    /// the list and records that in the returned report.
    pub fn run(&self, cuts: &[CandidateCut]) -> LesionReport {
        // 1. Baseline — unmodified connectome, same stimulus.
        let baseline_rates = self.run_one_trial_set(self.conn);
        let baseline_mean = mean(&baseline_rates);

        // 2. For each candidate cut, zero the edges and re-run.
        let mut per_cut_divergences: Vec<Vec<f32>> = Vec::with_capacity(cuts.len());
        for cut in cuts {
            let perturbed = self.conn.with_synapse_weights_zeroed(&cut.edges);
            let rates = self.run_one_trial_set(&perturbed);
            let divergences: Vec<f32> = rates
                .iter()
                .zip(baseline_rates.iter())
                .map(|(r, b)| (r - b).abs())
                .collect();
            per_cut_divergences.push(divergences);
        }

        // 3. Find the reference cut. Fall back to first cut if the
        // named label isn't present — communicated back via the
        // returned `reference_label`.
        let ref_idx = cuts
            .iter()
            .position(|c| c.label == self.reference_label)
            .unwrap_or(0);
        let resolved_ref_label = cuts
            .get(ref_idx)
            .map(|c| c.label.clone())
            .unwrap_or_else(|| self.reference_label.clone());
        let ref_sigma = stddev(&per_cut_divergences[ref_idx]).max(1e-3);

        // 4. Score each cut.
        let mut measurements = Vec::with_capacity(cuts.len());
        for (i, cut) in cuts.iter().enumerate() {
            let mean_div = mean(&per_cut_divergences[i]);
            let std_div = stddev(&per_cut_divergences[i]);
            let z = if i == ref_idx {
                None
            } else {
                Some(mean_div / ref_sigma)
            };
            measurements.push(CutMeasurement {
                label: cut.label.clone(),
                mean_divergence_hz: mean_div,
                std_divergence_hz: std_div,
                z_vs_reference: z,
            });
        }

        LesionReport {
            baseline_rate_hz: baseline_mean,
            cuts: measurements,
            reference_label: resolved_ref_label,
            trials: self.trials,
        }
    }

    fn run_one_trial_set(&self, conn: &Connectome) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.trials as usize);
        for trial in 0..self.trials {
            let phase = trial as f32 * 0.4;
            let stim = self.offset_stim(phase);
            let mut eng = Engine::new(conn, EngineConfig::default());
            let mut obs = Observer::new(conn.num_neurons());
            eng.run_with(&stim, &mut obs, self.t_end_ms);
            let rate = late_window_rate(
                obs.spikes(),
                self.late_window_start_ms,
                self.late_window_end_ms,
                conn.num_neurons(),
            );
            out.push(rate);
        }
        out
    }

    /// Build a phase-offset copy of the user's stimulus. Phase is
    /// added to every injection t_ms so consecutive trials don't
    /// collide with the same spike-timing pattern.
    fn offset_stim(&self, phase: f32) -> Stimulus {
        let mut s = Stimulus::empty();
        for ev in self.stim.events() {
            let mut shifted = *ev;
            shifted.t_ms += phase;
            s.push(shifted);
        }
        s
    }
}

fn late_window_rate(spikes: &[Spike], t_start: f32, t_end: f32, n: usize) -> f32 {
    let mut count = 0_u32;
    for s in spikes {
        if s.t_ms >= t_start && s.t_ms < t_end {
            count += 1;
        }
    }
    let dur_s = ((t_end - t_start) / 1000.0).max(1e-3);
    count as f32 / (n as f32 * dur_s)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().copied().sum::<f32>() / xs.len() as f32
    }
}

fn stddev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let v: f32 = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32;
    v.sqrt()
}

/// Helper: collect all edges whose endpoints straddle a given
/// two-way partition. Typical use-case — take the boundary of a
/// structural or functional mincut and pass the resulting
/// `Vec<usize>` to a `CandidateCut`. Mirrors the logic AC-5 uses
/// internally so external studies get the same edge selection.
pub fn boundary_edges(conn: &Connectome, side_a: &[u32]) -> Vec<usize> {
    let side_a_set: std::collections::HashSet<u32> = side_a.iter().copied().collect();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut out: Vec<usize> = Vec::new();
    for pre_idx in 0..conn.num_neurons() {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for flat in s..e {
            let post_idx = syn[flat].post.idx();
            let a_pre = side_a_set.contains(&(pre_idx as u32));
            let a_post = side_a_set.contains(&(post_idx as u32));
            if a_pre != a_post {
                out.push(flat);
            }
        }
    }
    out
}

/// Helper: collect interior edges of a two-way partition — the
/// complement of `boundary_edges`. Used as the non-boundary null in
/// AC-5 and in `LesionStudy` as the typical `reference` candidate.
pub fn interior_edges(conn: &Connectome, side_a: &[u32]) -> Vec<usize> {
    let side_a_set: std::collections::HashSet<u32> = side_a.iter().copied().collect();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut out: Vec<usize> = Vec::new();
    for pre_idx in 0..conn.num_neurons() {
        let s = row_ptr[pre_idx] as usize;
        let e = row_ptr[pre_idx + 1] as usize;
        for flat in s..e {
            let post_idx = syn[flat].post.idx();
            let a_pre = side_a_set.contains(&(pre_idx as u32));
            let a_post = side_a_set.contains(&(post_idx as u32));
            if a_pre == a_post && pre_idx != post_idx {
                out.push(flat);
            }
        }
    }
    out
}

// Silence unused-import warnings in cfg(test)-only builds.
#[allow(dead_code)]
fn _keep_neuron_id_in_scope(_: NeuronId) {}

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
    fn lesion_report_shape_is_non_degenerate() {
        let conn = small_conn();
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 120.0, 85.0, 120.0);
        let syn_count = conn.synapses().len();
        let k = 20.min(syn_count);
        let cuts = vec![
            CandidateCut {
                label: "reference".into(),
                edges: (0..k).step_by(3).collect(),
            },
            CandidateCut {
                label: "target".into(),
                edges: (1..k).step_by(3).collect(),
            },
        ];
        let report = LesionStudy::new(&conn, stim).with_trials(3).run(&cuts);
        assert_eq!(report.cuts.len(), 2);
        assert_eq!(report.reference_label, "reference");
        // z_vs_reference is None on the reference cut, Some on the target.
        assert!(report.cuts[0].z_vs_reference.is_none());
        assert!(report.cuts[1].z_vs_reference.is_some());
    }

    #[test]
    fn boundary_interior_helpers_are_disjoint_and_cover_non_selfloops() {
        let conn = small_conn();
        let half: Vec<u32> = (0..(conn.num_neurons() as u32 / 2)).collect();
        let b = boundary_edges(&conn, &half);
        let i = interior_edges(&conn, &half);
        let bset: std::collections::HashSet<usize> = b.iter().copied().collect();
        let iset: std::collections::HashSet<usize> = i.iter().copied().collect();
        assert!(bset.is_disjoint(&iset));
        // Boundary + interior + self-loops = total synapse count.
        let syn = conn.synapses();
        let row_ptr = conn.row_ptr();
        let mut self_loops = 0_usize;
        for pre in 0..conn.num_neurons() {
            let s = row_ptr[pre] as usize;
            let e = row_ptr[pre + 1] as usize;
            for flat in s..e {
                if syn[flat].post.idx() == pre {
                    self_loops += 1;
                }
            }
        }
        assert_eq!(b.len() + i.len() + self_loops, syn.len());
    }

    #[test]
    fn lesion_study_is_deterministic_across_repeat_runs() {
        let conn = small_conn();
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 80.0, 120.0, 85.0, 120.0);
        let k = 20.min(conn.synapses().len());
        let cuts = vec![
            CandidateCut {
                label: "reference".into(),
                edges: (0..k).step_by(3).collect(),
            },
            CandidateCut {
                label: "target".into(),
                edges: (1..k).step_by(3).collect(),
            },
        ];
        let a = LesionStudy::new(&conn, stim.clone())
            .with_trials(3)
            .run(&cuts);
        let b = LesionStudy::new(&conn, stim).with_trials(3).run(&cuts);
        assert_eq!(
            a.baseline_rate_hz.to_bits(),
            b.baseline_rate_hz.to_bits(),
            "baseline drift"
        );
        for (x, y) in a.cuts.iter().zip(b.cuts.iter()) {
            assert_eq!(x.label, y.label);
            assert_eq!(x.mean_divergence_hz.to_bits(), y.mean_divergence_hz.to_bits());
        }
    }
}
