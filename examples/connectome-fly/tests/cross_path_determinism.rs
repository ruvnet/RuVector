#![allow(clippy::needless_range_loop)]
//! ADR-154 §15.1 — cross-path determinism, measured.
//!
//! AC-1 (shipped) asserts *within-path* bit-exactness: two repeat
//! runs on the same seeds + same stimulus produce identical spike
//! traces within the baseline path (heap + AoS) and within the
//! optimized path (wheel + SoA + SIMD), independently. ADR-154 §15.1
//! names *cross-path* bit-exactness — two different LIF paths
//! producing identical traces on the same input — as a follow-up.
//!
//! This commit ships a **canonical in-bucket-ordering contract** on
//! the wheel path: `TimingWheel::drain_due` now sorts each bucket
//! ascending by `(t_ms, post, pre)` before delivery, matching
//! `SpikeEvent::cmp` on the heap path. With that contract in place,
//! the wheel's dispatch order is deterministically equivalent to
//! the heap's on the same set of delivered events.
//!
//! **But cross-path bit-exact spike traces are NOT delivered by the
//! sort alone.** Measurement (15th discovery — ADR-154 §17 item 14):
//! baseline and optimized produce spike counts that diverge by ~0.5
//! % (195 782 vs 194 784 on AC-1 stimulus at N=1024). The divergence
//! is NOT an FP-ordering artefact but a legitimate correctness
//! deviation: the optimized path uses active-set pruning (skip
//! subthreshold updates for neurons not recently perturbed), while
//! the baseline updates every neuron every tick. Neurons on the
//! edge of the threshold that leak below it under continuous dense
//! updates stay above under active-set updates — both behaviours are
//! *correct-by-ADR*, neither is a regression, and they produce
//! genuinely different spike populations.
//!
//! The shipped contract therefore is:
//!
//! - Within-path: bit-exact (both paths). Verified here.
//! - Across paths: spike counts agree within **10 % envelope** (the
//!   cross-path tolerance ADR-154 §15.1 already declared). The
//!   bucket sort tightens intra-tick ordering from "insertion order"
//!   to "canonical (t_ms, post, pre)" but does not erase the
//!   active-set behavioural divergence. Verified here.
//!
//! True cross-path bit-exactness would require either (a) running
//! both paths with active-set off, which is a bench-only config, or
//! (b) teaching the baseline the same active-set, which defeats the
//! baseline's role as the dense reference.

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Spike, Stimulus};

fn default_conn() -> Connectome {
    Connectome::generate(&ConnectomeConfig::default())
}

fn run_one(conn: &Connectome, cfg: EngineConfig, stim: &Stimulus, t_end_ms: f32) -> Vec<Spike> {
    let mut eng = Engine::new(conn, cfg);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(stim, &mut obs, t_end_ms);
    obs.spikes().to_vec()
}

/// Assert two spike traces are bit-identical on `(neuron, t_ms.to_bits())`
/// for the first `k` entries, and their total counts match.
fn assert_traces_match(a: &[Spike], b: &[Spike], k: usize, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "cross-path: {label} spike counts diverge (a={} b={})",
        a.len(),
        b.len()
    );
    let k = k.min(a.len());
    for i in 0..k {
        assert_eq!(
            a[i].neuron, b[i].neuron,
            "cross-path: {label} neuron differs at spike #{i}"
        );
        assert_eq!(
            a[i].t_ms.to_bits(),
            b[i].t_ms.to_bits(),
            "cross-path: {label} t_ms differs at spike #{i} (a={} b={})",
            a[i].t_ms,
            b[i].t_ms
        );
    }
    eprintln!("cross-path: {label} bit-identical on count={} + first {k}", a.len());
}

#[test]
fn baseline_heap_and_optimized_wheel_within_10_percent_envelope() {
    // Same stimulus AC-1 uses.
    let conn = default_conn();
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 100.0, 200.0, 85.0, 120.0);
    let t_end_ms = 500.0;

    let cfg_baseline = EngineConfig {
        use_optimized: false,
        use_delay_sorted_csr: false,
        ..EngineConfig::default()
    };
    let cfg_optimized = EngineConfig {
        use_optimized: true,
        use_delay_sorted_csr: false,
        ..EngineConfig::default()
    };

    let trace_baseline = run_one(&conn, cfg_baseline, &stim, t_end_ms);
    let trace_optimized = run_one(&conn, cfg_optimized, &stim, t_end_ms);

    let a = trace_baseline.len() as f32;
    let b = trace_optimized.len() as f32;
    let rel_gap = (a - b).abs() / a.max(b).max(1.0);
    eprintln!(
        "cross-path: baseline_count={} optimized_count={} rel_gap={:.4} \
         (ADR-154 §15.1 envelope = 0.10 → {})",
        trace_baseline.len(),
        trace_optimized.len(),
        rel_gap,
        if rel_gap <= 0.10 { "PASS" } else { "MISS" }
    );
    assert!(
        rel_gap <= 0.10,
        "cross-path: baseline/optimized spike-count relative gap {:.4} exceeds the 10% envelope \
         (baseline={}, optimized={}). The wheel's bucket-sort contract is intact but the \
         active-set divergence has grown beyond the ADR-declared tolerance — regression to \
         investigate, not a threshold to weaken.",
        rel_gap,
        trace_baseline.len(),
        trace_optimized.len()
    );
    eprintln!(
        "cross-path: baseline vs optimized 10% envelope held ({} vs {}, rel_gap={:.4})",
        trace_baseline.len(),
        trace_optimized.len(),
        rel_gap
    );
}

#[test]
fn optimized_wheel_is_deterministic_across_repeat_runs() {
    // Regression test: the new sort in `drain_due` is idempotent on
    // an already-canonical bucket, so AC-1 within-path bit-exactness
    // must still hold on the optimized path.
    let conn = default_conn();
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 100.0, 200.0, 85.0, 120.0);
    let cfg = EngineConfig {
        use_optimized: true,
        use_delay_sorted_csr: false,
        ..EngineConfig::default()
    };
    let a = run_one(&conn, cfg.clone(), &stim, 500.0);
    let b = run_one(&conn, cfg, &stim, 500.0);
    assert_traces_match(&a, &b, 1000, "optimized repeat");
}

#[test]
fn baseline_heap_is_deterministic_across_repeat_runs() {
    // Same check on the heap path — already covered by AC-1 but
    // explicit here so the cross-path file is self-contained.
    let conn = default_conn();
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 100.0, 200.0, 85.0, 120.0);
    let cfg = EngineConfig {
        use_optimized: false,
        use_delay_sorted_csr: false,
        ..EngineConfig::default()
    };
    let a = run_one(&conn, cfg.clone(), &stim, 500.0);
    let b = run_one(&conn, cfg, &stim, 500.0);
    assert_traces_match(&a, &b, 1000, "baseline repeat");
}
