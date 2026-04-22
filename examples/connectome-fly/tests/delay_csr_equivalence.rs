//! Opt D (delay-sorted CSR) equivalence test.
//!
//! The delay-sorted CSR reorders intra-row synapse pushes into the
//! timing wheel by delay. Because the wheel stores events within a
//! bucket in push-order, the new path does NOT produce a bit-exact
//! spike trace vs the insertion-order CSR — it produces a different
//! tie-break within a bucket for the rare case of two events with
//! identical `(t_ms, post)` landing in the same bucket from a single
//! pre-synaptic spike.
//!
//! ADR-154 §15.1 explicitly excludes cross-path bit-exactness from the
//! determinism contract, and README §Determinism documents the cross-
//! path tolerance as ~10 %. This test asserts that the delay-sorted
//! path stays inside that envelope on the saturated-regime `N=1024,
//! t_end=120ms` workload used by `lif_throughput_n_1024`.

use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig, Observer, Stimulus};

/// The saturated-regime reference workload — identical to
/// `benches/lif_throughput.rs::lif_throughput_n_1024` and
/// `benches/delay_csr.rs` so the equivalence claim sits on the same
/// workload as the speedup claim.
fn run_total_spikes(use_delay_sorted_csr: bool) -> u64 {
    let cfg = ConnectomeConfig {
        num_neurons: 1024,
        avg_out_degree: 48.0,
        seed: 0x51FE_D0FF_CAFE_BABE,
        ..ConnectomeConfig::default()
    };
    let conn = Connectome::generate(&cfg);
    let t_end_ms: f32 = 120.0;
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 10.0, t_end_ms - 20.0, 80.0, 100.0);
    let mut eng = Engine::new(
        &conn,
        EngineConfig {
            use_optimized: true,
            use_delay_sorted_csr,
            ..EngineConfig::default()
        },
    );
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, t_end_ms);
    obs.finalize().total_spikes
}

#[test]
fn delay_csr_spike_count_within_cross_path_tolerance() {
    // scalar-opt baseline: wheel + SoA, CSR in insertion order.
    let a = run_total_spikes(false);
    // Opt D: wheel + SoA + delay-sorted SoA CSR for spike delivery.
    let b = run_total_spikes(true);
    assert!(
        a > 0,
        "scalar-opt produced zero spikes — test is not exercising the kernel"
    );
    assert!(
        b > 0,
        "delay-csr path produced zero spikes — delivery path is broken"
    );
    let lo = a.min(b) as f64;
    let hi = a.max(b) as f64;
    let rel = (hi - lo) / lo;
    eprintln!(
        "delay_csr equivalence: scalar-opt={a} spikes, delay-csr={b} spikes, rel-gap={rel:.4} \
         (tolerance=0.10, per README §Determinism)"
    );
    // 10 % is the cross-path tolerance the demonstrator already documents
    // (README §Determinism; ADR-154 §15.1). Bit-exactness is NOT claimed.
    assert!(
        rel <= 0.10,
        "delay_csr equivalence: spike-count gap {rel:.4} exceeds 10 % cross-path tolerance \
         (scalar-opt={a}, delay-csr={b})"
    );
}

#[test]
fn delay_csr_repeatability_within_path() {
    // Within-path bit-exactness is still required: two runs of the
    // delay-sorted path on the same `(connectome_seed, engine_seed)`
    // must produce identical total spike counts.
    let x = run_total_spikes(true);
    let y = run_total_spikes(true);
    assert_eq!(
        x, y,
        "delay_csr within-path repeatability failed: {x} vs {y}"
    );
}
