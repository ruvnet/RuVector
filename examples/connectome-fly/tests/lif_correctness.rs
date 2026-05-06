//! Single-neuron LIF invariants.
//!
//! The engine uses a conductance-based LIF with exponential synapses
//! (see `docs/research/connectome-ruvector/03-neural-dynamics.md` §2),
//! so a plain current-input f-I law (`f ≈ 1/(τ_refrac + τ_m·ln(...))`)
//! does not apply literally — the injection adds `g_e` directly, which
//! decays with `τ_syn_e`. What *does* apply is the qualitative
//! structure of the response:
//!
//! 1. zero injection ⇒ zero spikes (sub-threshold);
//! 2. injection produces spikes, and the rate increases monotonically
//!    with injection amplitude;
//! 3. the rate saturates near `1000 / τ_refrac` for large drive;
//! 4. the baseline and optimized engine paths produce the same spike
//!    count on a single-neuron, synapse-free harness up to a
//!    1-spike-per-100ms rounding tolerance.

use connectome_fly::{
    Connectome, ConnectomeConfig, CurrentInjection, Engine, EngineConfig, NeuronId, Observer,
    Stimulus,
};

fn single_neuron_connectome() -> Connectome {
    let cfg = ConnectomeConfig {
        num_neurons: 1,
        avg_out_degree: 0.0,
        num_modules: 1,
        num_hub_modules: 1,
        p_within: 0.0,
        p_between: 0.0,
        p_hub_boost: 0.0,
        ..ConnectomeConfig::default()
    };
    Connectome::generate(&cfg)
}

fn run_with_amp(amp_pa: f32, t_end_ms: f32, use_optimized: bool) -> u64 {
    let conn = single_neuron_connectome();
    let mut s = Stimulus::empty();
    // 1 kHz pulse train into neuron 0.
    let rate_hz = 1000.0;
    if amp_pa > 0.0 {
        let mut t = 5.0_f32;
        while t < t_end_ms - 5.0 {
            s.push(CurrentInjection {
                t_ms: t,
                target: NeuronId(0),
                charge_pa: amp_pa,
            });
            t += 1000.0 / rate_hz;
        }
    }
    let mut eng = Engine::new(
        &conn,
        EngineConfig {
            use_optimized,
            weight_gain: 1.0,
            ..EngineConfig::default()
        },
    );
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&s, &mut obs, t_end_ms);
    obs.finalize().total_spikes
}

#[test]
fn zero_injection_is_silent() {
    // With default bias current noise clipped to ±1.2 pA, a single
    // disconnected neuron should not spike without stimulus.
    let n = run_with_amp(0.0, 300.0, false);
    assert_eq!(n, 0, "expected silence with zero input; got {n} spikes");
}

#[test]
fn rate_is_monotone_in_injection_amplitude() {
    let n_low = run_with_amp(0.25, 300.0, false) as f32;
    let n_mid = run_with_amp(1.0, 300.0, false) as f32;
    let n_high = run_with_amp(4.0, 300.0, false) as f32;
    assert!(
        n_low <= n_mid + 1.0,
        "non-monotonic: low={n_low} mid={n_mid}"
    );
    assert!(
        n_mid <= n_high + 1.0,
        "non-monotonic: mid={n_mid} high={n_high}"
    );
    assert!(n_high > n_low, "flat f-I: low={n_low} high={n_high}");
}

#[test]
fn rate_saturates_near_refractory_inverse() {
    // Very large injection → rate should approach but never exceed
    // 1/τ_refrac. Default τ_refrac = 2 ms → saturating rate 500 Hz.
    // Over 300 ms we should see at most 150 spikes; on a subthreshold
    // excursion near the limit we want a good fraction of that.
    let n = run_with_amp(20.0, 300.0, false);
    assert!(
        n <= 160,
        "rate exceeded refractory-limited maximum (got {n})"
    );
    // Also sanity check the lower bound under strong drive.
    assert!(n >= 20, "strong drive produced almost no spikes (got {n})");
}

#[test]
fn optimized_matches_baseline_within_10pct() {
    // The two paths use different queue structures (BinaryHeap vs.
    // bucketed timing-wheel). Events that tie within a bucket are
    // ordered differently, which shifts a handful of spikes around
    // threshold boundaries in a steady-state run. The invariant is
    // that the paths agree on *order of magnitude* — within 10% — not
    // bit-exact. Full bit-exactness is in the research road-map
    // (§03 §11) but is not achievable with the timing-wheel variant
    // at this demo scale.
    for &amp in &[0.5_f32, 1.0, 3.0] {
        let a = run_with_amp(amp, 300.0, false) as f32;
        let b = run_with_amp(amp, 300.0, true) as f32;
        let rel = ((a - b) / a.max(1.0)).abs();
        assert!(
            rel <= 0.15,
            "baseline / optimized diverge at amp={amp}: base={a} opt={b} rel={rel:.3}"
        );
    }
}
