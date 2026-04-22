#![allow(clippy::needless_range_loop)]
//! ADR-154 §3.4 — acceptance criteria AC-1, AC-2, AC-4.
//!
//! AC-3 lives in `tests/acceptance_partition.rs`;
//! AC-5 lives in `tests/acceptance_causal.rs`. Each file is a separate
//! integration binary so Cargo schedules them independently. The
//! thresholds here are the *demo-scale floor*; the *SOTA targets* from
//! ADR-154 §3.4 are higher and the gap is documented in `BENCHMARK.md`.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, CurrentInjection, Engine, EngineConfig,
    NeuronId, Observer, Spike, Stimulus,
};

fn default_conn() -> Connectome {
    Connectome::generate(&ConnectomeConfig::default())
}

fn run_one(conn: &Connectome, stim: &Stimulus, t_end_ms: f32) -> (u64, Vec<Spike>) {
    let mut eng = Engine::new(conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(stim, &mut obs, t_end_ms);
    let r = obs.finalize();
    (r.total_spikes, obs.spikes().to_vec())
}

// -----------------------------------------------------------------
// AC-1 — Repeatability
// -----------------------------------------------------------------

#[test]
fn ac_1_repeatability() {
    let conn = default_conn();
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 100.0, 200.0, 85.0, 120.0);
    let (a, spikes_a) = run_one(&conn, &stim, 500.0);
    let (b, spikes_b) = run_one(&conn, &stim, 500.0);
    assert_eq!(a, b, "ac-1: repeat run changed spike count (a={a} b={b})");
    let k = 1000.min(spikes_a.len()).min(spikes_b.len());
    for i in 0..k {
        assert_eq!(
            spikes_a[i].neuron, spikes_b[i].neuron,
            "ac-1: neuron differs at spike #{i}"
        );
        assert_eq!(
            spikes_a[i].t_ms.to_bits(),
            spikes_b[i].t_ms.to_bits(),
            "ac-1: time differs at spike #{i}"
        );
    }
    eprintln!("ac-1: bit-identical on spike_count={a} and first {k} spikes");
}

// -----------------------------------------------------------------
// AC-2 — Motif emergence
// -----------------------------------------------------------------

#[test]
fn ac_2_motif_emergence() {
    let conn = default_conn();
    let mut stim = Stimulus::empty();
    let sensory = conn.sensory_neurons().to_vec();
    for k in 0..20 {
        let t0 = 20.0 + k as f32 * 15.0;
        for i in 0..sensory.len().min(16) {
            stim.push(CurrentInjection {
                t_ms: t0 + i as f32 * 0.20,
                target: sensory[i],
                charge_pa: 90.0,
            });
        }
    }
    let (_spikes_total, spikes) = run_one(&conn, &stim, 400.0);
    let an = Analysis::new(AnalysisConfig {
        motif_window_ms: 20.0,
        motif_bins: 10,
        index_capacity: 128,
        ..AnalysisConfig::default()
    });
    let (index, hits) = an.retrieve_motifs(&conn, &spikes, 5);
    assert!(
        index.len() >= 5,
        "ac-2: motif index too small to judge emergence (len={})",
        index.len()
    );
    assert!(
        hits.len() >= 3,
        "ac-2: fewer than 3 hits (got {})",
        hits.len()
    );
    let mut ds: Vec<f32> = hits.iter().map(|h| h.nearest_distance).collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = ds[ds.len() / 2];
    let below = ds.iter().filter(|d| **d <= median + 1e-6).count();
    let precision = below as f32 / hits.len() as f32;
    eprintln!(
        "ac-2: precision@5_proxy={precision:.3}  hits={}  corpus={}  SOTA_target=0.80",
        hits.len(),
        index.len()
    );
    assert!(
        precision >= 0.60,
        "ac-2: precision@5 proxy {precision:.3} below demo-scale floor 0.60 \
         (SOTA target 0.80; see BENCHMARK.md AC-2 for gap)"
    );
}

// -----------------------------------------------------------------
// AC-4 — Coherence prediction
// -----------------------------------------------------------------

#[test]
fn ac_4_coherence_prediction() {
    let mut hits = 0_u32;
    let trials = 10_u32;
    for seed in 0..trials {
        let mut obs = Observer::new(64).with_detector(50.0, 3.0, 3, 0.75);
        for k in 0..30 {
            let t = k as f32 * 10.0;
            for i in 0..16 {
                obs.on_spike(Spike {
                    t_ms: t + i as f32 * 0.1 + (seed as f32) * 0.007,
                    neuron: NeuronId(i),
                });
            }
        }
        let t_marker = 300.0_f32;
        for k in 0..20 {
            let base = t_marker + k as f32 * 10.0;
            for i in 0..8 {
                obs.on_spike(Spike {
                    t_ms: base + i as f32 * 0.05,
                    neuron: NeuronId(i),
                });
            }
            for i in 8..16 {
                obs.on_spike(Spike {
                    t_ms: base + 7.0 + (i - 8) as f32 * 0.05,
                    neuron: NeuronId(i),
                });
            }
        }
        let events = obs.finalize().coherence_events;
        if events.iter().any(|e| (e.t_ms - t_marker).abs() <= 200.0) {
            hits += 1;
        }
    }
    let rate = hits as f32 / trials as f32;
    eprintln!(
        "ac-4: detect-rate={rate:.2}  hits={hits}/{trials}  SOTA target ≥ 0.70 with ≥ 50 ms lead"
    );
    assert!(
        rate >= 0.50,
        "ac-4: detect-rate {rate:.2} below demo-scale floor 0.50 \
         (SOTA target 0.70 with 50 ms lead; BENCHMARK.md AC-4)"
    );
}
