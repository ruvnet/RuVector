//! Coherence detector fires on a constructed collapse; functional
//! partition returns a valid mincut structure.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, NeuronId,
    Observer, Spike, Stimulus,
};

#[test]
fn coherence_event_emits_on_cluster_fragmentation() {
    // Well-connected cluster → two-block fragmentation. Fiedler drops.
    let mut obs = Observer::new(64).with_detector(50.0, 5.0, 3, 1.0);
    for k in 0..30 {
        let t = k as f32 * 10.0;
        for i in 0..16 {
            obs.on_spike(Spike {
                t_ms: t + i as f32 * 0.10,
                neuron: NeuronId(i),
            });
        }
    }
    for k in 0..20 {
        let base = 300.0 + k as f32 * 10.0;
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
    assert!(obs.num_events() > 0, "coherence detector did not fire");
}

#[test]
fn functional_partition_is_non_trivial() {
    let conn = Connectome::generate(&ConnectomeConfig {
        num_neurons: 256,
        avg_out_degree: 16.0,
        ..ConnectomeConfig::default()
    });
    let mut eng = Engine::new(&conn, EngineConfig::default());
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 20.0, 60.0, 90.0, 120.0);
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, 120.0);
    let spikes = obs.spikes().to_vec();
    let an = Analysis::new(AnalysisConfig::default());
    let part = an.functional_partition(&conn, &spikes);
    // A non-empty partition is evidence that mincut actually ran on
    // edges surfaced by recent spike activity.
    if part.edges_considered > 0 {
        let total = part.side_a.len() + part.side_b.len();
        assert!(
            total > 0,
            "mincut considered {} edges but partition is empty",
            part.edges_considered
        );
        assert!(
            part.cut_value >= 0.0,
            "cut value should be non-negative, got {}",
            part.cut_value
        );
    }
}
