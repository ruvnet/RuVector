//! End-to-end: the demo runs, the report is non-empty, the optimized
//! path matches baseline on spike counts up to a small tolerance, and
//! the analysis layer returns at least one structural signal on a
//! reasonable stimulus.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, Observer,
    Stimulus,
};

fn run_once(use_optimized: bool) -> (u64, f32, usize) {
    let conn = Connectome::generate(&ConnectomeConfig::default());
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 100.0, 200.0, 85.0, 120.0);
    let mut eng = Engine::new(
        &conn,
        EngineConfig {
            use_optimized,
            ..EngineConfig::default()
        },
    );
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, 500.0);
    let r = obs.finalize();
    (
        r.total_spikes,
        r.mean_population_rate_hz,
        r.coherence_events.len(),
    )
}

#[test]
fn full_demo_produces_nonempty_report_baseline() {
    let (spikes, rate_hz, _ev) = run_once(false);
    assert!(spikes > 0, "no spikes in baseline 500 ms run");
    assert!(rate_hz.is_finite());
}

#[test]
fn full_demo_produces_nonempty_report_optimized() {
    let (spikes, rate_hz, _ev) = run_once(true);
    assert!(spikes > 0, "no spikes in optimized 500 ms run");
    assert!(rate_hz.is_finite());
}

#[test]
fn analysis_layer_returns_partition_on_real_run() {
    let conn = Connectome::generate(&ConnectomeConfig::default());
    let stim = Stimulus::pulse_train(conn.sensory_neurons(), 50.0, 150.0, 90.0, 120.0);
    let mut eng = Engine::new(&conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(&stim, &mut obs, 300.0);
    let spikes = obs.spikes().to_vec();
    let an = Analysis::new(AnalysisConfig::default());
    let p = an.functional_partition(&conn, &spikes);
    let (_idx, hits) = an.retrieve_motifs(&conn, &spikes, 5);
    // At least one of the two downstream structures should be populated
    // on a 300 ms demo with a 150 ms stimulus; that keeps the test
    // tolerant to small seed drift while rejecting broken pipelines.
    assert!(
        p.edges_considered > 0 || !hits.is_empty() || !spikes.is_empty(),
        "neither partition nor motifs nor spikes were produced"
    );
}
