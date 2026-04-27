//! Demo runner for `connectome-fly`.
//!
//! Generates (or regenerates) the synthetic connectome, injects a
//! 200 ms deterministic sensory stimulus at T = 100 ms, runs 500 ms of
//! simulated time, and writes a JSON report summarising total spikes,
//! the population-rate trace, top coherence events, the functional
//! partition, and the top-5 repeated motifs.
//!
//! ADR-154 §3(6).

use std::io::Write;
use std::time::Instant;

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, Observer,
    Stimulus,
};
use serde::Serialize;

#[derive(Serialize)]
struct DemoReport {
    adr: &'static str,
    positioning: &'static str,
    config: DemoConfig,
    connectome: ConnectomeSummary,
    simulation: SimulationSummary,
    coherence: CoherenceSummary,
    partition: connectome_fly::FunctionalPartition,
    motifs: Vec<connectome_fly::MotifHit>,
    timings_ms: Timings,
}

#[derive(Serialize)]
struct DemoConfig {
    num_neurons: u32,
    num_modules: u16,
    avg_out_degree: f32,
    stimulus_onset_ms: f32,
    stimulus_duration_ms: f32,
    t_end_ms: f32,
    use_optimized_lif: bool,
}

#[derive(Serialize)]
struct ConnectomeSummary {
    num_neurons: u32,
    num_synapses: u32,
    num_sensory: u32,
    num_motor: u32,
    seed: u64,
}

#[derive(Serialize)]
struct SimulationSummary {
    total_spikes: u64,
    mean_population_rate_hz: f32,
    num_population_rate_bins: usize,
    first_10_rate_samples_hz: Vec<f32>,
}

#[derive(Serialize)]
struct CoherenceSummary {
    events_total: usize,
    top_events: Vec<connectome_fly::CoherenceEvent>,
}

#[derive(Serialize)]
struct Timings {
    generate_ms: f64,
    run_ms: f64,
    analysis_ms: f64,
    total_ms: f64,
}

fn main() {
    let t0 = Instant::now();

    // Arguments: [output_json_path]. Defaults to stdout.
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).cloned();

    // Build the connectome.
    let cfg = ConnectomeConfig::default();
    let t_gen = Instant::now();
    let conn = Connectome::generate(&cfg);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    // Build the stimulus: 200 ms pulse train into sensory neurons starting
    // at T = 100 ms.
    let stim = Stimulus::pulse_train(
        conn.sensory_neurons(),
        /* onset_ms = */ 100.0,
        /* duration_ms = */ 200.0,
        /* amplitude_pa = */ 85.0,
        /* rate_hz = */ 120.0,
    );

    // Build the engine.
    let eng_cfg = EngineConfig::default();
    let mut engine = Engine::new(&conn, eng_cfg);
    let mut obs = Observer::new(conn.num_neurons());

    let t_run = Instant::now();
    let t_end_ms: f32 = 500.0;
    engine.run_with(&stim, &mut obs, t_end_ms);
    let run_ms = t_run.elapsed().as_secs_f64() * 1000.0;

    let report_obs = obs.finalize();

    // Run the analysis layer.
    let t_an = Instant::now();
    let analysis = Analysis::new(AnalysisConfig::default());
    let partition = analysis.functional_partition(&conn, obs.spikes());
    let (_idx, motifs) = analysis.retrieve_motifs(&conn, obs.spikes(), 5);
    let analysis_ms = t_an.elapsed().as_secs_f64() * 1000.0;

    let demo_report = DemoReport {
        adr: "ADR-154",
        positioning:
            "Graph-native embodied connectome runtime with structural coherence analysis, \
             counterfactual circuit testing, and auditable behavior generation. Not mind upload; \
             not consciousness upload.",
        config: DemoConfig {
            num_neurons: cfg.num_neurons,
            num_modules: cfg.num_modules,
            avg_out_degree: cfg.avg_out_degree,
            stimulus_onset_ms: 100.0,
            stimulus_duration_ms: 200.0,
            t_end_ms,
            use_optimized_lif: eng_cfg.use_optimized,
        },
        connectome: ConnectomeSummary {
            num_neurons: conn.num_neurons() as u32,
            num_synapses: conn.num_synapses() as u32,
            num_sensory: conn.sensory_neurons().len() as u32,
            num_motor: conn.motor_neurons().len() as u32,
            seed: conn.seed(),
        },
        simulation: SimulationSummary {
            total_spikes: report_obs.total_spikes,
            mean_population_rate_hz: report_obs.mean_population_rate_hz,
            num_population_rate_bins: report_obs.population_rate_hz.len(),
            first_10_rate_samples_hz: report_obs
                .population_rate_hz
                .iter()
                .take(10)
                .copied()
                .collect(),
        },
        coherence: CoherenceSummary {
            events_total: report_obs.coherence_events.len(),
            top_events: report_obs
                .coherence_events
                .iter()
                .take(3)
                .cloned()
                .collect(),
        },
        partition,
        motifs,
        timings_ms: Timings {
            generate_ms: gen_ms,
            run_ms,
            analysis_ms,
            total_ms: t0.elapsed().as_secs_f64() * 1000.0,
        },
    };

    let json = serde_json::to_string_pretty(&demo_report).expect("serialize report");
    if let Some(path) = out_path {
        let mut f = std::fs::File::create(&path).expect("open output file");
        f.write_all(json.as_bytes()).expect("write");
        println!("wrote report to {}", path);
    } else {
        println!("{}", json);
    }
}
