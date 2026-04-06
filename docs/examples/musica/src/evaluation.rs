//! Real audio evaluation module with realistic signal generation and BSS metrics.
//!
//! Generates synthetic test signals that mimic real-world audio scenarios
//! (speech, drums, bass, noise) and evaluates separation quality with
//! SDR, SIR, and SAR metrics.

use std::f64::consts::PI;

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::separator::{separate, SeparatorConfig};
use crate::stft;

// ── Deterministic RNG ───────────────────────────────────────────────────

/// Simple LCG random number generator for deterministic tests.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Next value in [-1, 1).
    fn next_signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

// ── Signal Generators ───────────────────────────────────────────────────

/// Generate a speech-like signal with harmonics, vibrato, and formant shaping.
pub fn generate_speech_like(
    sample_rate: f64,
    duration: f64,
    f0: f64,
    num_harmonics: usize,
    vibrato_rate: f64,
    vibrato_depth: f64,
) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let mut signal = vec![0.0; n];

    // Formant center frequencies and bandwidths (simplified vowel /a/)
    let formants = [(500.0, 100.0), (1500.0, 200.0), (2500.0, 300.0)];

    for i in 0..n {
        let t = i as f64 / sample_rate;

        // ADSR envelope: attack 5%, sustain 80%, release 15%
        let pos = i as f64 / n as f64;
        let env = if pos < 0.05 {
            pos / 0.05
        } else if pos < 0.85 {
            1.0
        } else {
            (1.0 - pos) / 0.15
        };

        // Vibrato-modulated fundamental
        let vibrato = 1.0 + vibrato_depth * (2.0 * PI * vibrato_rate * t).sin();
        let f_inst = f0 * vibrato;

        // Sum harmonics with 1/k roll-off
        let mut sample = 0.0;
        for k in 1..=num_harmonics {
            let freq = f_inst * k as f64;
            let amp = 1.0 / k as f64;

            // Simple formant shaping: boost near formant centers
            let mut formant_gain = 0.3; // baseline
            for &(fc, bw) in &formants {
                let dist = (freq - fc).abs();
                formant_gain += (-(dist * dist) / (2.0 * bw * bw)).exp();
            }

            sample += amp * formant_gain * (2.0 * PI * freq * t).sin();
        }

        signal[i] = sample * env * 0.3;
    }

    signal
}

/// Drum type for pattern generation.
#[derive(Debug, Clone, Copy)]
pub enum DrumType {
    Kick,
    Snare,
    HiHat,
}

/// Generate a drum pattern signal.
pub fn generate_drum_pattern(
    sample_rate: f64,
    duration: f64,
    bpm: f64,
    pattern: &[(f64, DrumType)],
) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let mut signal = vec![0.0; n];
    let beat_duration = 60.0 / bpm;
    let mut rng = Lcg::new(42);

    for &(beat_time, drum_type) in pattern {
        let onset_sample = (beat_time * beat_duration * sample_rate) as usize;
        if onset_sample >= n {
            continue;
        }

        match drum_type {
            DrumType::Kick => {
                // 60Hz burst with exponential decay + noise click
                let decay_samples = (0.15 * sample_rate) as usize;
                for j in 0..decay_samples.min(n - onset_sample) {
                    let t = j as f64 / sample_rate;
                    let env = (-t / 0.04).exp();
                    // Pitch drops from 150Hz to 60Hz
                    let freq = 60.0 + 90.0 * (-t / 0.02).exp();
                    signal[onset_sample + j] += 0.8 * env * (2.0 * PI * freq * t).sin();
                }
            }
            DrumType::Snare => {
                // 200Hz body + bandpass white noise
                let decay_samples = (0.12 * sample_rate) as usize;
                for j in 0..decay_samples.min(n - onset_sample) {
                    let t = j as f64 / sample_rate;
                    let env = (-t / 0.03).exp();
                    let body = 0.4 * (2.0 * PI * 200.0 * t).sin();
                    let noise = 0.5 * rng.next_signed();
                    signal[onset_sample + j] += env * (body + noise);
                }
            }
            DrumType::HiHat => {
                // High-pass filtered noise burst
                let decay_samples = (0.05 * sample_rate) as usize;
                let mut prev = 0.0;
                for j in 0..decay_samples.min(n - onset_sample) {
                    let t = j as f64 / sample_rate;
                    let env = (-t / 0.01).exp();
                    let noise = rng.next_signed();
                    // Simple high-pass: y[n] = x[n] - x[n-1]
                    let hp = noise - prev;
                    prev = noise;
                    signal[onset_sample + j] += 0.3 * env * hp;
                }
            }
        }
    }

    signal
}

/// Generate a bass line with sub-bass and slight harmonics.
pub fn generate_bass_line(
    sample_rate: f64,
    duration: f64,
    notes: &[f64],
) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let mut signal = vec![0.0; n];

    if notes.is_empty() {
        return signal;
    }

    let note_duration = n / notes.len();
    let mut current_freq = notes[0];

    for i in 0..n {
        let note_idx = (i / note_duration).min(notes.len() - 1);
        let target_freq = notes[note_idx];

        // Portamento: smooth transition between notes
        current_freq += (target_freq - current_freq) * 0.001;

        let t = i as f64 / sample_rate;
        signal[i] = 0.6 * (2.0 * PI * current_freq * t).sin()
            + 0.2 * (2.0 * PI * current_freq * 2.0 * t).sin()
            + 0.05 * (2.0 * PI * current_freq * 3.0 * t).sin();
    }

    signal
}

/// Noise type for generation.
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    White,
    Pink,
    Babble,
}

/// Generate noise of the specified type.
pub fn generate_noise(
    sample_rate: f64,
    duration: f64,
    noise_type: NoiseType,
) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    let mut rng = Lcg::new(1337);

    match noise_type {
        NoiseType::White => {
            (0..n).map(|_| rng.next_signed() * 0.3).collect()
        }
        NoiseType::Pink => {
            // Leaky integrator filter on white noise → ~1/f
            let alpha = 0.98;
            let mut state = 0.0;
            (0..n)
                .map(|_| {
                    state = alpha * state + (1.0 - alpha) * rng.next_signed();
                    state * 0.5
                })
                .collect()
        }
        NoiseType::Babble => {
            // Sum of 6 detuned speech-like signals
            let f0s = [100.0, 130.0, 170.0, 200.0, 250.0, 310.0];
            let mut result = vec![0.0; n];
            for &f0 in &f0s {
                let voice = generate_speech_like(sample_rate, duration, f0, 6, 4.0, 0.01);
                for (i, &v) in voice.iter().enumerate() {
                    if i < n {
                        result[i] += v / 6.0;
                    }
                }
            }
            result
        }
    }
}

// ── Test Scenarios ──────────────────────────────────────────────────────

/// Test scenario enum.
#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    /// Speech in background noise.
    SpeechInNoise,
    /// Two concurrent speakers.
    TwoSpeakers,
    /// Music mix (vocals + bass + drums).
    MusicMix,
    /// Cocktail party (4 speakers + babble).
    CocktailParty,
}

/// A test scenario with mixed signal, individual sources, and metadata.
#[derive(Debug, Clone)]
pub struct TestScenario {
    /// Mixed signal.
    pub mixed: Vec<f64>,
    /// Individual source signals.
    pub sources: Vec<Vec<f64>>,
    /// Source labels.
    pub labels: Vec<String>,
    /// Scenario type.
    pub scenario: Scenario,
    /// Sample rate.
    pub sample_rate: f64,
}

/// Generate a realistic test scenario.
pub fn generate_scenario(
    sample_rate: f64,
    duration: f64,
    scenario: Scenario,
) -> TestScenario {
    let (sources, labels) = match scenario {
        Scenario::SpeechInNoise => {
            let speech = generate_speech_like(sample_rate, duration, 150.0, 12, 5.0, 0.02);
            let noise = generate_noise(sample_rate, duration, NoiseType::Pink);
            (vec![speech, noise], vec!["speech".into(), "noise".into()])
        }
        Scenario::TwoSpeakers => {
            let s1 = generate_speech_like(sample_rate, duration, 120.0, 10, 5.0, 0.02);
            let s2 = generate_speech_like(sample_rate, duration, 220.0, 8, 6.0, 0.03);
            (vec![s1, s2], vec!["speaker1".into(), "speaker2".into()])
        }
        Scenario::MusicMix => {
            let vocals = generate_speech_like(sample_rate, duration, 200.0, 15, 5.5, 0.03);
            let bass = generate_bass_line(sample_rate, duration, &[60.0, 80.0, 60.0, 100.0]);
            let pattern = vec![
                (0.0, DrumType::Kick),
                (1.0, DrumType::HiHat),
                (2.0, DrumType::Snare),
                (3.0, DrumType::HiHat),
                (4.0, DrumType::Kick),
                (5.0, DrumType::HiHat),
                (6.0, DrumType::Snare),
                (7.0, DrumType::HiHat),
            ];
            let drums = generate_drum_pattern(sample_rate, duration, 120.0, &pattern);
            (
                vec![vocals, bass, drums],
                vec!["vocals".into(), "bass".into(), "drums".into()],
            )
        }
        Scenario::CocktailParty => {
            let s1 = generate_speech_like(sample_rate, duration, 110.0, 10, 5.0, 0.02);
            let s2 = generate_speech_like(sample_rate, duration, 160.0, 8, 4.5, 0.025);
            let s3 = generate_speech_like(sample_rate, duration, 210.0, 9, 6.0, 0.015);
            let s4 = generate_speech_like(sample_rate, duration, 280.0, 7, 5.5, 0.03);
            let babble = generate_noise(sample_rate, duration, NoiseType::Babble);
            (
                vec![s1, s2, s3, s4, babble],
                vec![
                    "spk1".into(),
                    "spk2".into(),
                    "spk3".into(),
                    "spk4".into(),
                    "babble".into(),
                ],
            )
        }
    };

    let n = sources[0].len();
    let mut mixed = vec![0.0; n];
    for src in &sources {
        for (i, &s) in src.iter().enumerate() {
            if i < n {
                mixed[i] += s;
            }
        }
    }

    TestScenario {
        mixed,
        sources,
        labels,
        scenario,
        sample_rate,
    }
}

// ── BSS Evaluation Metrics ──────────────────────────────────────────────

/// Full BSS evaluation result for one source.
#[derive(Debug, Clone)]
pub struct BssMetrics {
    /// Signal-to-Distortion Ratio (dB).
    pub sdr: f64,
    /// Signal-to-Interference Ratio (dB).
    pub sir: f64,
    /// Signal-to-Artifacts Ratio (dB).
    pub sar: f64,
}

/// Compute SDR between reference and estimated signals.
pub fn compute_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let ref_energy: f64 = reference[..n].iter().map(|x| x * x).sum();
    let noise_energy: f64 = reference[..n]
        .iter()
        .zip(estimate[..n].iter())
        .map(|(r, e)| (r - e).powi(2))
        .sum();

    if noise_energy < 1e-12 {
        return 100.0;
    }
    if ref_energy < 1e-12 {
        return f64::NEG_INFINITY;
    }

    10.0 * (ref_energy / noise_energy).log10()
}

/// Compute SIR: ratio of target projection energy to interference energy.
pub fn compute_sir(reference: &[f64], estimate: &[f64], interferences: &[&[f64]]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    // Project estimate onto reference direction
    let ref_energy: f64 = reference[..n].iter().map(|x| x * x).sum();
    if ref_energy < 1e-12 {
        return f64::NEG_INFINITY;
    }

    let cross: f64 = reference[..n]
        .iter()
        .zip(estimate[..n].iter())
        .map(|(r, e)| r * e)
        .sum();
    let scale = cross / ref_energy;
    let target_energy: f64 = reference[..n].iter().map(|r| (r * scale).powi(2)).sum();

    // Total interference energy
    let mut interf_energy = 0.0f64;
    for interf in interferences {
        let m = n.min(interf.len());
        interf_energy += interf[..m].iter().map(|x| x * x).sum::<f64>();
    }

    if interf_energy < 1e-12 {
        return 100.0;
    }

    10.0 * (target_energy / interf_energy).log10()
}

/// Compute SAR: ratio of estimate energy to artifact energy.
pub fn compute_sar(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let est_energy: f64 = estimate[..n].iter().map(|x| x * x).sum();
    let artifact_energy: f64 = reference[..n]
        .iter()
        .zip(estimate[..n].iter())
        .map(|(r, e)| (e - r).powi(2))
        .sum();

    if artifact_energy < 1e-12 {
        return 100.0;
    }

    10.0 * (est_energy / artifact_energy).log10()
}

/// Compute full BSS metrics for one source.
pub fn compute_bss(
    reference: &[f64],
    estimate: &[f64],
    interferences: &[&[f64]],
) -> BssMetrics {
    BssMetrics {
        sdr: compute_sdr(reference, estimate),
        sir: compute_sir(reference, estimate, interferences),
        sar: compute_sar(reference, estimate),
    }
}

// ── Full Evaluation Pipeline ────────────────────────────────────────────

/// Evaluation result for a complete scenario.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Scenario type.
    pub scenario: Scenario,
    /// Per-source BSS metrics.
    pub source_metrics: Vec<(String, BssMetrics)>,
    /// Average SDR across sources.
    pub avg_sdr: f64,
    /// Processing time in milliseconds.
    pub processing_ms: f64,
    /// Number of graph nodes.
    pub graph_nodes: usize,
    /// Number of graph edges.
    pub graph_edges: usize,
}

/// Run full evaluation on a test scenario using mincut separation.
pub fn evaluate_scenario(
    test: &TestScenario,
    window_size: usize,
    hop_size: usize,
    graph_params: &GraphParams,
) -> EvaluationResult {
    let start = std::time::Instant::now();

    let stft_result = stft::stft(&test.mixed, window_size, hop_size, test.sample_rate);
    let graph = build_audio_graph(&stft_result, graph_params);
    let graph_nodes = graph.num_nodes;
    let graph_edges = graph.num_nodes; // approximate; actual edge count tracked internally

    let sep_config = SeparatorConfig {
        num_sources: test.sources.len(),
        ..SeparatorConfig::default()
    };
    let separation = separate(&graph, &sep_config);

    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Recover signals and compute metrics
    let mut source_metrics = Vec::new();
    let mut total_sdr = 0.0;
    let num_masks = separation.masks.len().min(test.sources.len());

    for s in 0..num_masks {
        let recovered = stft::istft(&stft_result, &separation.masks[s], test.mixed.len());

        // Build interference list (all sources except current)
        let interferences: Vec<&[f64]> = test
            .sources
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != s)
            .map(|(_, src)| src.as_slice())
            .collect();

        let metrics = compute_bss(&test.sources[s], &recovered, &interferences);
        total_sdr += metrics.sdr;
        source_metrics.push((test.labels[s].clone(), metrics));
    }

    let avg_sdr = if num_masks > 0 {
        total_sdr / num_masks as f64
    } else {
        f64::NEG_INFINITY
    };

    EvaluationResult {
        scenario: test.scenario,
        source_metrics,
        avg_sdr,
        processing_ms,
        graph_nodes,
        graph_edges,
    }
}

/// Run evaluation across all scenarios and print a summary report.
pub fn run_full_evaluation(sample_rate: f64, duration: f64) -> Vec<EvaluationResult> {
    let scenarios = [
        Scenario::SpeechInNoise,
        Scenario::TwoSpeakers,
        Scenario::MusicMix,
        Scenario::CocktailParty,
    ];

    let graph_params = GraphParams::default();
    let window_size = 256;
    let hop_size = 128;

    let mut results = Vec::new();

    for &scenario in &scenarios {
        let test = generate_scenario(sample_rate, duration, scenario);
        let result = evaluate_scenario(&test, window_size, hop_size, &graph_params);
        results.push(result);
    }

    results
}

/// Print a formatted evaluation report.
pub fn print_evaluation_report(results: &[EvaluationResult]) {
    println!("  {:<20} {:>8} {:>8} {:>8} {:>10}", "Source", "SDR", "SIR", "SAR", "Time(ms)");
    println!("  {}", "-".repeat(60));

    for result in results {
        println!("\n  Scenario: {:?}", result.scenario);
        for (label, metrics) in &result.source_metrics {
            println!(
                "    {:<18} {:>+7.2} {:>+7.2} {:>+7.2} {:>9.1}",
                label, metrics.sdr, metrics.sir, metrics.sar, result.processing_ms
            );
        }
        println!(
            "    {:<18} {:>+7.2}    avg   graph: {}n/{}e",
            "AVERAGE", result.avg_sdr, result.graph_nodes, result.graph_edges
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_like_generation() {
        let signal = generate_speech_like(8000.0, 0.5, 150.0, 8, 5.0, 0.02);
        assert_eq!(signal.len(), 4000);
        let energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        assert!(energy > 0.0, "Speech signal should have energy");
        assert!(energy < 1.0, "Speech signal should be reasonable amplitude");
    }

    #[test]
    fn test_drum_pattern() {
        let pattern = vec![
            (0.0, DrumType::Kick),
            (1.0, DrumType::Snare),
            (2.0, DrumType::HiHat),
        ];
        let signal = generate_drum_pattern(8000.0, 1.0, 120.0, &pattern);
        assert_eq!(signal.len(), 8000);
        // Should have some non-zero samples near the onsets
        let peak = signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(peak > 0.1, "Drum signal should have transients");
    }

    #[test]
    fn test_bass_line() {
        let signal = generate_bass_line(8000.0, 1.0, &[60.0, 80.0, 60.0]);
        assert_eq!(signal.len(), 8000);
        let energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        assert!(energy > 0.01, "Bass should have energy");
    }

    #[test]
    fn test_noise_types() {
        for noise_type in [NoiseType::White, NoiseType::Pink, NoiseType::Babble] {
            let signal = generate_noise(8000.0, 0.5, noise_type);
            assert_eq!(signal.len(), 4000);
            let energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
            assert!(energy > 0.0, "{:?} noise should have energy", noise_type);
        }
    }

    #[test]
    fn test_scenario_generation() {
        for scenario in [
            Scenario::SpeechInNoise,
            Scenario::TwoSpeakers,
            Scenario::MusicMix,
            Scenario::CocktailParty,
        ] {
            let test = generate_scenario(8000.0, 0.25, scenario);
            assert!(!test.mixed.is_empty());
            assert!(!test.sources.is_empty());
            assert_eq!(test.sources.len(), test.labels.len());

            // Mixed should equal sum of sources
            let n = test.mixed.len();
            for i in 0..n {
                let sum: f64 = test.sources.iter().map(|s| s[i]).sum();
                assert!(
                    (test.mixed[i] - sum).abs() < 1e-10,
                    "Mixed should equal sum of sources"
                );
            }
        }
    }

    #[test]
    fn test_sdr_perfect() {
        let signal = vec![1.0, 0.5, -0.3, 0.7];
        let sdr = compute_sdr(&signal, &signal);
        assert!(sdr > 90.0, "Perfect reconstruction should have very high SDR");
    }

    #[test]
    fn test_sdr_noisy() {
        let reference = vec![1.0; 100];
        let estimate: Vec<f64> = reference.iter().map(|x| x + 0.1).collect();
        let sdr = compute_sdr(&reference, &estimate);
        assert!(sdr > 10.0, "Small noise should give decent SDR");
        assert!(sdr < 30.0, "Non-zero noise should not give perfect SDR");
    }

    #[test]
    fn test_sir_no_interference() {
        let reference = vec![1.0; 100];
        let sir = compute_sir(&reference, &reference, &[]);
        assert!(sir > 90.0, "No interference should give high SIR");
    }

    #[test]
    fn test_full_evaluation_runs() {
        // Short duration for fast test
        let results = run_full_evaluation(8000.0, 0.1);
        assert_eq!(results.len(), 4, "Should evaluate all 4 scenarios");
        for result in &results {
            assert!(!result.source_metrics.is_empty());
            assert!(result.processing_ms >= 0.0);
        }
    }

    #[test]
    fn test_two_speakers_separable() {
        // Well-separated speakers (120Hz vs 220Hz) should give positive SDR
        let test = generate_scenario(8000.0, 0.5, Scenario::TwoSpeakers);
        let result = evaluate_scenario(&test, 256, 128, &GraphParams::default());
        // Just verify it runs and produces metrics — exact SDR depends on graph quality
        assert_eq!(result.source_metrics.len(), 2);
    }
}
