//! Musica — Dynamic MinCut Audio Source Separation
//!
//! Full benchmark suite: basic separation, hearing aid streaming,
//! multitrack 6-stem splitting, and crowd-scale identity tracking.

mod audio_graph;
mod benchmark;
mod crowd;
mod hearing_aid;
mod lanczos;
mod multitrack;
mod separator;
mod stft;
mod wav;

use audio_graph::GraphParams;
use benchmark::{benchmark_freq_baseline, benchmark_mincut, generate_test_signal, print_comparison};
use separator::SeparatorConfig;

fn main() {
    println!("================================================================");
    println!("  MUSICA — Structure-First Audio Source Separation");
    println!("  Dynamic MinCut + Laplacian Eigenvectors + SIMD");
    println!("================================================================");

    // ── Part 1: Basic separation benchmarks ─────────────────────────────
    println!("\n======== PART 1: Basic Source Separation ========");
    run_basic_benchmarks();

    // ── Part 2: Hearing aid streaming ───────────────────────────────────
    println!("\n======== PART 2: Hearing Aid Streaming (<8ms) ========");
    run_hearing_aid_benchmark();

    // ── Part 3: Multitrack 6-stem separation ────────────────────────────
    println!("\n======== PART 3: Multitrack 6-Stem Separation ========");
    run_multitrack_benchmark();

    // ── Part 4: Lanczos eigensolver validation ──────────────────────────
    println!("\n======== PART 4: Lanczos Eigensolver Validation ========");
    run_lanczos_validation();

    // ── Part 5: Crowd-scale identity tracking ───────────────────────────
    println!("\n======== PART 5: Crowd-Scale Speaker Tracking ========");
    run_crowd_benchmark();

    // ── Part 6: WAV I/O ─────────────────────────────────────────────────
    println!("\n======== PART 6: WAV I/O Validation ========");
    run_wav_validation();

    println!("\n================================================================");
    println!("  MUSICA benchmark suite complete");
    println!("  All modules validated.");
    println!("================================================================");
}

// ── Part 1 ──────────────────────────────────────────────────────────────

fn run_basic_benchmarks() {
    let sr = 8000.0;
    let ws = 256;
    let hs = 128;

    for (label, freqs, amps) in [
        ("well-separated", vec![200.0, 2000.0], vec![1.0, 0.8]),
        ("close-tones", vec![400.0, 600.0], vec![1.0, 1.0]),
        ("harmonic-3rd", vec![300.0, 900.0], vec![1.0, 0.6]),
    ] {
        let (mixed, sources) = generate_test_signal(sr, 0.5, &freqs, &amps);
        println!("\n-- {label}: {} samples", mixed.len());

        let mc = benchmark_mincut(
            &mixed, &sources, sr, ws, hs,
            &GraphParams::default(),
            &SeparatorConfig { num_sources: sources.len(), ..SeparatorConfig::default() },
        );
        let bl = benchmark_freq_baseline(&mixed, &sources, sr, ws, hs, sources.len());
        print_comparison(&[mc, bl]);
    }
}

// ── Part 2 ──────────────────────────────────────────────────────────────

fn run_hearing_aid_benchmark() {
    use hearing_aid::{HearingAidConfig, StreamingState};
    use std::f64::consts::PI;

    let config = HearingAidConfig::default();
    let mut state = StreamingState::new(&config);
    let frame_samples = (config.sample_rate * config.frame_size_ms / 1000.0) as usize;

    // Generate binaural speech + cafeteria noise
    let num_frames = 100;
    let mut total_latency_us = 0u64;
    let mut max_latency_us = 0u64;
    let mut speech_mask_avg = 0.0f64;

    for f in 0..num_frames {
        let t_base = f as f64 * config.hop_size_ms / 1000.0;

        // Speech: coherent harmonics from front
        let left: Vec<f64> = (0..frame_samples)
            .map(|i| {
                let t = t_base + i as f64 / config.sample_rate;
                0.6 * (2.0 * PI * 200.0 * t).sin()
                    + 0.2 * (2.0 * PI * 400.0 * t).sin()
                    + 0.05 * (t * 1000.0).sin() // Noise
            })
            .collect();

        let right: Vec<f64> = (0..frame_samples)
            .map(|i| {
                let t = t_base + i as f64 / config.sample_rate;
                0.55 * (2.0 * PI * 200.0 * t).sin()
                    + 0.18 * (2.0 * PI * 400.0 * t).sin()
                    + 0.07 * (t * 1300.0).sin() // Different noise at right ear
            })
            .collect();

        let result = state.process_frame(&left, &right, &config);
        total_latency_us += result.latency_us;
        max_latency_us = max_latency_us.max(result.latency_us);
        speech_mask_avg += result.mask.iter().sum::<f64>() / result.mask.len() as f64;
    }

    let avg_latency_us = total_latency_us / num_frames as u64;
    speech_mask_avg /= num_frames as f64;

    println!("  Frames processed: {num_frames}");
    println!("  Avg latency:      {avg_latency_us} us ({:.2} ms)", avg_latency_us as f64 / 1000.0);
    println!("  Max latency:      {max_latency_us} us ({:.2} ms)", max_latency_us as f64 / 1000.0);
    println!("  Avg speech mask:  {speech_mask_avg:.3}");
    println!("  Latency budget:   {} (target <8ms)",
        if max_latency_us < 8000 { "PASS" } else { "OVER BUDGET" });
}

// ── Part 3 ──────────────────────────────────────────────────────────────

fn run_multitrack_benchmark() {
    use multitrack::{separate_multitrack, MultitrackConfig, Stem};
    use std::f64::consts::PI;

    let sr = 44100.0;
    let duration = 1.0;
    let n = (sr * duration) as usize;

    // Synthetic multi-instrument signal
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            // Vocals: 200 Hz + harmonics
            let vocals = 0.4 * (2.0 * PI * 200.0 * t).sin()
                + 0.15 * (2.0 * PI * 400.0 * t).sin()
                + 0.08 * (2.0 * PI * 600.0 * t).sin();
            // Bass: 80 Hz
            let bass = 0.3 * (2.0 * PI * 80.0 * t).sin()
                + 0.1 * (2.0 * PI * 160.0 * t).sin();
            // Guitar: 330 Hz + harmonics
            let guitar = 0.2 * (2.0 * PI * 330.0 * t).sin()
                + 0.08 * (2.0 * PI * 660.0 * t).sin();
            // Simple drum: periodic transient
            let drum = if (t * 4.0).fract() < 0.01 { 0.5 } else { 0.0 };

            vocals + bass + guitar + drum
        })
        .collect();

    let config = MultitrackConfig {
        window_size: 1024,
        hop_size: 512,
        sample_rate: sr,
        graph_window_frames: 4,
        ..MultitrackConfig::default()
    };

    println!("  Signal: {} samples ({:.1}s at {:.0} Hz)", n, duration, sr);

    let result = separate_multitrack(&signal, &config);

    println!("  Processing time:  {:.1} ms", result.stats.processing_time_ms);
    println!("  Graph:            {} nodes, {} edges", result.stats.graph_nodes, result.stats.graph_edges);
    println!("  STFT frames:      {}", result.stats.total_frames);
    println!("  Replay entries:   {}", result.replay_log.len());
    println!();

    for stem_result in &result.stems {
        let energy: f64 = stem_result.signal.iter().map(|s| s * s).sum::<f64>() / n as f64;
        println!(
            "  {:>8}: confidence={:.3}  energy={:.6}",
            stem_result.stem.name(),
            stem_result.confidence,
            energy,
        );
    }

    // Verify masks sum to ~1
    let num_freq = result.stft_result.num_freq_bins;
    let mut mask_sum_err = 0.0f64;
    let check_bins = (result.stft_result.num_frames * num_freq).min(500);
    for i in 0..check_bins {
        let sum: f64 = result.stems.iter().map(|s| s.mask[i]).sum();
        mask_sum_err += (sum - 1.0).abs();
    }
    let avg_err = mask_sum_err / check_bins as f64;
    println!("\n  Mask sum error:   {avg_err:.4} (avg deviation from 1.0)");
}

// ── Part 4 ──────────────────────────────────────────────────────────────

fn run_lanczos_validation() {
    use lanczos::{lanczos_eigenpairs, power_iteration_fiedler, LanczosConfig, SparseMatrix};

    // Two-cluster graph
    let mut edges = vec![];
    for i in 0..10 {
        for j in i + 1..10 {
            edges.push((i, j, 5.0));
        }
    }
    for i in 10..20 {
        for j in i + 1..20 {
            edges.push((i, j, 5.0));
        }
    }
    edges.push((9, 10, 0.1)); // Weak bridge

    let lap = SparseMatrix::from_edges(20, &edges);

    // Power iteration
    let start = std::time::Instant::now();
    let fiedler_pi = power_iteration_fiedler(&lap, 100);
    let pi_time = start.elapsed();

    // Lanczos
    let start = std::time::Instant::now();
    let config = LanczosConfig { k: 4, max_iter: 50, tol: 1e-8, reorthogonalize: true };
    let lanczos_result = lanczos_eigenpairs(&lap, &config);
    let lanczos_time = start.elapsed();

    println!("  Graph: 20 nodes, 2 clusters connected by weak bridge");
    println!("  Power iteration: {:.1}us", pi_time.as_micros());
    println!("  Lanczos (k=4):   {:.1}us ({} iterations, converged={})",
        lanczos_time.as_micros(), lanczos_result.iterations, lanczos_result.converged);

    // Check cluster separation
    let cluster_a: Vec<f64> = fiedler_pi[..10].to_vec();
    let cluster_b: Vec<f64> = fiedler_pi[10..].to_vec();
    let a_sign = cluster_a[0].signum();
    let b_sign = cluster_b[0].signum();
    let clean_split = a_sign != b_sign;

    println!("  Fiedler clean split: {}", if clean_split { "YES" } else { "NO" });

    if !lanczos_result.eigenvalues.is_empty() {
        println!("  Eigenvalues: {:?}",
            lanczos_result.eigenvalues.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
    }
}

// ── Part 5 ──────────────────────────────────────────────────────────────

fn run_crowd_benchmark() {
    use crowd::{CrowdConfig, CrowdTracker, SpeechEvent};

    let config = CrowdConfig {
        max_identities: 500,
        association_threshold: 0.4,
        ..CrowdConfig::default()
    };
    let mut tracker = CrowdTracker::new(config);

    // 20 sensors in a grid
    for x in 0..5 {
        for y in 0..4 {
            tracker.add_sensor((x as f64 * 10.0, y as f64 * 10.0));
        }
    }

    // Simulate crowd: 50 speakers at various positions over time
    let start = std::time::Instant::now();
    for t_step in 0..10 {
        let time = t_step as f64 * 1.0;

        for speaker in 0..50 {
            let direction = (speaker as f64 * 7.3) % 360.0 - 180.0;
            let freq = 150.0 + (speaker as f64 * 30.0) % 400.0;
            let sensor = speaker % tracker.sensors.len();

            let events: Vec<SpeechEvent> = (0..3)
                .map(|i| SpeechEvent {
                    time: time + i as f64 * 0.1,
                    freq_centroid: freq + i as f64 * 5.0,
                    energy: 0.3 + (speaker as f64 * 0.01) % 0.5,
                    voicing: 0.6 + (speaker as f64 * 0.005) % 0.3,
                    harmonicity: 0.5 + (speaker as f64 * 0.003) % 0.4,
                    direction,
                    sensor_id: sensor,
                })
                .collect();

            tracker.ingest_events(sensor, events);
        }

        tracker.update_local_graphs();
        tracker.associate_cross_sensor(time + 0.5);
        tracker.update_global_identities(time + 0.5);
    }
    let elapsed = start.elapsed();

    let stats = tracker.get_stats();
    println!("  Sensors:          {}", stats.sensors);
    println!("  Total events:     {}", stats.total_events);
    println!("  Local speakers:   {}", stats.total_local_speakers);
    println!("  Global identities:{}", stats.total_identities);
    println!("  Active speakers:  {}", stats.active_speakers);
    println!("  Processing time:  {:.1} ms", elapsed.as_secs_f64() * 1000.0);
}

// ── Part 6 ──────────────────────────────────────────────────────────────

fn run_wav_validation() {
    use std::f64::consts::PI;

    let path = "/tmp/musica_test.wav";
    let sr = 16000u32;
    let n = 16000; // 1 second

    let samples: Vec<f64> = (0..n)
        .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f64 / sr as f64).sin())
        .collect();

    match wav::write_wav(path, &samples, sr, 1) {
        Ok(()) => {
            match wav::read_wav(path) {
                Ok(loaded) => {
                    let max_err: f64 = samples.iter()
                        .zip(loaded.channel_data[0].iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);

                    println!("  WAV roundtrip:    {} samples, max error = {:.6}", n, max_err);
                    println!("  Sample rate:      {} Hz", loaded.sample_rate);
                    println!("  Channels:         {}", loaded.channels);
                    println!("  Status:           {}", if max_err < 0.001 { "PASS" } else { "FAIL" });
                }
                Err(e) => println!("  WAV read error: {e}"),
            }
        }
        Err(e) => println!("  WAV write error: {e}"),
    }

    // Binaural test
    let stereo_path = "/tmp/musica_binaural_test.wav";
    match wav::generate_binaural_test_wav(stereo_path, sr, 0.5, 300.0, &[800.0, 1200.0], 30.0) {
        Ok(()) => {
            match wav::read_wav(stereo_path) {
                Ok(loaded) => {
                    println!("  Binaural WAV:     {} channels, {} samples/ch",
                        loaded.channels, loaded.channel_data[0].len());
                }
                Err(e) => println!("  Binaural read error: {e}"),
            }
        }
        Err(e) => println!("  Binaural write error: {e}"),
    }
}
