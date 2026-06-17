//! Benchmark binary for ruvector-drift semantic drift detectors.
//!
//! Measures: latency (p50/p95), throughput, detection accuracy across
//! three dataset sizes and three detector variants.
//!
//! Run: cargo run --release -p ruvector-drift --bin benchmark

use rand::{Rng, SeedableRng};
use ruvector_drift::{CentroidDrift, CoherenceDrift, DriftConfig, DriftDetector, PsiDrift};
use std::time::Instant;

fn gaussian(n: usize, d: usize, mean: f32, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| mean + rng.gen_range(-sigma..sigma))
                .collect()
        })
        .collect()
}

/// Gradual drift: returns `n` windows where the mean slides from 0.0 to `drift_amount`.
fn gradual_drift_windows(
    windows: usize,
    n: usize,
    d: usize,
    drift_amount: f32,
    seed: u64,
) -> Vec<Vec<Vec<f32>>> {
    (0..windows)
        .map(|w| {
            let mean = drift_amount * (w as f32 / windows as f32);
            gaussian(n, d, mean, 0.3, seed + w as u64)
        })
        .collect()
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * v.len() as f64) as usize;
    v[idx.min(v.len() - 1)]
}

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    n: usize,
    d: usize,
    windows: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_wps: f64,
    tp_rate: f64,
    fp_rate: f64,
    mem_kb: f64,
}

fn bench_detector<D: DriftDetector>(
    name: &'static str,
    cfg: &DriftConfig,
    n: usize,
    d: usize,
    repeats: usize,
    drift_threshold: f32,
    make_detector: impl Fn(DriftConfig) -> D,
) -> BenchResult {
    let windows = 20usize;
    // Half stable, half drifted
    let stable_windows = gradual_drift_windows(windows / 2, n, d, 0.0, 42);
    let drifted_windows = gradual_drift_windows(windows / 2, n, d, drift_threshold, 99);

    // Measure add_window + detect latency over `repeats` full passes
    let mut latencies: Vec<f64> = Vec::with_capacity(repeats * windows);
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;

    for rep in 0..repeats {
        // Stable scenario
        {
            let mut det = make_detector(cfg.clone());
            let base = &stable_windows[0];
            det.add_window(0, base);
            for (wi, win) in stable_windows[1..].iter().enumerate() {
                let t0 = Instant::now();
                det.add_window((wi + 1) as u64, win);
                let report = det.detect();
                let elapsed = t0.elapsed().as_micros() as f64;
                latencies.push(elapsed);
                if rep == 0 {
                    if report.is_drifted {
                        fp += 1;
                    } else {
                        tn += 1;
                    }
                }
            }
        }
        // Drifted scenario
        {
            let mut det = make_detector(cfg.clone());
            let base = &drifted_windows[0];
            det.add_window(0, base);
            for (wi, win) in drifted_windows[1..].iter().enumerate() {
                let t0 = Instant::now();
                det.add_window((wi + 1) as u64, win);
                let report = det.detect();
                let elapsed = t0.elapsed().as_micros() as f64;
                latencies.push(elapsed);
                if rep == 0 {
                    // True label: final 30% of drift windows should be drifted
                    let is_truly_drifted = wi >= windows / 2 - 3;
                    if is_truly_drifted {
                        if report.is_drifted {
                            tp += 1;
                        } else {
                            fn_ += 1;
                        }
                    }
                }
            }
        }
    }

    let total = latencies.len() as f64;
    let mean_us = latencies.iter().sum::<f64>() / total;
    let p50_us = percentile(latencies.clone(), 50.0);
    let p95_us = percentile(latencies, 95.0);
    let total_duration_s = mean_us * total / 1_000_000.0;
    let throughput_wps = total / total_duration_s.max(1e-9);

    let tp_rate = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let fp_rate = if fp + tn > 0 {
        fp as f64 / (fp + tn) as f64
    } else {
        0.0
    };

    // Memory estimate: two windows of n × d × 4 bytes
    let mem_kb = (2 * n * d * 4) as f64 / 1024.0;

    BenchResult {
        name,
        n,
        d,
        windows,
        mean_us,
        p50_us,
        p95_us,
        throughput_wps,
        tp_rate,
        fp_rate,
        mem_kb,
    }
}

fn print_env() {
    println!("=== ruvector-drift benchmark ===");
    println!("Date:       2026-06-11");
    println!("OS:         {}", std::env::consts::OS);
    println!("Arch:       {}", std::env::consts::ARCH);
    println!("Rust:       {}", rustc_version());
    println!();
}

fn rustc_version() -> String {
    // Read from env var set by Cargo's rustc wrapper when available, else hardcode known
    std::env::var("RUSTC_BOOTSTRAP").unwrap_or_else(|_| "stable (see cargo --version)".into())
}

fn print_table(results: &[BenchResult]) {
    println!(
        "{:<18} {:>6} {:>4} {:>8} {:>8} {:>8} {:>10} {:>9} {:>7} {:>7} {:>9}",
        "Variant",
        "N",
        "D",
        "mean_us",
        "p50_us",
        "p95_us",
        "wps",
        "mem_KB",
        "TPR",
        "FPR",
        "Acceptance"
    );
    println!("{}", "-".repeat(105));
    for r in results {
        let accept = if r.tp_rate >= 0.66 && r.fp_rate <= 0.33 {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<18} {:>6} {:>4} {:>8.2} {:>8.2} {:>8.2} {:>10.0} {:>9.1} {:>7.2} {:>7.2} {:>9}",
            r.name,
            r.n,
            r.d,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.throughput_wps,
            r.mem_kb,
            r.tp_rate,
            r.fp_rate,
            accept
        );
    }
}

fn main() {
    print_env();
    let cfg = DriftConfig::default();
    let repeats = 20usize;

    let configs: &[(usize, usize, f32)] = &[(500, 64, 2.0), (1_000, 128, 2.0), (5_000, 128, 2.0)];

    let mut all_results: Vec<BenchResult> = Vec::new();

    for &(n, d, drift) in configs {
        println!("--- Dataset N={n}, D={d}, drift_amount={drift:.1} ---");

        let r0 = bench_detector("CentroidDrift", &cfg, n, d, repeats, drift, |c| {
            CentroidDrift::new(c)
        });
        let r1 = bench_detector("PsiDrift", &cfg, n, d, repeats, drift, |c| PsiDrift::new(c));
        let r2 = bench_detector("CoherenceDrift", &cfg, n, d, repeats, drift, |c| {
            CoherenceDrift::new(c)
        });
        print_table(&[r0, r1, r2]);
        println!();
        all_results.extend(vec![
            bench_detector("CentroidDrift", &cfg, n, d, repeats, drift, |c| {
                CentroidDrift::new(c)
            }),
            bench_detector("PsiDrift", &cfg, n, d, repeats, drift, |c| PsiDrift::new(c)),
            bench_detector("CoherenceDrift", &cfg, n, d, repeats, drift, |c| {
                CoherenceDrift::new(c)
            }),
        ]);
    }

    // Acceptance criterion: at least 2 of 3 variants must PASS TPR ≥ 0.66
    let passing = all_results
        .iter()
        .filter(|r| r.tp_rate >= 0.66 && r.fp_rate <= 0.33)
        .count();
    let total = all_results.len();
    println!("=== Acceptance summary ===");
    println!("Detectors passing TPR≥0.66 + FPR≤0.33: {passing}/{total}");
    if passing >= total * 2 / 3 {
        println!("ACCEPTANCE RESULT: PASS");
    } else {
        println!("ACCEPTANCE RESULT: FAIL");
        std::process::exit(1);
    }
}
