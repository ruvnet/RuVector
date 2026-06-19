//! Larger-data PhotonLayer benchmark + WIN regression guard.
//!
//! Runs the learned-vs-random-vs-direct comparison with more samples per class
//! and a **compression sweep** across sensor sizes, so the flagship claim
//! ("a learned optical mask preserves task-useful info under heavy pixel
//! reduction") is measured on more data and CI-guarded — not just a single point.
//!
//! Run: `cargo test -p photonlayer-bench --release --test more_data_bench -- --nocapture`

use photonlayer_bench::baselines::{run_classification, run_compression, BenchReport};
use photonlayer_bench::learn::LearnConfig;

const GRID: usize = 16; // 16x16 input
const PER_CLASS: usize = 40; // 4x the bin's default — "more data"
const ITERS: usize = 300;

fn acc(report: &BenchReport, needle: &str) -> f32 {
    report
        .variants
        .iter()
        .find(|v| v.name.to_lowercase().contains(needle))
        .map(|v| v.test_accuracy)
        .unwrap_or(-1.0)
}

fn print_report(title: &str, r: &BenchReport) {
    println!(
        "\n== {title} ==  grid={} feature_dim={}",
        r.grid, r.feature_dim
    );
    println!(
        "  {:<22} {:>8} {:>8} {:>10}",
        "variant", "train", "test", "dec.params"
    );
    for v in &r.variants {
        println!(
            "  {:<22} {:>8.3} {:>8.3} {:>10}",
            v.name, v.train_accuracy, v.test_accuracy, v.decoder_params
        );
    }
}

#[test]
fn compression_sweep_more_data() {
    let lc = LearnConfig {
        iterations: ITERS,
        ..Default::default()
    };

    // Sensor side -> pixel count: 1->1, 2->4, 3->9, 4->16 (input is 16x16 = 256 px).
    println!("\nPhotonLayer compression sweep (grid={GRID}, per_class={PER_CLASS}, iters={ITERS})");
    println!(
        "input pixels = {} ; learned vs random optical mask vs direct pixel read",
        GRID * GRID
    );

    let mut learned_wins = 0;
    let sensors = [1usize, 2, 3, 4];
    for &s in &sensors {
        let r = run_compression(GRID, PER_CLASS, s, &lc);
        let px = s * s;
        let ratio = (GRID * GRID) as f32 / px as f32;
        print_report(
            &format!("compression {s}x{s} sensor = {px}px ({ratio:.0}x reduction)"),
            &r,
        );

        let learned = acc(&r, "learn");
        let random = acc(&r, "random");
        let direct = acc(&r, "direct").max(acc(&r, "pixel"));
        println!(
            "  -> learned={learned:.3}  random={random:.3}  direct={direct:.3}  [{px}px, {ratio:.0}x]"
        );
        // WIN guard: the learned mask must beat both baselines at the tightest sensors.
        if learned >= random && learned >= direct {
            learned_wins += 1;
        }
    }

    // The learned optical front end must win on the majority of the sweep.
    assert!(
        learned_wins >= sensors.len() - 1,
        "learned mask should beat random+direct on >= {} of {} sensor sizes; won {}",
        sensors.len() - 1,
        sensors.len(),
        learned_wins
    );
}

#[test]
fn classification_more_data() {
    let lc = LearnConfig {
        iterations: ITERS,
        ..Default::default()
    };
    let r = run_classification(GRID, PER_CLASS, &lc);
    print_report("classification (more data)", &r);
    let learned = acc(&r, "learn");
    let baseline = acc(&r, "random")
        .max(acc(&r, "direct"))
        .max(acc(&r, "digital"));
    println!("  -> learned={learned:.3}  best_baseline={baseline:.3}");
    assert!(
        learned > 0.0,
        "learned variant must be present and produce a score"
    );
}
