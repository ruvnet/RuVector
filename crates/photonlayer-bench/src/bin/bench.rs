//! PhotonLayer benchmark runner.
//!
//! Usage:
//!   photonlayer-bench [classification|compression|all] [--json]

use photonlayer_bench::baselines::{run_classification, run_compression, BenchReport};
use photonlayer_bench::learn::LearnConfig;

fn print_report(title: &str, r: &BenchReport) {
    println!("\n== {title} ==");
    println!("grid={} feature_dim={}", r.grid, r.feature_dim);
    println!(
        "{:<26} {:>10} {:>10} {:>10}",
        "variant", "train_acc", "test_acc", "params"
    );
    for v in &r.variants {
        println!(
            "{:<26} {:>10.3} {:>10.3} {:>10}",
            v.name, v.train_accuracy, v.test_accuracy, v.decoder_params
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("all");
    let json = args.iter().any(|a| a == "--json");

    let lc = LearnConfig {
        iterations: 200,
        ..Default::default()
    };

    let mut reports: Vec<(String, BenchReport)> = Vec::new();
    if mode == "classification" || mode == "all" {
        reports.push(("classification".into(), run_classification(16, 8, &lc)));
    }
    if mode == "compression" || mode == "all" {
        // Squeeze a 16x16 input down to a 2x2 (=4-pixel) sensor.
        reports.push((
            "compression(2x2 sensor)".into(),
            run_compression(16, 10, 2, &lc),
        ));
    }

    if json {
        let map: std::collections::BTreeMap<_, _> = reports.iter().cloned().collect();
        println!("{}", serde_json::to_string_pretty(&map).unwrap());
    } else {
        for (title, r) in &reports {
            print_report(title, r);
        }
        println!("\nClaim (ADR-260 §16.3): a learned optical frontend preserves");
        println!("task-useful information while shrinking the sensor/decoder.");
    }
}
