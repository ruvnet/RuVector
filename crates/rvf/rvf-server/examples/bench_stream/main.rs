//! Project Ofer benchmarks B1/B2/B6/B7/B8 (plain harness; run with --release).
//!
//!   cargo run --release -p rvf-server --example bench_stream -- \
//!       [--n 100000] [--dim 128] [--clusters 64] [--cache DIR] [--only b1,b2,...]
//!
//! Prints the load average before and after, then one table per benchmark.

mod b1_b2;
mod b6_b8;
mod common;

use common::*;
use std::path::PathBuf;

fn main() {
    let n: usize = arg("--n", "100000").parse().unwrap();
    let dim: usize = arg("--dim", "128").parse().unwrap();
    let clusters: usize = arg("--clusters", "64").parse().unwrap();
    let cache = PathBuf::from(arg("--cache", "/data/scratch/ofer-wt/cache"));
    let only = arg("--only", "b1,b2,b6,b7,b8");
    println!("host load average before: {}", loadavg());
    println!(
        "n={n} dim={dim} clusters={clusters} release={}",
        !cfg!(debug_assertions)
    );
    let f = if only == "b9" || only == "hash" {
        None
    } else {
        Some(fixture(n, dim, clusters, &cache))
    };
    for w in only.split(',') {
        match w {
            "b1" => b1_b2::b1(f.as_ref().unwrap(), n, dim),
            "b2" => b1_b2::b2(f.as_ref().unwrap()),
            "b6" => b6_b8::b6(f.as_ref().unwrap()),
            "b7" => b6_b8::b7(),
            "hash" => b6_b8::hash_backends(),
            "b9" => b6_b8::b9(),
            "b10" => b6_b8::b10(f.as_ref().unwrap()),
            "b8" => b6_b8::b8(f.as_ref().unwrap()),
            _ => {}
        }
    }
    println!("\nhost load average after: {}", loadavg());
}
