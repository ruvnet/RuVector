//! Proof-gate write throughput benchmark.
//!
//! Measures three gate variants across configurable dataset sizes and
//! dimensions. Prints per-variant stats: mean latency, p50, p95,
//! throughput, and memory estimate.
//!
//! Usage:
//!   cargo run --release -p ruvector-proof-gate --example benchmark
//!   cargo run --release -p ruvector-proof-gate --example benchmark -- 50000 128
//!   cargo run --release -p ruvector-proof-gate --example benchmark -- 10000 384

use std::time::{Duration, Instant};

use ruvector_proof_gate::{
    synthetic_payloads, HashChainGate, MerkleGate, NullGate, WriteGate, WritePayload,
};

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct BenchResult {
    variant: &'static str,
    n: usize,
    dims: usize,
    #[allow(dead_code)]
    queries: usize,
    mean_ns: f64,
    p50_ns: u64,
    p95_ns: u64,
    throughput_per_sec: f64,
    mem_bytes: usize,
    chain_root: [u8; 32],
    verify_ok: bool,
}

fn run_bench<G: WriteGate>(
    name: &'static str,
    gate: &mut G,
    payloads: &[WritePayload],
    dims: usize,
) -> BenchResult {
    let n = payloads.len();
    let mut latencies: Vec<Duration> = Vec::with_capacity(n);
    let mut receipts = Vec::with_capacity(n);

    for payload in payloads {
        let t0 = Instant::now();
        let receipt = gate.admit(payload).expect("admission failed");
        latencies.push(t0.elapsed());
        receipts.push(receipt);
    }

    latencies.sort_unstable();

    let total_ns: u128 = latencies.iter().map(|d| d.as_nanos()).sum();
    let mean_ns = total_ns as f64 / n as f64;
    let p50 = percentile(&latencies, 50.0).as_nanos() as u64;
    let p95 = percentile(&latencies, 95.0).as_nanos() as u64;
    let elapsed_secs = total_ns as f64 / 1_000_000_000.0;
    let throughput = n as f64 / elapsed_secs;

    // Verify a sample of receipts
    let verify_ok = receipts
        .iter()
        .step_by(10.max(n / 100))
        .all(|r| gate.verify_receipt(r));

    // Memory estimate: receipt overhead + internal gate state
    // HashChainGate stores 32 bytes per entry in its chain vec.
    // MerkleGate stores 32 bytes per leaf + ~2*log2(n)*32 for peaks.
    // NullGate: negligible.
    let mem_bytes = match gate.variant() {
        ruvector_proof_gate::GateVariant::Null => 8,
        ruvector_proof_gate::GateVariant::HashChain => n * 32,
        ruvector_proof_gate::GateVariant::Merkle => {
            let peaks_est = (n.max(1) as f64).log2().ceil() as usize + 1;
            n * 32 + peaks_est * 32
        }
    };

    BenchResult {
        variant: name,
        n,
        dims,
        queries: n,
        mean_ns,
        p50_ns: p50,
        p95_ns: p95,
        throughput_per_sec: throughput,
        mem_bytes,
        chain_root: gate.chain_root(),
        verify_ok,
    }
}

fn print_header() {
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│                 ruvector-proof-gate: Write Gate Benchmark                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

fn print_env(n: usize, dims: usize) {
    println!("  OS:           {}", std::env::consts::OS);
    println!("  Arch:         {}", std::env::consts::ARCH);
    println!("  Dataset:      {} vectors × {} dims", n, dims);
    println!("  Queries:      {} (same as writes)", n);
    println!("  Build:        release");
    println!();
}

fn print_result(r: &BenchResult) {
    println!("── {} ──", r.variant);
    println!("  Dataset:       {} vectors × {} dims", r.n, r.dims);
    println!("  Mean latency:  {:.1} ns", r.mean_ns);
    println!("  p50 latency:   {} ns", r.p50_ns);
    println!("  p95 latency:   {} ns", r.p95_ns);
    println!("  Throughput:    {:.0} writes/sec", r.throughput_per_sec);
    println!(
        "  Memory est.:   {} bytes ({:.1} KB)",
        r.mem_bytes,
        r.mem_bytes as f64 / 1024.0
    );
    println!("  Chain root:    {}...", hex_prefix(&r.chain_root));
    println!(
        "  Receipt verify: {}",
        if r.verify_ok { "PASS" } else { "FAIL" }
    );
    println!();
}

fn hex_prefix(b: &[u8; 32]) -> String {
    b[..8]
        .iter()
        .map(|x| format!("{:02x}", x))
        .collect::<Vec<_>>()
        .join("")
}

fn acceptance_check(results: &[BenchResult]) -> bool {
    // Acceptance thresholds (absolute, release build, x86_64):
    // NullGate is compared against itself only for receipt verify.
    // HashChain must sustain >50K writes/sec with SHA-256 (conservative for
    //   any modern CPU doing one SHA-256 per write of ~512 bytes payload).
    // Merkle must sustain >20K writes/sec (O(log n) SHA-256 per write).
    // Receipt verification must pass for all sampled receipts.
    // Chain roots must be non-zero (proves the chain/MMR is active).
    const HASH_CHAIN_MIN_TP: f64 = 50_000.0;
    const MERKLE_MIN_TP: f64 = 20_000.0;

    let mut pass = true;

    for r in results {
        if !r.verify_ok {
            eprintln!("FAIL: receipt verification failed for {}", r.variant);
            pass = false;
        }
        match r.variant {
            "HashChainGate" => {
                if r.throughput_per_sec < HASH_CHAIN_MIN_TP {
                    eprintln!(
                        "FAIL: HashChainGate {:.0} writes/sec < minimum {:.0}",
                        r.throughput_per_sec, HASH_CHAIN_MIN_TP
                    );
                    pass = false;
                }
                if r.chain_root == [0u8; 32] {
                    eprintln!("FAIL: HashChainGate root is zero (chain inactive)");
                    pass = false;
                }
            }
            "MerkleGate" => {
                if r.throughput_per_sec < MERKLE_MIN_TP {
                    eprintln!(
                        "FAIL: MerkleGate {:.0} writes/sec < minimum {:.0}",
                        r.throughput_per_sec, MERKLE_MIN_TP
                    );
                    pass = false;
                }
                if r.chain_root == [0u8; 32] {
                    eprintln!("FAIL: MerkleGate root is zero (MMR inactive)");
                    pass = false;
                }
            }
            _ => {}
        }
    }

    pass
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    print_header();
    print_env(n, dims);

    let payloads = synthetic_payloads(n, dims);

    let mut null_gate = NullGate::new();
    let r_null = run_bench("NullGate", &mut null_gate, &payloads, dims);

    let mut chain_gate = HashChainGate::new();
    let r_chain = run_bench("HashChainGate", &mut chain_gate, &payloads, dims);

    let mut merkle_gate = MerkleGate::new();
    let r_merkle = run_bench("MerkleGate", &mut merkle_gate, &payloads, dims);

    // Tabular summary
    println!(
        "{:<16} {:>8} {:>8} {:>10} {:>10} {:>14} {:>10}",
        "Variant", "N", "Dims", "Mean(ns)", "p50(ns)", "Throughput/s", "Mem(KB)"
    );
    println!("{}", "─".repeat(82));
    for r in &[&r_null, &r_chain, &r_merkle] {
        println!(
            "{:<16} {:>8} {:>8} {:>10.1} {:>10} {:>14.0} {:>10.1}",
            r.variant,
            r.n,
            r.dims,
            r.mean_ns,
            r.p50_ns,
            r.throughput_per_sec,
            r.mem_bytes as f64 / 1024.0,
        );
    }
    println!();

    for r in &[&r_null, &r_chain, &r_merkle] {
        print_result(r);
    }

    // Overhead analysis
    let null_mean = r_null.mean_ns;
    let chain_overhead = r_chain.mean_ns - null_mean;
    let merkle_overhead = r_merkle.mean_ns - null_mean;
    println!("── Overhead Analysis ──");
    println!(
        "  HashChain overhead vs NullGate: {:.1} ns/write",
        chain_overhead
    );
    println!(
        "  Merkle overhead vs NullGate:    {:.1} ns/write",
        merkle_overhead
    );
    println!(
        "  HashChain throughput ratio:     {:.4}",
        r_chain.throughput_per_sec / r_null.throughput_per_sec
    );
    println!(
        "  Merkle throughput ratio:        {:.4}",
        r_merkle.throughput_per_sec / r_null.throughput_per_sec
    );
    println!();

    let all = vec![r_null, r_chain, r_merkle];
    let pass = acceptance_check(&all);

    if pass {
        println!("ACCEPTANCE: PASS — all thresholds met.");
        std::process::exit(0);
    } else {
        println!("ACCEPTANCE: FAIL — see diagnostics above.");
        std::process::exit(1);
    }
}
