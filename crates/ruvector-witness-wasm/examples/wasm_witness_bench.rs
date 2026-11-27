//! Benchmark: does the ADR-347 `PerRecord`-vs-`BatchTail` amortization
//! result survive a wasm32-unknown-unknown port, and what does the port
//! cost in binary size and wasmtime-sandboxed latency relative to native?
//!
//! Reproduce:
//! ```bash
//! cargo build --target wasm32-unknown-unknown --release -p ruvector-witness-wasm
//! cargo run --release -p ruvector-witness-wasm --example wasm_witness_bench
//! ```

use std::time::Instant;

use ruvector_witness_wasm::{run_workload, tamper_check, Strategy};
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

/// Records per benchmark call. Chosen smaller than the native-only
/// 2026-09-16 nightly's N=20,000 so that REPEATS whole-workload calls
/// (through wasmtime, not just natively) finish in a practical time
/// while still giving >1ms-scale per-call totals for stable timing.
const N: u32 = 5_000;
/// Whole-workload repeats per variant, for run-level percentiles.
const REPEATS: usize = 15;
const BATCH_SIZE: u32 = 64;

const WASM_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/wasm32-unknown-unknown/release/ruvector_witness_wasm.wasm"
);

struct Stats {
    mean_ns: f64,
    p50_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
}

fn stats(mut samples_ns: Vec<f64>) -> Stats {
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ns.len();
    let mean_ns = samples_ns.iter().sum::<f64>() / n as f64;
    let q = |p: f64| -> f64 {
        let idx = (((n - 1) as f64) * p).round() as usize;
        samples_ns[idx]
    };
    Stats {
        mean_ns,
        p50_ns: q(0.50),
        p95_ns: q(0.95),
        p99_ns: q(0.99),
    }
}

fn print_row(label: &str, s: &Stats, signature_count: u32, per_op: bool) {
    let div = if per_op { N as f64 } else { 1.0 };
    println!(
        "{label:<28} mean={:>10.1}us  p50={:>10.1}us  p95={:>10.1}us  p99={:>10.1}us  signatures={signature_count:>6}",
        s.mean_ns / div / 1000.0,
        s.p50_ns / div / 1000.0,
        s.p95_ns / div / 1000.0,
        s.p99_ns / div / 1000.0,
    );
}

fn bench_native(strategy: Strategy, batch_size: u32) -> (Stats, u32) {
    let mut samples = Vec::with_capacity(REPEATS);
    let mut signature_count = 0;
    for _ in 0..REPEATS {
        let start = Instant::now();
        let r = run_workload(strategy, N, batch_size);
        samples.push(start.elapsed().as_nanos() as f64);
        assert_eq!(
            r.verify_failures, 0,
            "native honest run must verify cleanly"
        );
        signature_count = r.signature_count;
    }
    (stats(samples), signature_count)
}

fn bench_wasm(
    store: &mut Store<()>,
    func: &TypedFunc<(i32, u32, u32), i64>,
    strategy: Strategy,
    batch_size: u32,
) -> anyhow::Result<(Stats, u32)> {
    let mut samples = Vec::with_capacity(REPEATS);
    let mut signature_count = 0u32;
    for _ in 0..REPEATS {
        let start = Instant::now();
        let packed = func.call(&mut *store, (strategy as i32, N, batch_size))?;
        samples.push(start.elapsed().as_nanos() as f64);
        let verify_failures = (packed & 0xFFFF_FFFF) as u32;
        assert_eq!(verify_failures, 0, "wasm honest run must verify cleanly");
        signature_count = ((packed >> 32) & 0xFFFF_FFFF) as u32;
    }
    Ok((stats(samples), signature_count))
}

fn main() -> anyhow::Result<()> {
    println!("ruvector-witness-wasm portability benchmark");
    println!("N={N} records/call, REPEATS={REPEATS} whole-workload calls/variant, BATCH_SIZE={BATCH_SIZE}");
    println!();

    // --- Binary size ---
    let wasm_bytes = std::fs::read(WASM_PATH).map_err(|e| {
        anyhow::anyhow!(
            "could not read {WASM_PATH}: {e}\nrun: cargo build --target wasm32-unknown-unknown --release -p ruvector-witness-wasm"
        )
    })?;
    let wasm_size_bytes = wasm_bytes.len();
    println!(
        "wasm module size (release, no wasm-opt): {wasm_size_bytes} bytes ({:.1} KiB)",
        wasm_size_bytes as f64 / 1024.0
    );

    // --- Compile + instantiate ("startup cost") ---
    let engine = Engine::default();
    let compile_start = Instant::now();
    let module = Module::new(&engine, &wasm_bytes)?;
    let compile_elapsed = compile_start.elapsed();

    let mut store = Store::new(&engine, ());
    let instantiate_start = Instant::now();
    let instance = Instance::new(&mut store, &module, &[])?;
    let instantiate_elapsed = instantiate_start.elapsed();

    println!(
        "wasmtime module compile time:    {:.3} ms",
        compile_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "wasmtime instance instantiate time: {:.3} ms",
        instantiate_elapsed.as_secs_f64() * 1000.0
    );

    if let Some(memory) = instance.get_memory(&mut store, "memory") {
        let pages = memory.size(&store);
        println!(
            "wasm linear memory after instantiate: {pages} page(s) = {} KiB",
            pages * 64
        );
    } else {
        println!("wasm linear memory: no exported \"memory\" found");
    }
    println!();

    let run_workload_fn: TypedFunc<(i32, u32, u32), i64> =
        instance.get_typed_func(&mut store, "wasm_run_workload")?;
    let tamper_check_fn: TypedFunc<(i32, u32, u32), i32> =
        instance.get_typed_func(&mut store, "wasm_tamper_check")?;

    // --- Latency: native baseline vs wasmtime-sandboxed, both strategies ---
    let (native_per_record, sig_native_pr) = bench_native(Strategy::PerRecord, 1);
    let (native_batch_tail, sig_native_bt) = bench_native(Strategy::BatchTail, BATCH_SIZE);
    let (wasm_per_record, sig_wasm_pr) =
        bench_wasm(&mut store, &run_workload_fn, Strategy::PerRecord, 1)?;
    let (wasm_batch_tail, sig_wasm_bt) = bench_wasm(
        &mut store,
        &run_workload_fn,
        Strategy::BatchTail,
        BATCH_SIZE,
    )?;

    println!("Per-operation latency (mean/op = whole-call time / N, percentiles across {REPEATS} whole-call repeats):");
    print_row("native_per_record", &native_per_record, sig_native_pr, true);
    print_row(
        "native_batch_tail64",
        &native_batch_tail,
        sig_native_bt,
        true,
    );
    print_row("wasm_per_record", &wasm_per_record, sig_wasm_pr, true);
    print_row("wasm_batch_tail64", &wasm_batch_tail, sig_wasm_bt, true);
    println!();

    let native_amortization = native_per_record.mean_ns / native_batch_tail.mean_ns;
    let wasm_amortization = wasm_per_record.mean_ns / wasm_batch_tail.mean_ns;
    let wasm_overhead_per_record = wasm_per_record.mean_ns / native_per_record.mean_ns;
    let wasm_overhead_batch_tail = wasm_batch_tail.mean_ns / native_batch_tail.mean_ns;

    println!("native amortization (per_record / batch_tail64):     {native_amortization:.2}x");
    println!("wasm amortization   (per_record / batch_tail64):     {wasm_amortization:.2}x");
    println!("wasm/native overhead (per_record):                   {wasm_overhead_per_record:.2}x");
    println!("wasm/native overhead (batch_tail64):                 {wasm_overhead_batch_tail:.2}x");
    println!();

    // --- Correctness: diligent-forgery rejection, native and wasm ---
    let native_tamper_pr = tamper_check(Strategy::PerRecord, N, 1);
    let native_tamper_bt = tamper_check(Strategy::BatchTail, N, BATCH_SIZE);
    let wasm_tamper_pr = tamper_check_fn.call(&mut store, (Strategy::PerRecord as i32, N, 1))? != 0;
    let wasm_tamper_bt =
        tamper_check_fn.call(&mut store, (Strategy::BatchTail as i32, N, BATCH_SIZE))? != 0;

    println!("Diligent-forgery rejection (true = correctly detected):");
    println!("  native per_record     = {native_tamper_pr}");
    println!("  native batch_tail64   = {native_tamper_bt}");
    println!("  wasm   per_record     = {wasm_tamper_pr}");
    println!("  wasm   batch_tail64   = {wasm_tamper_bt}");
    println!();

    // --- Acceptance gates (fixed before this run; see research doc) ---
    let gate_size = wasm_size_bytes < 1024 * 1024;
    let gate_wasm_amortization = wasm_amortization >= 5.0;
    let gate_overhead_per_record = wasm_overhead_per_record <= 10.0;
    let gate_overhead_batch_tail = wasm_overhead_batch_tail <= 10.0;
    let gate_tamper = native_tamper_pr && native_tamper_bt && wasm_tamper_pr && wasm_tamper_bt;

    println!("Acceptance gates:");
    println!(
        "  wasm size < 1 MiB:                          {} ({:.1} KiB)",
        pass(gate_size),
        wasm_size_bytes as f64 / 1024.0
    );
    println!(
        "  wasm BatchTail amortization >= 5x:          {} ({wasm_amortization:.2}x)",
        pass(gate_wasm_amortization)
    );
    println!(
        "  wasm/native overhead (per_record) <= 10x:   {} ({wasm_overhead_per_record:.2}x)",
        pass(gate_overhead_per_record)
    );
    println!(
        "  wasm/native overhead (batch_tail64) <= 10x: {} ({wasm_overhead_batch_tail:.2}x)",
        pass(gate_overhead_batch_tail)
    );
    println!(
        "  diligent-forgery rejected (4/4 variants):   {}",
        pass(gate_tamper)
    );

    let accepted = gate_size
        && gate_wasm_amortization
        && gate_overhead_per_record
        && gate_overhead_batch_tail
        && gate_tamper;
    println!();
    println!("ACCEPTANCE: {}", if accepted { "ACCEPT" } else { "REJECT" });

    Ok(())
}

fn pass(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
