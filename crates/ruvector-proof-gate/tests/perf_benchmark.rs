//! Integrity-tax benchmark for the write gates (productionizes #506).
//!
//! Measures the cost of tamper-evident writes: `HashChainGate::admit` (2× SHA-256
//! per write: payload hash + chain commitment) and `verify_receipt`, vs the
//! `NullGate` baseline. Reports ns/op + ops/sec so CI can track the tax.
//!
//! Run: `cargo test -p ruvector-proof-gate --release --test perf_benchmark -- --ignored --nocapture`

use std::time::Instant;

use ruvector_proof_gate::{HashChainGate, NullGate, WriteGate, WritePayload};

const DIM: usize = 128;
const N: usize = 20_000;
// Generous per-write budget (ns) for ~2 SHA-256 over a ~600-byte payload.
const ADMIT_BUDGET_NS: f64 = 5_000.0;

fn payloads(n: usize) -> Vec<WritePayload> {
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..DIM).map(|d| (i + d) as f32 * 0.5).collect();
            WritePayload::new(i as u64, v)
                .with_agent([7u8; 16])
                .with_timestamp(i as u64)
        })
        .collect()
}

fn time_admit<G: WriteGate>(
    mut gate: G,
    ps: &[WritePayload],
) -> (f64, Vec<ruvector_proof_gate::WriteReceipt>) {
    // warm up
    for p in ps.iter().take(64) {
        let _ = gate.admit(p);
    }
    let mut g = gate; // fresh state after warmup not required for relative timing
    let mut receipts = Vec::with_capacity(ps.len());
    let t = Instant::now();
    for p in ps {
        receipts.push(g.admit(p).unwrap());
    }
    let ns = t.elapsed().as_secs_f64() / ps.len() as f64 * 1e9;
    let _ = &g;
    (ns, receipts)
}

#[test]
#[ignore = "perf benchmark; run with: cargo test --release -- --ignored"]
fn integrity_tax() {
    let ps = payloads(N);

    let (null_ns, _) = time_admit(NullGate::new(), &ps);
    let (hash_ns, receipts) = time_admit(HashChainGate::new(), &ps);

    // verify throughput on the hash chain
    let g = {
        let mut g = HashChainGate::new();
        for p in &ps {
            g.admit(p).unwrap();
        }
        g
    };
    let t = Instant::now();
    let mut ok = 0usize;
    for r in &receipts {
        if g.verify_receipt(r) {
            ok += 1;
        }
    }
    let verify_ns = t.elapsed().as_secs_f64() / receipts.len() as f64 * 1e9;
    std::hint::black_box(ok);

    eprintln!("integrity tax (DIM={DIM}, N={N}):");
    eprintln!(
        "  NullGate.admit       {null_ns:8.1} ns/write   {:.2} M/s",
        1e3 / null_ns.max(1e-6)
    );
    eprintln!(
        "  HashChainGate.admit  {hash_ns:8.1} ns/write   {:.2} M/s",
        1e3 / hash_ns
    );
    eprintln!(
        "  verify_receipt       {verify_ns:8.1} ns/op     {:.2} M/s",
        1e3 / verify_ns.max(1e-6)
    );
    eprintln!(
        "  integrity tax: {:.0} ns/write over the unguarded baseline",
        hash_ns - null_ns
    );

    assert!(
        hash_ns < ADMIT_BUDGET_NS,
        "HashChainGate.admit {hash_ns:.0} ns/write exceeds {ADMIT_BUDGET_NS} ns budget (regression)"
    );
}
