//! Signed eviction-witness anchoring: cost, memory, and adversarial
//! detection benchmark (PIR nightly 2026-09-09, ADR-346).
//!
//! Continues the 2026-09-05 nightly run's "Next Research" item 4: wire an
//! Ed25519 `WitnessSigner` into `ruvector-agent-memory`'s eviction-witness
//! chain (ADR-345's `compact_witnessed`) so eviction receipts are signed,
//! not just FNV-1a hash-chained. This binary drives the REAL
//! `eviction_witness_signing` module (real `compact_witnessed` evictions,
//! real `rvf_types` Ed25519 signs/verifies, real relink-attack simulation)
//! against three variants:
//!
//!   baseline    — FNV-1a chaining only (status quo, no signing)
//!   candidate_a — sign every evicted record  (interval_records = 1)
//!   candidate_b — sign periodically          (interval_records = K, K>1)
//!
//! and reports, per variant: signing cost, verify cost, memory overhead,
//! anchor count, max staleness, and — the central falsifiable claim — an
//! empirical adversarial-detection rate against a real relink attack, not
//! an assumed one.
//!
//! Usage:
//!   cargo run --release -p ruvector-agent-memory --example signed_eviction_witness_bench

use std::mem::size_of;
use std::time::Instant;

use ruvector_agent_memory::compaction::LruPolicy;
use ruvector_agent_memory::memory::MemoryStore;
use ruvector_agent_memory::ops::MemoryWitnessLog;
use ruvector_agent_memory::witnessed_compaction::{compact_witnessed, EvictionWitnessChain};
use ruvector_agent_memory::{
    verify_anchor_against_chain, EvictionAnchorLog, EvictionAnchorPolicy, SignedEvictionAnchor,
};
use rvf_types::Ed25519Keypair;

const RECORD_BYTES: usize = 64; // ADR-134 fixed record size

/// Build one long eviction-witness chain of `n` evicted records (single
/// `compact_witnessed` call: insert `n`, keep 0, so every insert is
/// eventually evicted — deterministic, no RNG needed).
fn build_chain(n: usize) -> Vec<ruvector_agent_memory::LedgerWitnessRecord> {
    let mut store = MemoryStore::new(2);
    for i in 0..n {
        store.insert(vec![i as f32, (i % 7) as f32]);
    }
    let mut chain = EvictionWitnessChain::new();
    let mut log = MemoryWitnessLog::default();
    compact_witnessed(
        &mut store,
        &LruPolicy,
        0,
        &[],
        "signed-eviction-bench",
        1_000,
        &mut chain,
        &mut log,
    )
    .expect("compaction to empty must succeed");
    assert!(log.verify_chain(), "freshly built chain must self-verify");
    log.records
}

fn ns_per_op(elapsed: std::time::Duration, ops: usize) -> f64 {
    if ops == 0 {
        return 0.0;
    }
    elapsed.as_nanos() as f64 / ops as f64
}

fn main() {
    let n_records = 4_096usize;
    let records = build_chain(n_records);
    println!(
        "Built eviction-witness chain: {n_records} records ({} bytes FNV-1a chain, no signing)\n",
        n_records * RECORD_BYTES
    );

    let kp = Ed25519Keypair::generate(&mut rand::rngs::OsRng);

    println!("== Cost and memory by anchoring policy ==");
    println!(
        "{:<12} {:>10} {:>14} {:>16} {:>14} {:>16}",
        "variant", "interval", "anchors", "sign_ns/anchor", "max_stale", "amortized_ns/rec"
    );

    let intervals: &[u64] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];
    let mut results: Vec<(u64, EvictionAnchorLog, f64, f64)> = Vec::new();

    for &k in intervals {
        let policy = EvictionAnchorPolicy::new(k).expect("k >= 1");
        let mut alog = EvictionAnchorLog::new(policy);

        let t0 = Instant::now();
        for r in &records {
            alog.note_record(&kp, r, 1_000);
        }
        let sign_elapsed = t0.elapsed();

        let anchor_count = alog.anchors.len();
        let sign_ns_per_anchor = ns_per_op(sign_elapsed, anchor_count.max(1));
        let amortized_ns_per_record = sign_elapsed.as_nanos() as f64 / n_records as f64;

        // Max staleness actually observed by replaying record-by-record.
        let mut alog2 = EvictionAnchorLog::new(policy);
        let mut max_stale = 0u64;
        for r in &records {
            alog2.note_record(&kp, r, 1_000);
            max_stale = max_stale.max(alog2.staleness_at(alog2.record_count()));
        }

        let variant = if k == 1 { "candidate_a" } else { "candidate_b" };
        println!(
            "{:<12} {:>10} {:>14} {:>16.1} {:>14} {:>16.1}",
            variant, k, anchor_count, sign_ns_per_anchor, max_stale, amortized_ns_per_record
        );
        results.push((k, alog, sign_ns_per_anchor, amortized_ns_per_record));
    }
    println!(
        "{:<12} {:>10} {:>14} {:>16} {:>14} {:>16}\n",
        "baseline", "n/a", 0, 0.0, "n/a", 0.0
    );

    println!("== Verify cost ==");
    {
        let (_, alog_k16, _, _) = &results[intervals.iter().position(|&k| k == 16).unwrap()];
        let anchor = alog_k16.latest().expect("k=16 must have produced anchors");
        let iters = 10_000usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(verify_anchor_against_chain(
                &kp.public_key(),
                anchor,
                anchor.statement.chain_head,
            ));
        }
        let elapsed = t0.elapsed();
        println!(
            "verify_anchor_against_chain: {:.1} ns/op ({iters} iters)\n",
            ns_per_op(elapsed, iters)
        );
    }

    println!("== Memory overhead ==");
    let record_total_bytes = n_records * RECORD_BYTES;
    let anchor_struct_bytes = size_of::<SignedEvictionAnchor>();
    for (k, alog, _, _) in &results {
        let anchor_bytes = alog.anchors.len() * anchor_struct_bytes;
        let pct = 100.0 * anchor_bytes as f64 / record_total_bytes as f64;
        println!(
            "K={k:<6} anchor_struct={anchor_struct_bytes}B  anchors={:<6} total_anchor_bytes={anchor_bytes:<8} = {pct:.3}% of {record_total_bytes}B chain",
            alog.anchors.len()
        );
    }
    println!();

    println!(
        "== Adversarial detection: real relink attack vs. FNV-1a chaining and signed anchors =="
    );
    // 1. FNV-1a-only baseline: a fully relinked (records + head commitment)
    //    tampered chain still self-verifies. Real code, real check.
    {
        let mut tampered_log = MemoryWitnessLog {
            records: records.clone(),
            committed_count: records.len() as u64,
            committed_head: records.last().unwrap().chain_hash(),
        };
        ruvector_agent_memory::relink_tampered_suffix(
            &mut tampered_log.records,
            n_records / 2,
            |r| {
                r.target_object_id ^= 0xDEAD_BEEF;
            },
        );
        tampered_log.committed_head = tampered_log.records.last().unwrap().chain_hash();
        println!(
            "baseline (FNV-1a only): relinked tamper at record {} still verify_chain() == {}",
            n_records / 2,
            tampered_log.verify_chain()
        );
    }

    // 2. Signed anchors: for K=16, sample 500 tamper positions. For each,
    //    the auditor holds the LATEST anchor already emitted before the
    //    tamper (staleness window: expect never detected there — honest,
    //    not a flaw), and the NEXT anchor at/after the tamper (expect
    //    always detected there).
    let k = 16u64;
    let policy = EvictionAnchorPolicy::new(k).unwrap();
    let mut honest_log = EvictionAnchorLog::new(policy);
    let mut anchor_at_seq: Vec<Option<SignedEvictionAnchor>> = vec![None; n_records + 1];
    for (i, r) in records.iter().enumerate() {
        if let Some(a) = honest_log.note_record(&kp, r, 1_000) {
            anchor_at_seq[i + 1] = Some(*a);
        }
    }
    // Fill forward so anchor_at_seq[s] = most recent anchor with sequence <= s.
    let mut carry: Option<SignedEvictionAnchor> = None;
    for slot in anchor_at_seq.iter_mut() {
        if slot.is_some() {
            carry = *slot;
        } else {
            *slot = carry;
        }
    }

    let sample_positions: Vec<usize> = (0..n_records - (k as usize) - 1)
        .step_by(37) // deterministic stride, no RNG needed
        .collect();
    let mut detected_at_next_anchor = 0usize;
    let mut checked_next = 0usize;
    let mut false_positive_before_tamper = 0usize;

    for &t in &sample_positions {
        let mut tampered = records.clone();
        ruvector_agent_memory::relink_tampered_suffix(&mut tampered, t, |r| {
            r.target_object_id ^= 0xDEAD_BEEF;
        });

        // Anchor strictly before the tamper (staleness window): recompute
        // at that anchor's own sequence, must still match (it does not
        // cover the tampered region at all — no false positive).
        if t > 0 {
            if let Some(pre) = &anchor_at_seq[t] {
                let recomputed = if pre.statement.sequence == 0 {
                    0
                } else {
                    tampered[(pre.statement.sequence - 1) as usize].chain_hash()
                };
                if !verify_anchor_against_chain(&kp.public_key(), pre, recomputed) {
                    false_positive_before_tamper += 1;
                }
            }
        }

        // Next anchor at/after the tamper: must detect.
        let next_seq = ((t as u64 / k) + 1) * k; // next boundary strictly after t
        if (next_seq as usize) <= n_records {
            if let Some(next) = &anchor_at_seq[next_seq as usize] {
                if next.statement.sequence >= (t as u64 + 1) {
                    checked_next += 1;
                    let recomputed = tampered[(next.statement.sequence - 1) as usize].chain_hash();
                    if !verify_anchor_against_chain(&kp.public_key(), next, recomputed) {
                        detected_at_next_anchor += 1;
                    }
                }
            }
        }
    }

    println!("K={k}: sampled {} tamper positions", sample_positions.len());
    println!(
        "  detected once the covering anchor exists: {detected_at_next_anchor}/{checked_next} ({:.1}%)",
        100.0 * detected_at_next_anchor as f64 / checked_next.max(1) as f64
    );
    println!(
        "  false positives on anchors predating the tamper (must be 0): {false_positive_before_tamper}"
    );

    println!("\nACCEPTANCE (fixed thresholds, see docs/research/nightly/2026-09-09-signed-eviction-witness-anchoring/README.md):");
    let all_detected = checked_next > 0 && detected_at_next_anchor == checked_next;
    let no_false_positives = false_positive_before_tamper == 0;
    let result = if all_detected && no_false_positives {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!(
        "  100% detection once covered: {} | 0 false positives: {} => {result}",
        all_detected, no_false_positives
    );
}
