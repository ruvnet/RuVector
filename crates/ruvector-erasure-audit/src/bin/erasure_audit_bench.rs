//! Erasure-leak audit benchmark (headline run).
//!
//! ```text
//! cargo run --release -p ruvector-erasure-audit --bin erasure-audit-bench
//! ```
//!
//! Everything printed is measured at run time. The acceptance gates are
//! compiled in as `GATE_*` constants and were fixed before the first full run;
//! the binary prints PASS/FAIL against them itself rather than leaving the call
//! to a later narrative.

use ruvector_erasure_audit::audit::ProbeConfig;
use ruvector_erasure_audit::certificate::CertificateChain;
use ruvector_erasure_audit::harness::{
    run_leak_audit, run_utility, ExperimentConfig, LeakResult, UtilityResult,
};
use std::time::Instant;

// --- Pre-registered acceptance gates (fixed before the first full run) ------

/// G0 (precondition): the baseline must actually leak, otherwise there is
/// nothing to close and the run is INCONCLUSIVE.
const GATE_BASELINE_MIN_ACC: f64 = 0.60;
/// G1 (primary): the candidate's held-out distinguishing accuracy.
const GATE_CANDIDATE_MAX_ACC: f64 = 0.55;
/// G2 (utility): recall@10 loss versus EagerRepair, in percentage points.
const GATE_RECALL_LOSS_PP: f64 = 2.0;
/// G3 (cost): mean erasure latency versus EagerRepair.
const GATE_LATENCY_RATIO: f64 = 5.0;
/// G4 (exposure): bytes of the erased vector still resident.
const GATE_RETAINED_BYTES: usize = 0;

const CHURN_LEVELS: [usize; 3] = [0, 50, 200];
const DELETE_FRACTION: f64 = 0.20;
const WARMUP_QUERIES: usize = 200;
const N_QUERIES: usize = 300;

fn main() {
    let cfg = ExperimentConfig::default();
    println!("=== ruvector-erasure-audit — nightly 2026-09-22 ===");
    println!(
        "config: base_n={} dim={} trials={} churn_levels={CHURN_LEVELS:?} ef_rebuild={} \
         delete_fraction={DELETE_FRACTION}",
        cfg.base_n, cfg.dim, cfg.trials, cfg.ef_rebuild
    );
    println!(
        "gates (pre-registered): baseline_acc>={GATE_BASELINE_MIN_ACC} \
         candidate_acc<={GATE_CANDIDATE_MAX_ACC} recall_loss<={GATE_RECALL_LOSS_PP}pp \
         latency_ratio<={GATE_LATENCY_RATIO}x retained_bytes=={GATE_RETAINED_BYTES}"
    );

    let t = Instant::now();
    let base = cfg.build_base();
    println!(
        "\nbase index: {} nodes built in {:.2}s",
        base.live_count(),
        t.elapsed().as_secs_f64()
    );
    let probe = ProbeConfig::default();
    println!(
        "probe: k={} ef_lo={} ef_hi={} jitter_trials={} jitter_sigma={}",
        probe.k, probe.ef_lo, probe.ef_hi, probe.jitter_trials, probe.jitter_sigma
    );

    // --- Part 1: leak audit -------------------------------------------------
    println!("\n--- Part 1: erasure-leak audit (paired distinguishing accuracy) ---");
    println!(
        "{:<13} {:>6} {:>18} {:>4} {:>9} {:>9} {:>17} {:>6} {:>7}",
        "mode", "churn", "feature", "dir", "sel_acc", "hold_acc", "95% CI", "leak?", "secs"
    );
    let mut leaks: Vec<LeakResult> = Vec::new();
    for &churn in &CHURN_LEVELS {
        for mode in cfg.modes() {
            let r = run_leak_audit(&cfg, mode, churn, &base, &probe);
            println!(
                "{:<13} {:>6} {:>18} {:>4} {:>9.4} {:>9.4}  [{:.3}, {:.3}] {:>6} {:>7.1}",
                r.mode,
                r.churn,
                r.feature,
                r.direction,
                r.select_acc,
                r.holdout_acc,
                r.ci.0,
                r.ci.1,
                if r.leaks() { "YES" } else { "no" },
                r.elapsed_s
            );
            leaks.push(r);
        }
    }

    println!(
        "\nper-feature accuracy over all {} pairs (direction fixed on the selection half):",
        cfg.trials
    );
    print!("{:<13} {:>6}", "mode", "churn");
    for (name, _) in &leaks[0].per_feature {
        print!(" {name:>17}");
    }
    println!();
    for r in &leaks {
        print!("{:<13} {:>6}", r.mode, r.churn);
        for (_, acc) in &r.per_feature {
            print!(" {acc:>17.4}");
        }
        println!();
    }

    // --- Part 2: utility and cost -------------------------------------------
    println!("\n--- Part 2: utility and cost (20% deletion, identical victim ids) ---");
    let queries = cfg.queries(N_QUERIES);
    println!(
        "{:<13} {:>9} {:>9} {:>9} {:>12} {:>11} {:>11} {:>11} {:>9} {:>9}",
        "mode",
        "recall_b",
        "recall_a",
        "delta_pp",
        "del_us_mean",
        "del_us_p50",
        "del_us_p95",
        "srch_p95us",
        "ref_mean",
        "rtn_KiB"
    );
    let mut utils: Vec<UtilityResult> = Vec::new();
    for mode in cfg.modes() {
        let u = run_utility(&cfg, mode, &base, &queries, DELETE_FRACTION, WARMUP_QUERIES);
        println!(
            "{:<13} {:>9.4} {:>9.4} {:>9.2} {:>12.1} {:>11.1} {:>11.1} {:>11.1} {:>9.1} {:>9.1}",
            u.mode,
            u.recall_before,
            u.recall_after,
            (u.recall_after - u.recall_before) * 100.0,
            u.delete_us.mean,
            u.delete_us.p50,
            u.delete_us.p95,
            u.search_us.p95,
            u.referrers_mean,
            u.retained_bytes_total as f64 / 1024.0
        );
        utils.push(u);
    }
    if let Some(u) = utils.iter().find(|u| u.mode == "LocalRebuild") {
        println!(
            "LocalRebuild recomputed {:.1} neighbour lists per erasure (mean).",
            u.rebuilt_mean
        );
    }

    // --- Part 3: certificate chain ------------------------------------------
    println!("\n--- Part 3: erasure-audit certificate chain ---");
    let mut chain = CertificateChain::new();
    let n_del = ((cfg.base_n as f64) * DELETE_FRACTION) as usize;
    for (i, r) in leaks.iter().enumerate() {
        let u = utils.iter().find(|u| u.mode == r.mode).unwrap();
        chain.append(
            &format!("audit-{}-churn{}", r.mode, r.churn),
            match r.mode {
                "Tombstone" => "Tombstone",
                "EagerRepair" => "EagerRepair",
                _ => "LocalRebuild",
            },
            u.referrers_mean as u32,
            u.rebuilt_mean as u32,
            (u.retained_bytes_total / n_del.max(1)) as u32,
            (r.holdout_acc * 1000.0) as u32,
            1_000_000 + i as u64,
        );
    }
    println!(
        "appended {} records, head=0x{:016x}",
        chain.len(),
        chain.head()
    );
    println!("verify(clean) = {:?}", chain.verify());
    let detected = (0..chain.len())
        .filter(|&idx| {
            let mut c = chain.clone();
            let v = c.records()[idx].retained_vector_bytes;
            c.tamper_retained_bytes(idx, v.wrapping_add(1));
            c.verify().is_err()
        })
        .count();
    println!("tamper detection: {detected}/{} records", chain.len());

    // --- Acceptance ----------------------------------------------------------
    println!("\n--- Acceptance (thresholds fixed before the run) ---");
    let pick = |m: &str, c: usize| {
        leaks
            .iter()
            .find(|r| r.mode == m && r.churn == c)
            .unwrap()
            .clone()
    };
    let base_leak = pick("Tombstone", 0);
    let eager_leak = pick("EagerRepair", 0);
    let cand_leak = pick("LocalRebuild", 0);
    let eager_u = utils.iter().find(|u| u.mode == "EagerRepair").unwrap();
    let cand_u = utils.iter().find(|u| u.mode == "LocalRebuild").unwrap();

    let g0 = base_leak.holdout_acc >= GATE_BASELINE_MIN_ACC;
    let g1 = cand_leak.holdout_acc <= GATE_CANDIDATE_MAX_ACC;
    let recall_loss_pp = (eager_u.recall_after - cand_u.recall_after) * 100.0;
    let g2 = recall_loss_pp <= GATE_RECALL_LOSS_PP;
    let lat_ratio = cand_u.delete_us.mean / eager_u.delete_us.mean.max(1e-9);
    let g3 = lat_ratio <= GATE_LATENCY_RATIO;
    let g4 = cand_u.retained_bytes_total == GATE_RETAINED_BYTES;

    let row = |name: &str, thr: String, got: String, pass: bool| {
        println!(
            "{:<42} {:>18} {:>18}  {}",
            name,
            thr,
            got,
            if pass { "PASS" } else { "FAIL" }
        );
    };
    row(
        "G0 baseline leaks (Tombstone, churn=0)",
        format!(">= {GATE_BASELINE_MIN_ACC:.2}"),
        format!("{:.4}", base_leak.holdout_acc),
        g0,
    );
    row(
        "G1 candidate indistinguishable",
        format!("<= {GATE_CANDIDATE_MAX_ACC:.2}"),
        format!("{:.4}", cand_leak.holdout_acc),
        g1,
    );
    row(
        "G2 recall@10 loss vs EagerRepair",
        format!("<= {GATE_RECALL_LOSS_PP:.1} pp"),
        format!("{recall_loss_pp:.2} pp"),
        g2,
    );
    row(
        "G3 erase latency vs EagerRepair",
        format!("<= {GATE_LATENCY_RATIO:.1}x"),
        format!("{lat_ratio:.2}x"),
        g3,
    );
    row(
        "G4 retained vector bytes",
        format!("== {GATE_RETAINED_BYTES}"),
        format!("{}", cand_u.retained_bytes_total),
        g4,
    );

    println!(
        "\n[unregistered, exploratory] EagerRepair leak @churn=0: {:.4} CI [{:.3}, {:.3}] leak={}",
        eager_leak.holdout_acc,
        eager_leak.ci.0,
        eager_leak.ci.1,
        eager_leak.leaks()
    );
    println!(
        "  -> this was NOT the pre-registered hypothesis. Confirm on fresh seeds with:\n\
         \x20    cargo run --release -p ruvector-erasure-audit --bin erasure-replicate"
    );

    let verdict = if !g0 {
        "INCONCLUSIVE — the pre-registered baseline (Tombstone) did not leak measurably, \
         so the primary hypothesis could not be tested as written"
    } else if g1 && g2 && g3 && g4 {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!("\nVERDICT: {verdict}");
}
