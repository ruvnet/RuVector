#![allow(clippy::needless_range_loop)]
//! ADR-154 §17 item 10 — the "labels" axis of the three-axis AC-2
//! remediation framing.
//!
//! Discovery #10 (commit 15/16): stimulus-protocol labels can't be
//! recovered from SDPA embeddings on this substrate — the saturated
//! regime dominates, protocol identity dissipates inside ~150 ms.
//!
//! Discovery #12 (commit 19): raw rate-histogram encoder ties SDPA
//! at sub-random precision@5 on the same 8-protocol labeled corpus.
//! **Encoder axis is ruled out.**
//!
//! This test runs the remaining "labels" axis: drop stimulus-protocol
//! identity as the ground-truth label and use instead the raster
//! signature the encoder actually tracks — `(dominant_class_idx,
//! spike_count_bucket)`. If the SDPA embedding is "protocol-blind but
//! raster-sensitive", this re-labeling should show precision@5 well
//! above random and above the stimulus-protocol score. If it doesn't,
//! the substrate-axis is the only remaining candidate for AC-2 work.
//!
//! Diagnostic-only: the test prints the measured precision for both
//! label schemes but does NOT hard-fail on the number. The ADR §14
//! risk register forbids relaxing SOTA thresholds; this is a new
//! measurement to be documented, not a gate.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, CurrentInjection, Engine, EngineConfig,
    Observer, Stimulus,
};

fn default_conn() -> Connectome {
    Connectome::generate(&ConnectomeConfig::default())
}

/// Run one stimulus through the connectome and return the indexed
/// SDPA embeddings alongside their raster-regime signatures.
///
/// Returns `(vectors, signatures)` where each signature is a
/// `(dominant_class_idx, spike_count, t_center_ms)` triple.
fn run_and_collect(
    conn: &Connectome,
    stim: &Stimulus,
    t_end_ms: f32,
) -> (Vec<Vec<f32>>, Vec<(u8, u32, f32)>) {
    let mut eng = Engine::new(conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    eng.run_with(stim, &mut obs, t_end_ms);
    let spikes = obs.spikes().to_vec();
    let an = Analysis::new(AnalysisConfig {
        motif_window_ms: 20.0,
        motif_bins: 10,
        index_capacity: 256,
        ..AnalysisConfig::default()
    });
    let (index, _hits) = an.retrieve_motifs(conn, &spikes, 5);
    let vectors: Vec<Vec<f32>> = index.vectors().to_vec();
    let signatures = index.window_signatures();
    (vectors, signatures)
}

/// Eight distinct stimulus protocols — same shape as the rate-encoder
/// comparison. Returned as `(protocol_id, Stimulus)` pairs.
fn make_8_protocols(conn: &Connectome) -> Vec<(u8, Stimulus)> {
    let sensory = conn.sensory_neurons().to_vec();
    let n = sensory.len();
    let range = |lo: usize, hi: usize| sensory[lo.min(n)..hi.min(n)].to_vec();

    let mut out: Vec<(u8, Stimulus)> = Vec::new();
    let specs: &[(usize, usize, f32, f32, u32)] = &[
        (0, n / 2, 15.0, 90.0, 20),
        (n / 2, n, 15.0, 90.0, 20),
        (0, n, 8.0, 90.0, 30),
        (0, n, 25.0, 90.0, 14),
        (0, n / 4, 15.0, 60.0, 20),
        (3 * n / 4, n, 15.0, 120.0, 20),
        (n / 4, 3 * n / 4, 12.0, 90.0, 25),
        (0, n, 15.0, 90.0, 20),
    ];
    for (i, (lo, hi, period, amp, pulses)) in specs.iter().copied().enumerate() {
        let pool = range(lo, hi);
        let mut s = Stimulus::empty();
        for k in 0..pulses {
            let t0 = 20.0 + k as f32 * period;
            for (pos, &target) in pool.iter().enumerate() {
                s.push(CurrentInjection {
                    t_ms: t0 + pos as f32 * 0.20,
                    target,
                    charge_pa: amp,
                });
            }
        }
        out.push((i as u8, s));
    }
    out
}

/// Bucket a spike count into one of 4 bins. Boundaries chosen so
/// typical fly-scale window counts (0..2000) are split roughly evenly
/// across the active regime.
fn bucket_count(n: u32) -> u8 {
    match n {
        0..=50 => 0,
        51..=200 => 1,
        201..=800 => 2,
        _ => 3,
    }
}

/// Compose a raster-regime label from (dominant_class, count_bucket).
/// 15 classes × 4 buckets = 60 possible labels; in practice ~8-15
/// are populated in a typical 8-protocol run.
fn raster_label(sig: (u8, u32, f32)) -> u16 {
    let (class, count, _t) = sig;
    let bucket = bucket_count(count) as u16;
    (class as u16) * 4 + bucket
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Precision@k on a labeled corpus, leave-one-out over queries.
fn precision_at_k(vectors: &[Vec<f32>], labels: &[u16], k: usize) -> f32 {
    let n = vectors.len();
    if n < 2 {
        return 0.0;
    }
    let k = k.min(n - 1);
    if k == 0 {
        return 0.0;
    }
    let mut total_hits = 0.0_f32;
    let mut total_queries = 0.0_f32;
    for qi in 0..n {
        let qv = &vectors[qi];
        let qlbl = labels[qi];
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j == qi {
                continue;
            }
            dists.push((j, l2_dist(qv, &vectors[j])));
        }
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let hits: usize = dists
            .iter()
            .take(k)
            .filter(|(j, _)| labels[*j] == qlbl)
            .count();
        total_hits += hits as f32 / k as f32;
        total_queries += 1.0;
    }
    if total_queries == 0.0 {
        0.0
    } else {
        total_hits / total_queries
    }
}

#[test]
fn ac_2_raster_regime_labels_vs_protocol_labels() {
    let conn = default_conn();
    let protocols = make_8_protocols(&conn);

    // Collect all indexed vectors + their metadata + stimulus-protocol id.
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut protocol_labels: Vec<u16> = Vec::new();
    let mut raster_signatures: Vec<(u8, u32, f32)> = Vec::new();
    for (pid, stim) in &protocols {
        let (v, sigs) = run_and_collect(&conn, stim, 140.0);
        assert_eq!(v.len(), sigs.len(), "vectors and signatures mismatched");
        for (vec, sig) in v.into_iter().zip(sigs.into_iter()) {
            vectors.push(vec);
            protocol_labels.push(*pid as u16);
            raster_signatures.push(sig);
        }
    }

    let corpus = vectors.len();
    assert!(corpus >= 40, "corpus too small to judge precision ({corpus})");

    // Build the raster-regime labels from signatures.
    let raster_labels: Vec<u16> = raster_signatures
        .iter()
        .copied()
        .map(raster_label)
        .collect();

    // Histogram both label schemes for diagnostic context.
    let mut proto_counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
    for &l in &protocol_labels {
        *proto_counts.entry(l).or_insert(0) += 1;
    }
    let mut raster_counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
    for &l in &raster_labels {
        *raster_counts.entry(l).or_insert(0) += 1;
    }
    let proto_distinct = proto_counts.len();
    let raster_distinct = raster_counts.len();
    let proto_max_share = proto_counts.values().max().copied().unwrap_or(0) as f32 / corpus as f32;
    let raster_max_share =
        raster_counts.values().max().copied().unwrap_or(0) as f32 / corpus as f32;

    // Compute precision@5 under both label schemes on the same corpus.
    let proto_precision = precision_at_k(&vectors, &protocol_labels, 5);
    let raster_precision = precision_at_k(&vectors, &raster_labels, 5);

    // Random-chance baseline under each scheme (assumes uniform class
    // prior, which is conservative given max_share details below).
    let proto_random = 1.0 / proto_distinct as f32;
    let raster_random = 1.0 / raster_distinct as f32;

    eprintln!(
        "ac-2-raster-regime:\n\
         ===== protocol-id labels =====\n\
           corpus={corpus}  distinct={proto_distinct}  max_share={proto_max_share:.2}\n\
           precision@5={proto_precision:.3}  random={proto_random:.3}  \
           above_random={:.3}\n\
         ===== raster-regime labels (dominant_class × spike_count_bucket) =====\n\
           corpus={corpus}  distinct={raster_distinct}  max_share={raster_max_share:.2}\n\
           precision@5={raster_precision:.3}  random={raster_random:.3}  \
           above_random={:.3}",
        proto_precision - proto_random,
        raster_precision - raster_random,
    );

    let delta = raster_precision - proto_precision;
    eprintln!("ac-2-raster-regime: raster - protocol = {delta:+.3}");
    // Verdict: whether raster-regime labels are "real" depends on
    // BOTH precision AND class balance. A raster_precision=1.0 when
    // max_share=0.92 is trivially-dominant-class, not signal.
    let is_trivial_dominance = raster_max_share > 0.70;
    eprintln!(
        "ac-2-raster-regime: verdict — {}",
        if is_trivial_dominance {
            "RASTER-REGIME COLLAPSES TO DOMINANT-CLASS MONOCULTURE — the substrate saturates into one (class, count-bucket) regime across all 8 protocols (max_share > 0.70). precision@5 ≈ 1.0 is trivial under such imbalance; not a real signal. Confirms the substrate-axis diagnosis: at synthetic N=1024 scale, re-labeling can't rescue AC-2 — only a heterogeneous substrate (real FlyWire v783) produces the label diversity the encoder needs to discriminate."
        } else if raster_precision >= 0.30 && raster_precision > proto_precision + 0.10 {
            "RASTER-REGIME LABELS ARE THE LEVER (encoder tracks raster structure; protocol identity is the wrong ground truth)"
        } else if raster_precision > proto_precision + 0.05 {
            "RASTER-REGIME modestly better; encoder has some raster sensitivity but substrate axis may still be needed"
        } else {
            "RASTER-REGIME ≈ PROTOCOL at this scale — neither label scheme recovers signal; substrate axis (FlyWire) is the remaining lever"
        }
    );

    // Diagnostic-only: the test publishes the measured precisions and
    // class balance for ADR §17 item 10's three-axis roll-up. It does
    // NOT gate on raster-regime precision, because the finding itself
    // (collapse or separation) is the content.
    assert!(corpus >= 40, "corpus too small to judge ({corpus})");
    assert!(proto_distinct >= 6, "protocol labels nearly trivial");
    // raster_distinct can legitimately be 1 or 2 on this substrate —
    // that *is* the finding. Don't hard-fail on it.
}
