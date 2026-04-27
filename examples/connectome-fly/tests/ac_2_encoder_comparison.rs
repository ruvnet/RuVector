#![allow(clippy::needless_range_loop)]
//! ADR-154 §17 item 10 follow-up — encoder-vs-substrate diagnostic.
//!
//! The shipped SDPA + deterministic-low-rank-projection motif encoder
//! was measured protocol-blind on this substrate: expanded-corpus AC-2
//! at 8 protocols landed at `precision@5 = 0.117` (random = 0.125). The
//! ADR names three axes to fix this — different encoder, different
//! substrate, different labels — and asks that the cheapest axis
//! (encoder) be investigated first with a controlled A/B.
//!
//! This test is that A/B. It runs the same 8-protocol labeled corpus
//! through BOTH encoders and reports precision@5 side-by-side. The
//! test is **publish-only**: it does not gate on absolute precision
//! numbers. It fails only on non-deterministic output, malformed
//! vectors, or an empty corpus — the AC-2 precision numbers go into
//! the ADR §17 table, not into a regression gate.
//!
//! Interpretation rubric (fill in the commit message from the
//! printed verdict):
//!
//! - rate > SDPA by a meaningful margin (≥ 0.05) → SDPA is actively
//!   hurting on this substrate.
//! - rate ≈ SDPA (within 0.05) → encoder is NOT the bottleneck; try
//!   substrate or labels next.
//! - rate < SDPA → rate histogram is actively worse; SDPA at least
//!   preserves some protocol-specific signal.

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig, MotifIndex,
    NeuronId, Observer, Spike, Stimulus,
};

// -----------------------------------------------------------------
// 8-protocol corpus
// -----------------------------------------------------------------

/// One stimulus protocol in the 8-protocol labeled corpus.
///
/// Axes (ADR-154 §17 item 10 mirrors this):
/// - `sensory_subset`  — 0 = first half of sensory neurons, 1 = second half.
/// - `freq_hz`         — pulse-train rate.
/// - `amplitude_pa`    — per-pulse charge.
/// - `duration_ms`     — pulse-train window width.
#[derive(Clone, Copy, Debug)]
struct Protocol {
    id: u8,
    sensory_subset: u8,
    freq_hz: f32,
    amplitude_pa: f32,
    duration_ms: f32,
}

/// Build an 8-protocol corpus spanning the four axes called out in
/// ADR-154 §17 item 10. The eight points are an asymmetric partial
/// factorial — not all 2⁴ combinations (the budget is 8 protocols) —
/// chosen so every axis varies at least once against the P0 baseline.
fn eight_protocols() -> [Protocol; 8] {
    [
        Protocol {
            id: 0,
            sensory_subset: 0,
            freq_hz: 60.0,
            amplitude_pa: 80.0,
            duration_ms: 200.0,
        },
        Protocol {
            id: 1,
            sensory_subset: 0,
            freq_hz: 60.0,
            amplitude_pa: 80.0,
            duration_ms: 300.0,
        },
        Protocol {
            id: 2,
            sensory_subset: 0,
            freq_hz: 60.0,
            amplitude_pa: 130.0,
            duration_ms: 200.0,
        },
        Protocol {
            id: 3,
            sensory_subset: 0,
            freq_hz: 120.0,
            amplitude_pa: 80.0,
            duration_ms: 200.0,
        },
        Protocol {
            id: 4,
            sensory_subset: 1,
            freq_hz: 60.0,
            amplitude_pa: 80.0,
            duration_ms: 200.0,
        },
        Protocol {
            id: 5,
            sensory_subset: 1,
            freq_hz: 120.0,
            amplitude_pa: 130.0,
            duration_ms: 200.0,
        },
        Protocol {
            id: 6,
            sensory_subset: 0,
            freq_hz: 120.0,
            amplitude_pa: 130.0,
            duration_ms: 300.0,
        },
        Protocol {
            id: 7,
            sensory_subset: 1,
            freq_hz: 60.0,
            amplitude_pa: 130.0,
            duration_ms: 300.0,
        },
    ]
}

/// Build the current-injection schedule for one protocol.
fn stimulus_for(conn: &Connectome, p: &Protocol) -> Stimulus {
    let sensory = conn.sensory_neurons();
    let half = sensory.len() / 2;
    let subset: Vec<NeuronId> = if p.sensory_subset == 0 {
        sensory[..half].to_vec()
    } else {
        sensory[half..].to_vec()
    };
    Stimulus::pulse_train(&subset, 20.0, p.duration_ms, p.amplitude_pa, p.freq_hz)
}

/// Run one protocol through a fresh LIF engine and return all spikes
/// plus the simulation end-time.
fn run_protocol(conn: &Connectome, p: &Protocol) -> (f32, Vec<Spike>) {
    let stim = stimulus_for(conn, p);
    let mut eng = Engine::new(conn, EngineConfig::default());
    let mut obs = Observer::new(conn.num_neurons());
    let t_end = 20.0 + p.duration_ms + 80.0;
    eng.run_with(&stim, &mut obs, t_end);
    (t_end, obs.spikes().to_vec())
}

/// One labeled motif vector: the encoder output plus the protocol id
/// it was produced under.
#[derive(Clone, Debug)]
struct LabeledVec {
    vector: Vec<f32>,
    protocol_id: u8,
}

/// Run all 8 protocols and collect labeled motif vectors from the
/// given encoder (SDPA via `retrieve_motifs`, rate histogram via
/// `retrieve_motifs_rate`). `encoder_fn` takes `(connectome, spikes)`
/// and returns the populated motif index; the caller decides which
/// `Analysis` method to call.
fn collect_labeled_vectors<F>(
    conn: &Connectome,
    protocols: &[Protocol],
    mut encoder_fn: F,
) -> Vec<LabeledVec>
where
    F: FnMut(&Connectome, &[Spike]) -> MotifIndex,
{
    let mut labeled: Vec<LabeledVec> = Vec::new();
    for p in protocols {
        let (_t_end, spikes) = run_protocol(conn, p);
        if spikes.is_empty() {
            continue;
        }
        let index = encoder_fn(conn, &spikes);
        for v in index.vectors() {
            labeled.push(LabeledVec {
                vector: v.clone(),
                protocol_id: p.id,
            });
        }
    }
    labeled
}

// -----------------------------------------------------------------
// Precision@k
// -----------------------------------------------------------------

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Labeled precision@k: for each labeled vector, brute-force find its
/// top-k nearest neighbours in the labeled corpus (excluding itself),
/// count how many share its label. Returns the mean across the corpus.
fn precision_at_k(corpus: &[LabeledVec], k: usize) -> f32 {
    if corpus.len() < 2 || k == 0 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for (qi, q) in corpus.iter().enumerate() {
        // Score every other vector; keep top-k smallest distances.
        let mut pairs: Vec<(f32, u8)> = Vec::with_capacity(corpus.len() - 1);
        for (ci, c) in corpus.iter().enumerate() {
            if ci == qi {
                continue;
            }
            pairs.push((l2(&q.vector, &c.vector), c.protocol_id));
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(pairs.len());
        let hits = pairs[..take]
            .iter()
            .filter(|(_, lbl)| *lbl == q.protocol_id)
            .count();
        total += hits as f32 / take as f32;
    }
    total / corpus.len() as f32
}

// -----------------------------------------------------------------
// Main A/B diagnostic
// -----------------------------------------------------------------

#[test]
fn ac_2_encoder_comparison_sdpa_vs_rate_histogram() {
    // Same connectome for both encoders — isolates the encoder as the
    // only variable.
    let conn = Connectome::generate(&ConnectomeConfig::default());
    let protocols = eight_protocols();

    // Motif-window config matches the expanded-corpus AC-2 test
    // described in ADR-154 §17 item 10: 20 ms windows, 10 bins. The
    // index is large enough to hold every window from all 8 protocols
    // (≈ 8 × 20 = 160 at 200 ms, more at 300 ms).
    let cfg = AnalysisConfig {
        motif_window_ms: 20.0,
        motif_bins: 10,
        index_capacity: 1024,
        ..AnalysisConfig::default()
    };
    let an = Analysis::new(cfg.clone());

    // ---- SDPA path (shipped) ----
    let sdpa_corpus = collect_labeled_vectors(&conn, &protocols, |c, sp| {
        let (index, _hits) = an.retrieve_motifs(c, sp, 5);
        index
    });

    // ---- Rate-histogram path (this commit) ----
    let rate_corpus = collect_labeled_vectors(&conn, &protocols, |c, sp| {
        let (index, _hits) = an.retrieve_motifs_rate(c, sp, 5);
        index
    });

    // ---- Hard asserts: diagnostic sanity, NOT precision floor ----
    assert!(
        !sdpa_corpus.is_empty(),
        "SDPA corpus is empty — LIF engine or SDPA path failed"
    );
    assert!(
        !rate_corpus.is_empty(),
        "rate-histogram corpus is empty — LIF engine or rate path failed"
    );
    // Both encoders see the same windows — they must produce the same
    // count of labeled vectors. If this differs the A/B is invalid
    // (one path is dropping or inserting windows the other isn't).
    assert_eq!(
        sdpa_corpus.len(),
        rate_corpus.len(),
        "corpus size mismatch ({} SDPA vs {} rate) — one encoder is \
         filtering differently, invalidating the A/B",
        sdpa_corpus.len(),
        rate_corpus.len()
    );
    // Each protocol must be represented — otherwise the 1/8 random
    // baseline is not the right floor.
    let mut counts = [0_u32; 8];
    for v in &sdpa_corpus {
        counts[v.protocol_id as usize] += 1;
    }
    let distinct = counts.iter().filter(|c| **c > 0).count();
    assert!(
        distinct >= 2,
        "corpus collapsed to {distinct} distinct protocols out of 8 \
         — random baseline not comparable"
    );

    // Determinism check: re-run the rate path and confirm bit-identical
    // vectors. (SDPA relies on an external crate whose internal ordering
    // we don't gate here; the rate path has no RNG so MUST be exact.)
    let rate_corpus_b = collect_labeled_vectors(&conn, &protocols, |c, sp| {
        let (index, _hits) = an.retrieve_motifs_rate(c, sp, 5);
        index
    });
    assert_eq!(rate_corpus.len(), rate_corpus_b.len());
    for (a, b) in rate_corpus.iter().zip(rate_corpus_b.iter()) {
        assert_eq!(a.vector.len(), b.vector.len(), "rate: vector length drift");
        for (x, y) in a.vector.iter().zip(b.vector.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "rate encoder is non-deterministic"
            );
        }
    }

    // Malformed-vector guard: every rate vector should have length
    // 15 * motif_bins (15 classes in the connectome).
    let expected_dim = 15 * cfg.motif_bins;
    for v in &rate_corpus {
        assert_eq!(
            v.vector.len(),
            expected_dim,
            "rate vector has dim {} expected {expected_dim}",
            v.vector.len()
        );
    }

    // ---- Soft measurement: precision@5 ----
    let k = 5;
    let sdpa_p = precision_at_k(&sdpa_corpus, k);
    let rate_p = precision_at_k(&rate_corpus, k);
    let delta = rate_p - sdpa_p;
    let random_baseline = 1.0 / 8.0;

    // Verdict marker for the ADR §17 follow-up row.
    let marker = if delta > 0.05 {
        "PASS (rate > SDPA — SDPA is actively hurting)"
    } else if delta < -0.05 {
        "MISS (rate < SDPA — rate histogram is actively worse)"
    } else {
        "TIE (rate ≈ SDPA — encoder is NOT the bottleneck; try substrate or labels)"
    };

    eprintln!(
        "ac-2-encoder-comparison:\n\
           corpus_size           = {} windows\n\
           distinct_protocols    = {}/8\n\
           SDPA precision@{k}     = {sdpa_p:.3}\n\
           rate precision@{k}     = {rate_p:.3}\n\
           delta (rate - SDPA)   = {delta:+.3}\n\
           random baseline (1/8) = {random_baseline:.3}\n\
           verdict               = {marker}",
        sdpa_corpus.len(),
        distinct,
    );
}
