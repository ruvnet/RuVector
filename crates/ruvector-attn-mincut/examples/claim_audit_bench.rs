//! Nightly research: audits this crate's README performance claims
//! ("15-40% KV-cache reduction", "10-20% lower energy", "<1% coherence
//! degradation") against real measurements, and tests whether a
//! "best-sink" min-cut search (candidate_B) closes any sparsity gap found
//! in the shipped fixed-sink implementation (candidate_A).
//!
//! Run with: `cargo run --release -p ruvector-attn-mincut --example claim_audit_bench`
//!
//! Hypotheses (fixed before running, see docs/research/nightly/2026-10-05-attn-mincut-best-sink-audit/README.md):
//!   H1: candidate_A's min-cut step removes >=5pp of edges beyond what the
//!       trivial eps-only elementwise threshold already removes, averaged
//!       over FULL_SEQ_LENS x LAMBDAS.
//!   H2: candidate_B (best-sink) removes >=3pp more mincut-specific edges
//!       than candidate_A, averaged over the matched BESTSINK_SEQ_LENS
//!       subset x LAMBDAS -- i.e. the fixed-sink choice (not the min-cut
//!       concept) is the primary driver of any H1 shortfall.
//!   Coherence gate (descriptive pass/fail, not a hypothesis): cosine
//!       similarity of candidate_A's output vs baseline attn_softmax must
//!       be >= 0.99 at lambda=0.5, seq_len=128 -- the README's literal
//!       "<1% degradation" claim.
//!   Latency: reported honestly with no acceptance gate. The README frames
//!       min-cut gating as cheaper than softmax; we report the measured
//!       ratio and call out a contradiction if candidate_A is *slower*.

use ruvector_attn_mincut::{
    attn_mincut, attn_mincut_best_sink, attn_softmax, compute_logits, eps_only_keep_mask,
};
use ruvector_coherence::{compare_attention_masks, quality_check};
use std::time::Instant;

const D: usize = 64;
const LAMBDA_TAU: usize = 2; // tau is unused by the gating math itself (see mincut.rs), kept for API parity
const EPS: f32 = 0.01;
const LAMBDAS: [f32; 3] = [0.3, 0.5, 0.7];
/// Grid for baseline / candidate_A / eps-only ablation: one Dinic call each, cheap up to 256.
const FULL_SEQ_LENS: [usize; 4] = [32, 64, 128, 256];
/// Grid for candidate_B: O(seq_len) Dinic calls each, kept to a subset of FULL_SEQ_LENS.
const BESTSINK_SEQ_LENS: [usize; 3] = [32, 64, 128];

/// Deterministic LCG, no external `rand` dependency (keeps this crate dependency-free).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5 // roughly uniform in [-0.5, 0.5]
    }
}

/// Synthetic Q/K/V with a mild locality bias (shared low-frequency position
/// feature between Q and K) plus deterministic pseudo-noise, so the attention
/// graph has real structure instead of being pure noise -- closer to how real
/// token embeddings behave, while staying fully reproducible without a corpus.
fn gen_qkv(seq_len: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let mut q = vec![0f32; seq_len * d];
    let mut k = vec![0f32; seq_len * d];
    let mut v = vec![0f32; seq_len * d];
    for i in 0..seq_len {
        for h in 0..d {
            let pos_feat = ((i as f32) * (h as f32 + 1.0) * 0.07).sin();
            q[i * d + h] = pos_feat * 0.5 + rng.next_f32() * 0.3;
            k[i * d + h] = pos_feat * 0.5 + rng.next_f32() * 0.3;
            v[i * d + h] = rng.next_f32();
        }
    }
    (q, k, v)
}

fn mincut_specific_pp(eps_mask: &[bool], variant_mask: &[bool]) -> f64 {
    let n = eps_mask.len() as f64;
    let eps_kept = eps_mask.iter().filter(|&&b| b).count() as f64;
    let variant_kept = variant_mask.iter().filter(|&&b| b).count() as f64;
    // variant_mask is a subset of eps_mask by construction (see mincut.rs); clamp defensively.
    ((eps_kept - variant_kept).max(0.0) / n) * 100.0
}

fn timed_ns<F: FnMut()>(reps: u32, mut f: F) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_nanos() as f64 / reps as f64
}

struct Row {
    seq_len: usize,
    lambda: f32,
    variant: &'static str,
    edges_kept: usize,
    edges_total: usize,
    mincut_pp: f64,
    cosine_sim: f64,
    latency_us: f64,
    latency_ratio_vs_baseline: f64,
}

fn print_row(r: &Row) {
    println!(
        "{:>4} | {:<4} | {:<12} | {:>6}/{:<6} | {:>6.2}pp | {:>7.4} | {:>10.2}us | {:>6.2}x",
        r.seq_len,
        r.lambda,
        r.variant,
        r.edges_kept,
        r.edges_total,
        r.mincut_pp,
        r.cosine_sim,
        r.latency_us,
        r.latency_ratio_vs_baseline
    );
}

fn main() {
    println!("=== ruvector-attn-mincut nightly claim audit ===");
    println!("d={D} eps={EPS} lambdas={LAMBDAS:?}");
    println!(
        "{:>4} | {:<4} | {:<12} | {:^15} | {:>8} | {:>7} | {:>12} | {:>7}",
        "seq", "lam", "variant", "edges kept/tot", "mincutPP", "cosine", "latency", "ratio"
    );

    let mut h1_samples: Vec<f64> = Vec::new();
    let mut h2_a_matched: Vec<f64> = Vec::new();
    let mut h2_b_matched: Vec<f64> = Vec::new();
    let mut coherence_gate_cosine: Option<f64> = None;
    let mut any_candidate_a_slower = false;

    for &seq_len in FULL_SEQ_LENS.iter() {
        let (q, k, v) = gen_qkv(seq_len, D, 0xC0FF_EE00 ^ seq_len as u64);
        let logits = compute_logits(&q, &k, D, seq_len);
        let eps_mask = eps_only_keep_mask(&logits, EPS);

        let reps: u32 = if seq_len <= 64 { 20 } else if seq_len <= 128 { 8 } else { 3 };
        let baseline_ns = timed_ns(reps, || {
            std::hint::black_box(attn_softmax(&q, &k, &v, D, seq_len));
        });
        let baseline_out = attn_softmax(&q, &k, &v, D, seq_len);

        for &lambda in LAMBDAS.iter() {
            let a_out = attn_mincut(&q, &k, &v, D, seq_len, lambda, LAMBDA_TAU, EPS);
            let a_ns = timed_ns(reps, || {
                std::hint::black_box(attn_mincut(&q, &k, &v, D, seq_len, lambda, LAMBDA_TAU, EPS));
            });
            let a_pp = mincut_specific_pp(&eps_mask, &a_out.gating.keep_mask);
            let a_quality = quality_check(&baseline_out, &a_out.output, 0.99);
            let _mask_cmp = compare_attention_masks(&eps_mask, &a_out.gating.keep_mask);

            if seq_len == 64 || seq_len == 128 || seq_len == 256 {
                h1_samples.push(a_pp);
            }
            if BESTSINK_SEQ_LENS.contains(&seq_len) {
                h2_a_matched.push(a_pp);
            }
            if seq_len == 128 && (lambda - 0.5).abs() < 1e-6 {
                coherence_gate_cosine = Some(a_quality.cosine_sim);
            }
            if a_ns > baseline_ns {
                any_candidate_a_slower = true;
            }

            print_row(&Row {
                seq_len,
                lambda,
                variant: "candidate_A",
                edges_kept: a_out.gating.edges_kept,
                edges_total: a_out.gating.edges_total,
                mincut_pp: a_pp,
                cosine_sim: a_quality.cosine_sim,
                latency_us: a_ns / 1000.0,
                latency_ratio_vs_baseline: a_ns / baseline_ns,
            });

            if BESTSINK_SEQ_LENS.contains(&seq_len) {
                // Best-sink is O(seq_len) Dinic calls: single-shot timing only (no repetition),
                // documented honestly rather than inflating the rep count for this expensive path.
                let b_start = Instant::now();
                let b_out =
                    attn_mincut_best_sink(&q, &k, &v, D, seq_len, lambda, LAMBDA_TAU, EPS);
                let b_ns = b_start.elapsed().as_nanos() as f64;
                let b_pp = mincut_specific_pp(&eps_mask, &b_out.gating.keep_mask);
                let b_quality = quality_check(&baseline_out, &b_out.output, 0.99);
                h2_b_matched.push(b_pp);

                print_row(&Row {
                    seq_len,
                    lambda,
                    variant: "candidate_B",
                    edges_kept: b_out.gating.edges_kept,
                    edges_total: b_out.gating.edges_total,
                    mincut_pp: b_pp,
                    cosine_sim: b_quality.cosine_sim,
                    latency_us: b_ns / 1000.0,
                    latency_ratio_vs_baseline: b_ns / baseline_ns,
                });
            }
        }
    }

    let h1_avg = h1_samples.iter().sum::<f64>() / h1_samples.len() as f64;
    let h2_a_avg = h2_a_matched.iter().sum::<f64>() / h2_a_matched.len() as f64;
    let h2_b_avg = h2_b_matched.iter().sum::<f64>() / h2_b_matched.len() as f64;
    let h2_delta = h2_b_avg - h2_a_avg;
    let coherence_cosine = coherence_gate_cosine.expect("seq_len=128, lambda=0.5 sample must exist");

    let h1_result = if h1_avg >= 5.0 { "ACCEPT" } else { "REJECT" };
    let h2_result = if h2_delta >= 3.0 { "ACCEPT" } else { "REJECT" };
    let coherence_result = if coherence_cosine >= 0.99 { "PASS" } else { "FAIL" };

    println!();
    println!("=== Hypothesis results (thresholds fixed before this run) ===");
    println!(
        "H1 (candidate_A mincut-specific pruning >= 5pp over seq_len in [64,128,256] x lambda {:?}): avg={:.3}pp -> {}",
        LAMBDAS, h1_avg, h1_result
    );
    println!(
        "H2 (candidate_B - candidate_A mincut-specific pruning >= 3pp over {:?} x {:?}): A_avg={:.3}pp B_avg={:.3}pp delta={:.3}pp -> {}",
        BESTSINK_SEQ_LENS, LAMBDAS, h2_a_avg, h2_b_avg, h2_delta, h2_result
    );
    println!(
        "Coherence gate (README '<1% degradation' claim, seq_len=128 lambda=0.5): cosine_sim={:.5} (threshold 0.99) -> {}",
        coherence_cosine, coherence_result
    );
    println!(
        "Latency: candidate_A slower than baseline attn_softmax on at least one grid point: {}",
        any_candidate_a_slower
    );
    println!(
        "NOTE: 'latency ratio' is wall-clock compute cost, NOT the README's 'energy per sample' claim -- \
         this crate does not instrument energy, and no KV-cache exists in this toy op; both README \
         rows are therefore descriptive claims this benchmark can only partially speak to (see report)."
    );

    let overall = if h1_result == "REJECT" || coherence_result == "FAIL" {
        "REJECT (core min-cut sparsity or coherence claim falsified)"
    } else {
        "ACCEPT"
    };
    println!();
    println!("OVERALL ACCEPTANCE: {overall}");

    root_cause_probe();
}

/// Supplementary diagnostic, run after the frozen H1/H2 grid above and
/// reported separately -- it does NOT feed into the ACCEPT/REJECT decision.
/// H1 found mincut-specific pruning was *exactly* 0.00pp at every seq_len
/// >= 32 in the official grid. This probe scans small seq_len at the default
/// lambda=0.5 plus a deliberately huge lambda to show where the cut gate
/// (`cut_cost <= lambda * mean_weight`) stops firing, to explain *why*:
/// cut_cost scales with the number of edges crossing the cut (which grows
/// with graph size/density), while `lambda * mean_weight` is a single-edge
/// weight scale that does not grow with seq_len -- a dimensional mismatch
/// that makes the gate fire on toy graphs (this crate's own doctest/unit
/// tests use seq_len 1-5) and essentially never fire at realistic context
/// lengths, independent of tuning lambda within its documented [0,1] range.
fn root_cause_probe() {
    println!();
    println!("=== Supplementary root-cause probe (diagnostic only, not part of H1/H2) ===");
    println!("seq_len | lambda | cut_cost<=threshold fired? | mincut-specific pp");
    for &seq_len in &[4usize, 8, 16, 32, 64] {
        let (q, k, _v) = gen_qkv(seq_len, D, 0xC0FF_EE00 ^ seq_len as u64);
        let logits = compute_logits(&q, &k, D, seq_len);
        let eps_mask = eps_only_keep_mask(&logits, EPS);
        for &lambda in &[0.5f32, 5.0, 50.0] {
            let gating = ruvector_attn_mincut::dynamic_min_cut(&logits, seq_len, lambda, LAMBDA_TAU, EPS);
            let pp = mincut_specific_pp(&eps_mask, &gating.keep_mask);
            println!(
                "{seq_len:>7} | {lambda:>6} | {:>26} | {pp:.2}pp",
                if pp > 0.0 { "yes" } else { "no (or no-op cut)" }
            );
        }
    }
}
