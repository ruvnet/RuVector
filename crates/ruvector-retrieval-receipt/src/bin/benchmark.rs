//! Retrieval-receipt benchmark: measures the real cost and tamper-detection
//! reliability of attaching witness-chained provenance to ANN query results.
//!
//! Usage:
//!   cargo run --release -p ruvector-retrieval-receipt --bin benchmark
//!   cargo run --release -p ruvector-retrieval-receipt --bin benchmark -- 5000 128 10 200

use std::time::{Duration, Instant};

use ruvector_retrieval_receipt::{
    query_hash, synthetic_queries, verify_root, AnchorContext, AnchorPurpose, BatchAnchor, Issuer,
    ReceiptVariant, ResultItem, RetrievalIndex, RetrievalReceipt, SignedRoot,
};

const BENCHMARK_ISSUED_AT_UNIX_MS: u64 = 1_788_134_400_000;

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct VariantStats {
    variant: ReceiptVariant,
    gen_mean_ns: f64,
    gen_p95_ns: u64,
    verify_worst_mean_ns: f64,
    proof_bytes_worst: usize,
    total_receipt_bytes_mean: f64,
    tamper_trials: usize,
    tamper_detected: usize,
}

/// Deterministic xorshift for tamper-trial selection (which query, which
/// tamper kind), independent of the dataset/query streams so tamper trials
/// are reproducible without adding an external RNG dependency.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone, Copy)]
enum TamperKind {
    ScoreMutation,
    VectorIdSubstitution,
    RankSwap,
    WriteReceiptHashFlip,
}

fn apply_tamper(results: &mut [ResultItem], kind: TamperKind, rng: &mut Xorshift64) -> usize {
    let idx = rng.next_range(results.len());
    match kind {
        TamperKind::ScoreMutation => {
            results[idx].score += 0.5;
        }
        TamperKind::VectorIdSubstitution => {
            results[idx].vector_id = results[idx].vector_id.wrapping_add(999_999);
        }
        TamperKind::RankSwap => {
            let other = (idx + 1) % results.len();
            results.swap(idx, other);
        }
        TamperKind::WriteReceiptHashFlip => {
            results[idx].write_receipt.payload_hash[0] ^= 0xFF;
        }
    }
    idx
}

fn run_variant(
    variant: ReceiptVariant,
    index: &RetrievalIndex,
    queries: &[Vec<f32>],
    k: usize,
    index_root: [u8; 32],
    tamper_trials_per_kind: usize,
) -> VariantStats {
    let mut gen_latencies = Vec::with_capacity(queries.len());
    let mut verify_worst_latencies = Vec::with_capacity(queries.len());
    let mut total_bytes = Vec::with_capacity(queries.len());
    let mut proof_bytes_worst = 0usize;

    for query in queries {
        let results = index.search(query, k);
        let qh = query_hash(query);

        let t0 = Instant::now();
        let receipt = RetrievalReceipt::build(variant, qh, index_root, &results);
        gen_latencies.push(t0.elapsed());

        let worst_idx = results.len().saturating_sub(1);
        proof_bytes_worst = receipt.proof_bytes_for(worst_idx);
        total_bytes.push(receipt.total_bytes() as f64);

        if variant != ReceiptVariant::None {
            let t1 = Instant::now();
            let ok = receipt.verify_item(worst_idx, qh, index_root, &results[worst_idx]);
            verify_worst_latencies.push(t1.elapsed());
            assert!(ok, "honest receipt must verify");
        }
    }

    gen_latencies.sort_unstable();
    let gen_mean_ns = gen_latencies.iter().map(|d| d.as_nanos()).sum::<u128>() as f64
        / gen_latencies.len() as f64;
    let gen_p95_ns = percentile(&gen_latencies, 95.0).as_nanos() as u64;

    let verify_worst_mean_ns = if verify_worst_latencies.is_empty() {
        0.0
    } else {
        verify_worst_latencies
            .iter()
            .map(|d| d.as_nanos())
            .sum::<u128>() as f64
            / verify_worst_latencies.len() as f64
    };

    let total_receipt_bytes_mean = total_bytes.iter().sum::<f64>() / total_bytes.len() as f64;

    // Tamper-detection trials: for each tamper kind, corrupt a fresh honest
    // result set and confirm verify_full() rejects it. None/no-receipt is
    // skipped: there is nothing to verify, by design.
    let mut tamper_trials = 0usize;
    let mut tamper_detected = 0usize;
    if variant != ReceiptVariant::None {
        let kinds = [
            TamperKind::ScoreMutation,
            TamperKind::VectorIdSubstitution,
            TamperKind::RankSwap,
            TamperKind::WriteReceiptHashFlip,
        ];
        let mut rng = Xorshift64(0xF00D_CAFE_1234_5678);
        for kind in kinds {
            for trial in 0..tamper_trials_per_kind {
                let query = &queries[trial % queries.len()];
                let results = index.search(query, k);
                let qh = query_hash(query);
                let receipt = RetrievalReceipt::build(variant, qh, index_root, &results);
                let mut tampered = results.clone();
                apply_tamper(&mut tampered, kind, &mut rng);
                tamper_trials += 1;
                if !receipt.verify_full(qh, index_root, &tampered) {
                    tamper_detected += 1;
                }
            }
        }
    }

    VariantStats {
        variant,
        gen_mean_ns,
        gen_p95_ns,
        verify_worst_mean_ns,
        proof_bytes_worst,
        total_receipt_bytes_mean,
        tamper_trials,
        tamper_detected,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Signed anchoring benchmark — candidate_A (per-query signing, batch=1) and
// candidate_B (batched signing, batch>1) layered on top of MerkleReceipt.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SignTamperKind {
    /// A receipt root is corrupted before being checked against its own
    /// inclusion proof (a forged root citation).
    RootByte,
    /// One byte of the batch signature is flipped.
    SignatureByte,
    /// One byte of an inclusion-proof sibling hash is flipped.
    ProofSibling,
}

struct SigningStats {
    batch_size: usize,
    sign_amortized_ns: f64,
    /// Full cost of authenticating one query with no caching: a fresh
    /// signature verify plus the inclusion-proof check, every query. This
    /// is what a verifier pays if it does *not* trust batch membership
    /// across queries.
    verify_naive_mean_ns: f64,
    /// Cost of authenticating one query when the batch signature has
    /// already been verified once and the verifier trusts that result for
    /// the rest of the batch: inclusion-proof check only.
    verify_cached_mean_ns: f64,
    /// One-time signature-verify cost per batch (not multiplied by
    /// `batch_size`), reported separately so `verify_cached_mean_ns` is
    /// not read as "free".
    sig_verify_once_ns: f64,
    proof_bytes_worst: usize,
    tamper_trials: usize,
    tamper_detected: usize,
}

fn apply_sign_tamper(
    roots: &mut [[u8; 32]],
    signed: &mut SignedRoot,
    proofs: &mut [Vec<([u8; 32], bool)>],
    kind: SignTamperKind,
    rng: &mut Xorshift64,
) -> usize {
    let idx = rng.next_range(roots.len());
    match kind {
        SignTamperKind::RootByte => {
            roots[idx][rng.next_range(32)] ^= 0xFF;
        }
        SignTamperKind::SignatureByte => {
            signed.signature[rng.next_range(64)] ^= 0xFF;
        }
        SignTamperKind::ProofSibling => {
            if let Some((sibling, _)) = proofs[idx].first_mut() {
                sibling[0] ^= 0xFF;
            }
        }
    }
    idx
}

fn run_signing_batch(
    issuer: &Issuer,
    index: &RetrievalIndex,
    queries: &[Vec<f32>],
    k: usize,
    index_root: [u8; 32],
    batch_size: usize,
    tamper_trials_per_kind: usize,
) -> SigningStats {
    let context = AnchorContext::new(AnchorPurpose::Batch, index_root);
    let mut sign_latencies = Vec::new();
    let mut verify_naive_latencies = Vec::new();
    let mut verify_cached_latencies = Vec::new();
    let mut sig_verify_once_latencies = Vec::new();
    let mut proof_bytes_worst = 0usize;

    for chunk in queries.chunks(batch_size) {
        let roots: Vec<[u8; 32]> = chunk
            .iter()
            .map(|q| {
                let results = index.search(q, k);
                let qh = query_hash(q);
                let receipt =
                    RetrievalReceipt::build(ReceiptVariant::Merkle, qh, index_root, &results);
                receipt.root().expect("Merkle receipt always has a root")
            })
            .collect();

        let t0 = Instant::now();
        let anchor = BatchAnchor::build(&roots).expect("benchmark batches are nonempty");
        let signed = issuer.sign_root(context, anchor.root(), BENCHMARK_ISSUED_AT_UNIX_MS);
        sign_latencies.push(t0.elapsed());

        let worst = roots.len() - 1;
        // SignedRoot is 170 canonical bytes including the root. Batch proof
        // bytes include that same 32-byte root, so add the other 138 bytes.
        proof_bytes_worst = proof_bytes_worst.max(
            138 + anchor
                .proof_bytes_for(worst)
                .expect("worst index is in bounds"),
        );

        // naive: every query pays a full signature verify + inclusion check
        for (i, root) in roots.iter().enumerate() {
            let proof = anchor.proof_for(i).expect("enumerated index is in bounds");
            let t1 = Instant::now();
            let verified = verify_root(&issuer.verifying_key, context, &signed);
            let incl_ok = verified
                .as_ref()
                .is_some_and(|trusted| BatchAnchor::verify_inclusion(*root, &proof, trusted));
            verify_naive_latencies.push(t1.elapsed());
            assert!(incl_ok, "honest batch member must verify (naive)");
        }

        // cached: signature verified once, reused for every query in the batch
        let t2 = Instant::now();
        let verified = verify_root(&issuer.verifying_key, context, &signed);
        sig_verify_once_latencies.push(t2.elapsed());
        let verified = verified.expect("honest batch signature must verify");
        for (idx, root) in roots.iter().enumerate() {
            let proof = anchor
                .proof_for(idx)
                .expect("enumerated index is in bounds");
            let t3 = Instant::now();
            let incl_ok = BatchAnchor::verify_inclusion(*root, &proof, &verified);
            verify_cached_latencies.push(t3.elapsed());
            assert!(incl_ok, "honest batch member must verify (cached)");
        }
    }

    let mean_ns = |ds: &[Duration]| -> f64 {
        if ds.is_empty() {
            0.0
        } else {
            ds.iter().map(|d| d.as_nanos()).sum::<u128>() as f64 / ds.len() as f64
        }
    };

    let total_sign_ns: f64 = sign_latencies.iter().map(|d| d.as_nanos() as f64).sum();
    let total_queries = queries.len() as f64;
    let sign_amortized_ns = total_sign_ns / total_queries;

    // Tamper trials: rebuild a fresh honest batch per trial, corrupt one
    // dimension of it, confirm the combined (signature + inclusion) check
    // rejects it.
    let mut tamper_trials = 0usize;
    let mut tamper_detected = 0usize;
    let kinds = [
        SignTamperKind::RootByte,
        SignTamperKind::SignatureByte,
        SignTamperKind::ProofSibling,
    ];
    let mut rng = Xorshift64(0xABCD_EF01_2345_6789 ^ (batch_size as u64));
    let batch_len = batch_size.min(queries.len()).max(1);
    for kind in kinds {
        // A batch of one has no Merkle level (root == leaf directly), so
        // there is no inclusion-proof sibling to corrupt. Skip rather than
        // silently count a no-op tamper as "not detected".
        if batch_len == 1 && matches!(kind, SignTamperKind::ProofSibling) {
            continue;
        }
        for trial in 0..tamper_trials_per_kind {
            let start = (trial * batch_len) % queries.len().max(1);
            let chunk: Vec<Vec<f32>> = (0..batch_len)
                .map(|j| queries[(start + j) % queries.len()].clone())
                .collect();
            let mut roots: Vec<[u8; 32]> = chunk
                .iter()
                .map(|q| {
                    let results = index.search(q, k);
                    let qh = query_hash(q);
                    let receipt =
                        RetrievalReceipt::build(ReceiptVariant::Merkle, qh, index_root, &results);
                    receipt.root().expect("Merkle receipt always has a root")
                })
                .collect();
            let anchor = BatchAnchor::build(&roots).expect("benchmark batches are nonempty");
            let mut signed = issuer.sign_root(context, anchor.root(), BENCHMARK_ISSUED_AT_UNIX_MS);
            let mut proofs: Vec<Vec<([u8; 32], bool)>> = (0..roots.len())
                .map(|i| anchor.proof_for(i).expect("enumerated index is in bounds"))
                .collect();

            let idx = apply_sign_tamper(&mut roots, &mut signed, &mut proofs, kind, &mut rng);
            tamper_trials += 1;

            let verified = verify_root(&issuer.verifying_key, context, &signed);
            let accepted = verified.as_ref().is_some_and(|trusted| {
                BatchAnchor::verify_inclusion(roots[idx], &proofs[idx], trusted)
            });
            if !accepted {
                tamper_detected += 1;
            }
        }
    }

    SigningStats {
        batch_size,
        sign_amortized_ns,
        verify_naive_mean_ns: mean_ns(&verify_naive_latencies),
        verify_cached_mean_ns: mean_ns(&verify_cached_latencies),
        sig_verify_once_ns: mean_ns(&sig_verify_once_latencies),
        proof_bytes_worst,
        tamper_trials,
        tamper_detected,
    }
}

fn variant_name(v: ReceiptVariant) -> &'static str {
    match v {
        ReceiptVariant::None => "NoReceipt",
        ReceiptVariant::PerResult => "PerResultReceipt",
        ReceiptVariant::Merkle => "MerkleReceipt",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let num_queries: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let tamper_trials_per_kind = 50usize; // 4 kinds x 50 = 200 trials per variant

    println!("=== ruvector-retrieval-receipt benchmark ===");
    println!(
        "n={n} dims={dims} k={k} queries={num_queries} tamper_trials_per_kind={tamper_trials_per_kind}"
    );
    println!(
        "hardware: {} logical CPUs (see `nproc`), rustc build profile: release-required for meaningful numbers",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );

    let t_ingest = Instant::now();
    let index = RetrievalIndex::ingest(n, dims, 0xC0FF_EE01_D00D);
    let ingest_elapsed = t_ingest.elapsed();
    assert!(index.verify_write_history(), "write history must verify");
    println!(
        "ingest: {n} vectors in {:.3} ms ({:.1} writes/ms), index_state_root non-zero: {}",
        ingest_elapsed.as_secs_f64() * 1000.0,
        n as f64 / (ingest_elapsed.as_secs_f64() * 1000.0),
        index.index_state_root() != [0u8; 32]
    );

    let queries = synthetic_queries(num_queries, dims, 0xA5A5_5A5A_1111);
    let index_root = index.index_state_root();

    // Baseline search-only latency (shared across all variants; receipts
    // are built *after* the same brute-force scan, so this isolates the
    // provenance layer's added cost from retrieval cost).
    let mut search_latencies = Vec::with_capacity(queries.len());
    for query in &queries {
        let t0 = Instant::now();
        let _ = index.search(query, k);
        search_latencies.push(t0.elapsed());
    }
    search_latencies.sort_unstable();
    let search_mean_ns = search_latencies.iter().map(|d| d.as_nanos()).sum::<u128>() as f64
        / search_latencies.len() as f64;
    println!(
        "\nbaseline brute-force search: mean={:.0}ns p95={:.0}ns over {} queries",
        search_mean_ns,
        percentile(&search_latencies, 95.0).as_nanos(),
        queries.len()
    );

    let variants = [
        ReceiptVariant::None,
        ReceiptVariant::PerResult,
        ReceiptVariant::Merkle,
    ];
    let stats: Vec<VariantStats> = variants
        .iter()
        .map(|&v| run_variant(v, &index, &queries, k, index_root, tamper_trials_per_kind))
        .collect();

    println!(
        "\n{:<18} {:>14} {:>14} {:>16} {:>14} {:>18} {:>16}",
        "variant",
        "gen_mean_ns",
        "gen_p95_ns",
        "verify_worst_ns",
        "proof_bytes",
        "total_bytes_mean",
        "tamper_detect"
    );
    for s in &stats {
        let tamper_str = if s.tamper_trials == 0 {
            "n/a".to_string()
        } else {
            format!("{}/{}", s.tamper_detected, s.tamper_trials)
        };
        println!(
            "{:<18} {:>14.0} {:>14} {:>16.0} {:>14} {:>18.1} {:>16}",
            variant_name(s.variant),
            s.gen_mean_ns,
            s.gen_p95_ns,
            s.verify_worst_mean_ns,
            s.proof_bytes_worst,
            s.total_receipt_bytes_mean,
            tamper_str
        );
    }

    // ── Acceptance evaluation (thresholds fixed before this run; see the
    // nightly research README for the formalized hypothesis) ──────────────
    let per_result = &stats[1];
    let merkle = &stats[2];

    let all_tamper_detected = stats
        .iter()
        .filter(|s| s.tamper_trials > 0)
        .all(|s| s.tamper_detected == s.tamper_trials);

    let merkle_proof_smaller = merkle.proof_bytes_worst < per_result.proof_bytes_worst;

    let overhead_threshold = 0.15; // receipt generation must add < 15% of baseline search latency
    let merkle_overhead = merkle.gen_mean_ns / search_mean_ns;
    let per_result_overhead = per_result.gen_mean_ns / search_mean_ns;
    let overhead_ok =
        merkle_overhead < overhead_threshold && per_result_overhead < overhead_threshold;

    println!("\n=== acceptance ===");
    println!("tamper detection 100% across all kinds: {all_tamper_detected}");
    println!(
        "merkle worst-case proof bytes ({}) < per-result worst-case proof bytes ({}): {merkle_proof_smaller}",
        merkle.proof_bytes_worst, per_result.proof_bytes_worst
    );
    println!(
        "generation overhead < {:.0}% of baseline search: merkle={:.1}% per_result={:.1}% -> {overhead_ok}",
        overhead_threshold * 100.0,
        merkle_overhead * 100.0,
        per_result_overhead * 100.0
    );

    let verdict = if all_tamper_detected && merkle_proof_smaller && overhead_ok {
        "ACCEPT"
    } else if all_tamper_detected && merkle_proof_smaller {
        "INCONCLUSIVE" // core provenance claims hold, overhead threshold missed
    } else {
        "REJECT"
    };
    println!("\nACCEPTANCE RESULT: {verdict}");

    // ── Signed anchoring (candidate_A = batch_size 1, candidate_B = batch_size > 1) ──
    println!("\n=== signed anchoring benchmark (Ed25519 over MerkleReceipt roots) ===");
    let issuer = Issuer::generate();
    let warmup_context = AnchorContext::new(AnchorPurpose::Batch, index_root);
    for byte in 0u8..128 {
        let signed = issuer.sign_root(warmup_context, [byte; 32], BENCHMARK_ISSUED_AT_UNIX_MS);
        assert!(verify_root(&issuer.verifying_key, warmup_context, &signed).is_some());
    }
    let batch_sizes = [1usize, 8, 32, 128];
    let sign_tamper_trials_per_kind = 50usize; // 3 kinds x 50 = 150 trials per batch size
    let sign_stats: Vec<SigningStats> = batch_sizes
        .iter()
        .map(|&b| {
            run_signing_batch(
                &issuer,
                &index,
                &queries,
                k,
                index_root,
                b,
                sign_tamper_trials_per_kind,
            )
        })
        .collect();

    println!(
        "\n{:<12} {:>18} {:>18} {:>18} {:>18} {:>14} {:>16}",
        "batch_size",
        "sign_amort_ns",
        "verify_naive_ns",
        "verify_cached_ns",
        "sig_verify_once_ns",
        "proof_bytes",
        "tamper_detect"
    );
    for s in &sign_stats {
        println!(
            "{:<12} {:>18.1} {:>18.0} {:>18.0} {:>18.0} {:>14} {:>16}",
            s.batch_size,
            s.sign_amortized_ns,
            s.verify_naive_mean_ns,
            s.verify_cached_mean_ns,
            s.sig_verify_once_ns,
            s.proof_bytes_worst,
            format!("{}/{}", s.tamper_detected, s.tamper_trials)
        );
    }

    let per_query = &sign_stats[0]; // batch_size = 1
    let largest_batch = sign_stats.last().unwrap(); // batch_size = 128

    let all_sign_tamper_detected = sign_stats
        .iter()
        .all(|s| s.tamper_detected == s.tamper_trials);

    // Hypothesis: batching amortizes signing cost roughly linearly with
    // batch size. Fixed threshold decided before this run (see nightly
    // research README): amortized signing cost at the largest batch size
    // must drop below 10% of the per-query (batch=1) cost.
    let amortization_threshold = 0.10;
    let amortization_ratio = largest_batch.sign_amortized_ns / per_query.sign_amortized_ns;
    let amortization_ok = amortization_ratio < amortization_threshold;

    // Attack-pass check: does batching help a verifier that does *not*
    // cache the signature check? It must not (naive verify cost is
    // dominated by the O(1) Ed25519 verify regardless of batch size) —
    // asserting the opposite would mean the benchmark is silently
    // rewarding an unrealistic verifier.
    let naive_verify_flat = {
        let min_naive = sign_stats
            .iter()
            .map(|s| s.verify_naive_mean_ns)
            .fold(f64::INFINITY, f64::min);
        let max_naive = sign_stats
            .iter()
            .map(|s| s.verify_naive_mean_ns)
            .fold(0.0, f64::max);
        // "flat" = largest batch doesn't cut naive-verify cost by more than half;
        // a strict drop there would indicate the benchmark accidentally
        // amortizes something it claims not to.
        max_naive / min_naive.max(1.0) < 2.0
    };

    println!("\n=== signed anchoring acceptance ===");
    println!("tamper detection 100% across all kinds and batch sizes: {all_sign_tamper_detected}");
    println!(
        "amortized signing cost drops below {:.0}% of per-query cost by batch={}: {:.1}% -> {amortization_ok}",
        amortization_threshold * 100.0,
        largest_batch.batch_size,
        amortization_ratio * 100.0
    );
    println!(
        "naive (uncached) per-query verify cost stays flat across batch sizes (batching does not help an uncaching verifier): {naive_verify_flat}"
    );

    let sign_verdict = if all_sign_tamper_detected && amortization_ok && naive_verify_flat {
        "ACCEPT"
    } else if all_sign_tamper_detected {
        "INCONCLUSIVE"
    } else {
        "REJECT"
    };
    println!("\nSIGNED ANCHORING ACCEPTANCE RESULT: {sign_verdict}");
}
