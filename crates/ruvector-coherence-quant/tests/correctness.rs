use ruvector_attn_mincut::MinCutConfig;
use ruvector_coherence_quant::{allocator, coherence, dataset};
use ruvector_turboquant::Turbo4Codec;

#[test]
fn coherence_scores_are_bounded_and_deterministic_over_a_real_corpus() {
    let corpus = dataset::generate_corpus(6, 40, 32, 1);
    let cfg = MinCutConfig::default();
    let a =
        coherence::corpus_coherence(&corpus.vectors, &corpus.cluster_id, &corpus.centroids, &cfg);
    let b =
        coherence::corpus_coherence(&corpus.vectors, &corpus.cluster_id, &corpus.centroids, &cfg);
    assert_eq!(
        a, b,
        "coherence scoring must be deterministic for a fixed seed"
    );
    assert_eq!(a.len(), corpus.vectors.len());
    assert!(a.iter().all(|&s| (0.0..=1.0).contains(&s)));
}

#[test]
fn candidate_a_uses_fewer_bytes_than_uniform_baseline() {
    let corpus = dataset::generate_corpus(8, 50, 32, 2);
    let cfg = MinCutConfig::default();
    let coh =
        coherence::corpus_coherence(&corpus.vectors, &corpus.cluster_id, &corpus.centroids, &cfg);
    let codec = Turbo4Codec::new(32, 7).unwrap();

    let baseline = allocator::encode_corpus(
        &corpus.vectors,
        &coh,
        &corpus.cluster_id,
        corpus.centroids.len(),
        allocator::Variant::Baseline,
        &codec,
        0.3,
        0.05,
    );
    let candidate_a = allocator::encode_corpus(
        &corpus.vectors,
        &coh,
        &corpus.cluster_id,
        corpus.centroids.len(),
        allocator::Variant::CandidateA,
        &codec,
        0.3,
        0.05,
    );

    let baseline_bytes: usize = baseline.iter().map(|e| e.bytes()).sum();
    let candidate_bytes: usize = candidate_a.iter().map(|e| e.bytes()).sum();
    assert!(
        candidate_bytes < baseline_bytes,
        "coherence-gated allocation ({candidate_bytes} bytes) should use less memory than uniform Turbo4 ({baseline_bytes} bytes) when most vectors are coherent enough to drop to 1-bit"
    );

    // Roughly high_precision_frac of vectors should have kept Turbo4.
    let turbo4_count = candidate_a.iter().filter(|e| e.turbo4.is_some()).count();
    let frac = turbo4_count as f32 / candidate_a.len() as f32;
    assert!(
        (0.0..=0.6).contains(&frac),
        "unexpectedly large high-precision fraction: {frac}"
    );
}

#[test]
fn candidate_b_adds_exact_residuals_only_for_the_most_incoherent_vectors() {
    let corpus = dataset::generate_corpus(8, 50, 32, 3);
    let cfg = MinCutConfig::default();
    let coh =
        coherence::corpus_coherence(&corpus.vectors, &corpus.cluster_id, &corpus.centroids, &cfg);
    let codec = Turbo4Codec::new(32, 7).unwrap();

    let candidate_b = allocator::encode_corpus(
        &corpus.vectors,
        &coh,
        &corpus.cluster_id,
        corpus.centroids.len(),
        allocator::Variant::CandidateB,
        &codec,
        0.3,
        0.05,
    );

    let exact_count = candidate_b.iter().filter(|e| e.exact.is_some()).count();
    assert!(
        exact_count > 0,
        "expected at least one exact-cached outlier"
    );
    assert!(
        exact_count <= candidate_b.len() / 5,
        "residual cache should stay a small minority of the corpus, got {exact_count}/{}",
        candidate_b.len()
    );
    // Every exact residual must be attached to a Turbo4-precision entry.
    assert!(candidate_b
        .iter()
        .filter(|e| e.exact.is_some())
        .all(|e| e.turbo4.is_some()));
}
