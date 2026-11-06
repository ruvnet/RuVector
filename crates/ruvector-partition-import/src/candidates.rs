//! `candidate_a` ("trust-the-signature") and `candidate_b` ("verify +
//! spot-check") partition-import strategies.
//!
//! Both candidates verify the Ed25519 certificate first; an invalid
//! signature always falls back to [`crate::baseline::global_min_cut`].
//! The difference is what happens when the signature *is* valid:
//!
//! - `candidate_a` accepts the supplied partition and cut value
//!   unconditionally — this is structurally unable to detect a
//!   signed-but-suboptimal partition, by design: a signature only proves
//!   who sent the message, never that the message's claim is true.
//! - `candidate_b` additionally samples a fraction of vertices and checks
//!   a *necessary* (not sufficient) local-optimality condition for a
//!   global minimum cut: moving any single sampled vertex to the other
//!   side of the partition must not strictly decrease the cut value. This
//!   is computable in `O(degree(v))` per sampled vertex. Failing the check
//!   on *any* sampled vertex rejects the partition and falls back to
//!   [`crate::baseline::global_min_cut`]. Passing the check on every
//!   sampled vertex is evidence of, but does not prove, global optimality
//!   — an adversary who perturbs only vertices outside the sample evades
//!   detection entirely. See the nightly research report for measured
//!   detection rates at several sampling fractions.

use ed25519_dalek::VerifyingKey;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::baseline::global_min_cut;
use crate::certificate::{verify_against, CutCertificate};
use crate::graph::{Partition, WeightedGraph};

/// How an import attempt was resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportOutcome {
    /// `candidate_a`: signature valid, partition accepted with no further
    /// check.
    AcceptedTrusted,
    /// `candidate_b`: signature valid and every sampled vertex passed the
    /// local-optimality spot-check.
    AcceptedSpotChecked,
    /// Signature verification failed (bad signature, or the certificate's
    /// digests do not match the supplied graph/partition); fell back to
    /// recomputing the global minimum cut from scratch.
    RejectedSignatureInvalid,
    /// Signature was valid but the local-optimality spot-check failed on
    /// at least one sampled vertex; fell back to recomputing.
    RejectedSpotCheckFailed,
}

/// A trivial, non-fatal error type reserved for future use (e.g. malformed
/// input validation at a real transport boundary). No current code path
/// returns it; import attempts always resolve to an [`ImportDecision`],
/// falling back to the baseline rather than failing outright, because a
/// real agent B must always end up with *some* partition to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportError {
    InvalidGraph,
}

/// The result of an import attempt.
#[derive(Clone, Debug)]
pub struct ImportDecision {
    pub outcome: ImportOutcome,
    pub cut_value: f64,
    pub partition: Partition,
    /// `true` if the global minimum cut had to be recomputed from scratch
    /// (the slow path), `false` if the supplied certificate/partition was
    /// accepted directly.
    pub recomputed: bool,
}

fn fallback(graph: &WeightedGraph, outcome: ImportOutcome) -> ImportDecision {
    let result = global_min_cut(graph);
    ImportDecision {
        outcome,
        cut_value: result.cut_value,
        partition: result.partition,
        recomputed: true,
    }
}

/// `candidate_a`: verify the signature; if valid, accept the supplied
/// partition and cut value with no recomputation. If invalid, fall back to
/// the baseline.
pub fn candidate_a(
    graph: &WeightedGraph,
    partition: &Partition,
    cert: &CutCertificate,
    verifying_key: &VerifyingKey,
) -> ImportDecision {
    if verify_against(verifying_key, graph, partition, cert) {
        ImportDecision {
            outcome: ImportOutcome::AcceptedTrusted,
            cut_value: cert.statement.cut_value,
            partition: partition.clone(),
            recomputed: false,
        }
    } else {
        fallback(graph, ImportOutcome::RejectedSignatureInvalid)
    }
}

/// `candidate_b`: verify the signature, then spot-check local optimality
/// on a sampled fraction `sample_fraction` (e.g. `0.1` for 10%) of
/// vertices, deterministically chosen from `sample_seed`. Falls back to
/// the baseline if either check fails.
pub fn candidate_b(
    graph: &WeightedGraph,
    partition: &Partition,
    cert: &CutCertificate,
    verifying_key: &VerifyingKey,
    sample_fraction: f64,
    sample_seed: u64,
) -> ImportDecision {
    if !verify_against(verifying_key, graph, partition, cert) {
        return fallback(graph, ImportOutcome::RejectedSignatureInvalid);
    }

    if local_optimality_spot_check(graph, partition, sample_fraction, sample_seed) {
        ImportDecision {
            outcome: ImportOutcome::AcceptedSpotChecked,
            cut_value: cert.statement.cut_value,
            partition: partition.clone(),
            recomputed: false,
        }
    } else {
        fallback(graph, ImportOutcome::RejectedSpotCheckFailed)
    }
}

/// Sample `round(sample_fraction * n)` (at least 1, at most `n`) distinct
/// vertices deterministically from `sample_seed`, and check the necessary
/// local-optimality condition for a global minimum cut at each: moving the
/// vertex to the other side of `partition` must not strictly decrease the
/// cut value. Returns `false` on the first sampled vertex that fails.
pub fn local_optimality_spot_check(
    graph: &WeightedGraph,
    partition: &Partition,
    sample_fraction: f64,
    sample_seed: u64,
) -> bool {
    let n = graph.n;
    if n == 0 {
        return true;
    }
    let k = ((sample_fraction * n as f64).round() as usize).clamp(1, n);

    let mut rng = StdRng::seed_from_u64(sample_seed);
    let indices: Vec<usize> = (0..n).collect();
    let sampled: Vec<usize> = indices.choose_multiple(&mut rng, k).copied().collect();

    // A side with exactly one member cannot have that member "moved" to the
    // other side as a valid competing cut: the result would leave one side
    // empty, which is not a two-sided partition at all (and trivially has
    // cut value 0). A global minimum cut over connected graphs quite often
    // *is* a singleton side (isolating the lowest-weighted-degree vertex),
    // so this is a real, expected case on sparse random graphs, not a
    // corner case to special-case away for convenience.
    let count_true = partition.side.iter().filter(|&&s| s).count();
    let count_false = n - count_true;

    let adj = graph.adjacency();
    for v in sampled {
        let side_v = partition.side[v];
        if (side_v && count_true == 1) || (!side_v && count_false == 1) {
            continue;
        }
        let mut weight_same = 0.0f64;
        let mut weight_other = 0.0f64;
        for &(nbr, w) in &adj[v] {
            if partition.side[nbr as usize] == side_v {
                weight_same += w;
            } else {
                weight_other += w;
            }
        }
        // Moving v to the other side changes the cut by
        // (weight_same - weight_other): previously-crossing edges
        // (weight_other) stop crossing, previously-internal edges
        // (weight_same) start crossing. A global minimum cut requires this
        // delta to be >= 0 for every vertex (otherwise moving v would
        // strictly decrease the cut, contradicting minimality).
        if weight_same < weight_other {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::Signer;
    use crate::graph::generate_graph;

    /// Two triangles joined by a single bridge edge: the correct min cut
    /// isolates the bridge; every vertex should pass the spot-check.
    fn two_triangles() -> WeightedGraph {
        WeightedGraph {
            n: 6,
            edges: vec![
                (0, 1, 5.0),
                (1, 2, 5.0),
                (0, 2, 5.0),
                (3, 4, 5.0),
                (4, 5, 5.0),
                (3, 5, 5.0),
                (2, 3, 1.0),
            ],
        }
    }

    #[test]
    fn spot_check_passes_on_correct_partition_every_vertex() {
        let g = two_triangles();
        let side = vec![false, false, false, true, true, true];
        let p = Partition::new(side);
        for v in 0..g.n {
            assert!(
                local_optimality_spot_check(&g, &p, 1.0, v as u64),
                "vertex {v} should pass on the correct partition"
            );
        }
    }

    #[test]
    fn spot_check_fails_on_misplaced_vertex() {
        let g = two_triangles();
        // Move vertex 0 to the wrong side: it has weight 10 to its
        // original triangle (same side) and 0 to the other side, so this
        // is a clearly bad move the full-sample check must catch.
        let side = vec![true, false, false, true, true, true];
        let p = Partition::new(side);
        assert!(!local_optimality_spot_check(&g, &p, 1.0, 1));
    }

    #[test]
    fn candidate_a_blindly_accepts_signed_suboptimal_partition() {
        let signer = Signer::from_seed([5u8; 32]);
        let g = two_triangles();
        let correct = Partition::new(vec![false, false, false, true, true, true]);
        let correct_cut = correct.cut_value(&g);

        // Deliberately wrong: isolates a single vertex instead of the
        // real bridge, a strictly worse cut.
        let wrong = Partition::new(vec![true, false, false, false, false, false]);
        let wrong_cut = wrong.cut_value(&g);
        assert!(
            wrong_cut > correct_cut,
            "the constructed adversarial partition must be worse"
        );

        let cert = signer.sign(&g, &wrong, wrong_cut);
        let decision = candidate_a(&g, &wrong, &cert, &signer.verifying_key);
        assert_eq!(decision.outcome, ImportOutcome::AcceptedTrusted);
        assert!((decision.cut_value - wrong_cut).abs() < 1e-9);
        assert!(!decision.recomputed);
    }

    #[test]
    fn candidate_b_catches_a_locally_non_optimal_wrong_partition_at_full_sample() {
        // Note: the singleton-isolation partition used in
        // `candidate_a_blindly_accepts_signed_suboptimal_partition` below
        // (isolating vertex 0 alone) is, perhaps surprisingly, *locally
        // optimal* under single-vertex moves (no single vertex's move
        // off the singleton side is checkable at all — moving the last
        // member of a side is excluded, since it would empty that side —
        // and every other vertex's move is a tie or a worsening move).
        // That is itself a real limitation this crate documents: local
        // optimality is necessary, not sufficient, for global optimality,
        // so it is the wrong fixture to prove the detector works at all.
        // This test instead uses a single-vertex flip of the *correct*
        // partition (vertex 0 moved to the wrong side), which the
        // necessary condition can and must catch.
        let signer = Signer::from_seed([5u8; 32]);
        let g = two_triangles();
        let wrong = Partition::new(vec![true, false, false, true, true, true]);
        let wrong_cut = wrong.cut_value(&g);
        let correct_cut = Partition::new(vec![false, false, false, true, true, true]).cut_value(&g);
        assert!(
            wrong_cut > correct_cut,
            "the flipped partition must be worse"
        );
        let cert = signer.sign(&g, &wrong, wrong_cut);

        let decision = candidate_b(&g, &wrong, &cert, &signer.verifying_key, 1.0, 0);
        assert_eq!(decision.outcome, ImportOutcome::RejectedSpotCheckFailed);
        assert!(decision.recomputed);
    }

    #[test]
    fn invalid_signature_always_falls_back_for_both_candidates() {
        let signer = Signer::from_seed([5u8; 32]);
        let impostor = Signer::from_seed([6u8; 32]);
        let g = two_triangles();
        let correct = Partition::new(vec![false, false, false, true, true, true]);
        let cut = correct.cut_value(&g);
        let cert = impostor.sign(&g, &correct, cut);

        let a = candidate_a(&g, &correct, &cert, &signer.verifying_key);
        assert_eq!(a.outcome, ImportOutcome::RejectedSignatureInvalid);
        assert!(a.recomputed);

        let b = candidate_b(&g, &correct, &cert, &signer.verifying_key, 0.5, 0);
        assert_eq!(b.outcome, ImportOutcome::RejectedSignatureInvalid);
        assert!(b.recomputed);
    }

    #[test]
    fn honest_optimal_partition_accepted_by_both_with_matching_cut() {
        let signer = Signer::from_seed([5u8; 32]);
        let g = generate_graph(60, 9, 6);
        let result = global_min_cut(&g);
        let cert = signer.sign(&g, &result.partition, result.cut_value);

        let a = candidate_a(&g, &result.partition, &cert, &signer.verifying_key);
        assert_eq!(a.outcome, ImportOutcome::AcceptedTrusted);
        assert!((a.cut_value - result.cut_value).abs() < 1e-9);

        let b = candidate_b(
            &g,
            &result.partition,
            &cert,
            &signer.verifying_key,
            0.25,
            42,
        );
        assert_eq!(b.outcome, ImportOutcome::AcceptedSpotChecked);
        assert!((b.cut_value - result.cut_value).abs() < 1e-9);
    }
}
