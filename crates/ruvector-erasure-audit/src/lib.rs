//! # ruvector-erasure-audit
//!
//! **Question:** when a vector is deleted from an HNSW index, is it actually
//! *erased* — or can a query-only adversary still tell that it was once there?
//!
//! The 2026-06-18 nightly (`ruvector-hnsw-repair`, ADR-259) measured HNSW
//! deletion strategies on a *utility* axis: recall@10 and latency after 20% of
//! nodes are removed. It never asked the erasure question. This crate attacks
//! that gap.
//!
//! ## Threat model
//!
//! A black-box adversary holds a candidate vector `v` and has ordinary query
//! access to the index: it can submit a query and read back `(id, distance)`
//! pairs, with a caller-chosen `ef`. It cannot read neighbour lists, the stored
//! vectors, or any log. It asks: *was `v` ever in this index?*
//!
//! This is the "right to erasure" verification question (and, inverted, a
//! membership-inference attack): tombstoning a node hides it from result sets
//! but leaves its structural imprint — the edges it acquired, the neighbour
//! lists its insertion pruned — in the graph.
//!
//! ## Measurement
//!
//! Leakage is measured as a *paired distinguishing advantage*. For each trial
//! we build two indexes that differ in exactly one event:
//!
//! - **A** ("erased"): base corpus, then `v` is inserted and then deleted.
//! - **B** ("never present"): base corpus, then an unrelated decoy `w` is
//!   inserted and then deleted, at the same position in the insert sequence.
//!
//! Both are then subjected to the same post-deletion churn. The adversary
//! queries both with `q = v` and computes a scalar feature. Accuracy is the
//! fraction of pairs where the feature ranks A above B (ties count 0.5);
//! 0.5 means no leak, 1.0 means a perfect oracle. Feature *and direction* are
//! selected on a split-half and scored on the held-out half, so the reported
//! number is not a best-of-N fishing artefact.
//!
//! ## Modules
//!
//! - [`data`] — deterministic seeded clustered-vector generation (no `rand`).
//! - [`erasure`] — three erasure modes: tombstone, eager repair (prior art),
//!   and the new `LocalRebuild`.
//! - [`audit`] — black-box adversary features and the paired distinguisher.
//! - [`certificate`] — a chained, tamper-evident erasure-audit certificate.
//! - [`harness`] — index construction plus the paired leak and utility runs,
//!   shared verbatim by the headline benchmark and the replication binary.

pub mod audit;
pub mod certificate;
pub mod data;
pub mod erasure;
pub mod harness;

pub use audit::{observe, Features, PairedDistinguisher, ProbeConfig, FEATURE_NAMES};
pub use certificate::{CertificateChain, ErasureCertificate};
pub use data::{ClusteredSource, DatasetConfig, Rng};
pub use erasure::{erase, ErasureMode, ErasureStats};

/// Wilson score interval (95%) for a binomial proportion.
///
/// Returned as `(low, high)`. Used to report whether a measured distinguishing
/// accuracy is separated from 0.5 by more than sampling noise.
pub fn wilson95(successes: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n_f = n as f64;
    let z = 1.959_963_984_540_054_f64;
    let p = successes / n_f;
    let denom = 1.0 + z * z / n_f;
    let centre = p + z * z / (2.0 * n_f);
    let margin = z * ((p * (1.0 - p) / n_f) + (z * z / (4.0 * n_f * n_f))).sqrt();
    (
        ((centre - margin) / denom).max(0.0),
        ((centre + margin) / denom).min(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_brackets_point_estimate() {
        let (lo, hi) = wilson95(50.0, 100);
        assert!(lo < 0.5 && hi > 0.5, "got ({lo}, {hi})");
        // A coin-flip result at n=100 must not be distinguishable from 0.5.
        assert!(lo < 0.5 && 0.5 < hi);
    }

    #[test]
    fn wilson_interval_excludes_half_for_strong_signal() {
        let (lo, hi) = wilson95(90.0, 100);
        assert!(lo > 0.5, "lower bound {lo} should exclude 0.5");
        assert!(hi <= 1.0);
    }

    #[test]
    fn wilson_handles_zero_samples() {
        let (lo, hi) = wilson95(0.0, 0);
        assert_eq!((lo, hi), (0.0, 1.0));
    }
}
