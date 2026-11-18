//! # RuVector Memory Admission
//!
//! Global-min-cut gated write-time cluster admission for streaming agent
//! memory.
//!
//! `ruvector-namespace-merge` (ADR-299) answered a *read-time* question:
//! given a query and a fixed set of namespaces, which namespaces should be
//! searched? It used S-T max-flow/min-cut, which needs a source and a sink —
//! natural for "relevant vs. irrelevant to this query".
//!
//! This crate answers a *write-time* question that has no natural source or
//! sink: given a stream of incoming agent-memory vectors and a growing set
//! of clusters, should the next vector merge into an existing cluster, or
//! does it belong to a new one? Framing this with a fixed threshold on
//! cosine-to-nearest-centroid (the obvious baseline) ignores the rest of the
//! cluster graph — it cannot tell "this point is a legitimate but distant
//! member of a naturally spread-out cluster" apart from "this point is only
//! weakly attached to everything, including its nearest centroid". The
//! global minimum cut of the (existing clusters + candidate point)
//! similarity graph answers exactly that: it finds the single weakest link
//! in the *whole* graph, and checks whether the candidate point sits on the
//! weak side of it.
//!
//! Four admission policies are implemented, all behind [`AdmissionPolicy`]:
//!
//! 1. [`policy::NearestCentroidThreshold`] — baseline: merge into the
//!    nearest centroid if cosine similarity clears a fixed threshold, else
//!    spawn a new cluster.
//! 2. [`policy::MincutGatedAdmission`] — ADR-344, accepted: build a small
//!    weighted graph over *all* existing centroids + the candidate point,
//!    run [`mincut::global_min_cut`], and gate on the *average*
//!    crossing-edge weight of the cut against a fixed coherence threshold.
//! 3. [`policy::AdaptiveMincutAdmission`] — ADR-344, candidate B: identical
//!    mechanism, but the coherence threshold is self-calibrating — a
//!    running mean/std (Welford) of observed cut weights sets the threshold
//!    online instead of requiring a hand-tuned constant.
//! 4. [`bounded_degree::BoundedDegreeAdmission`] — tonight's candidate
//!    (2026-11-18, answering ADR-344's own Next Research item 1): identical
//!    mechanism to policy 2, but the graph is built over only the `K`
//!    nearest existing centroids to the candidate point, not all of them —
//!    bounding Stoer-Wagner's O(C^3) cost to O(K^3), independent of the
//!    total cluster count `C`.

pub mod bounded_degree;
pub mod dataset;
pub mod mincut;
pub mod policy;

/// Cosine similarity for normalised (unit-length) vectors — dot product
/// suffices.
#[inline(always)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared L2 distance (no sqrt; monotone for ranking).
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
