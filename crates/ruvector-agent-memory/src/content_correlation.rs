//! Content-correlation-augmented arbitration (nightly research, 2026-11-01).
//!
//! [`crate::arbitration`] (ADR-330, CAMA-pattern) clusters a retrieved memory
//! set into independent evidence lineages using **declared** causal ancestry
//! only. Its own module docs are explicit about the gap this leaves:
//!
//! > "Correlation is detected from declared `causal_parents`, never from
//! > content. A memory that copies another memory's content while declaring
//! > no parent is, to this module, a fresh root ... Deployments that need
//! > adversarial robustness must pair this with content-level near-duplicate
//! > detection or authenticated citation at ingest."
//!
//! `tests/arbitration.rs`'s
//! `content_copying_yields_no_correlation_downgrade_under_default_reliability`
//! locks in the exact failure numbers: 20 agents re-asserting one claim's
//! content with no declared parent score naive `18.0` / arbitrated `16.2`
//! and mint **20** lineages — a correlation-blind result dressed up by the
//! reliability multiplier to merely *look* discounted.
//!
//! This module is that missing pairing. It is **purely additive**: it does
//! not modify [`crate::arbitration`], re-derives nothing about its
//! invariants, and calls [`crate::arbitration::arbitrate`] as an opaque first
//! pass. It then re-clusters (or discounts) the *declared* lineages using a
//! caller-supplied embedding per memory and a cosine-similarity
//! near-duplicate signal — the embedding is deliberately not sourced from
//! [`crate::observation::AtomicObservation`] (which carries no vector field);
//! bridging content to a vector is the integrator's job, exactly as
//! [`crate::scoring`] (`MemoryEntry`/`CoherencePolicy`) and the causal-fusion
//! layer are today two separate subsystems in this crate.
//!
//! # Two independent variants
//!
//! - [`CorrelationMode::HardMerge`]: any two lineages with a cross-pair
//!   cosine similarity at or above `similarity_threshold` are merged into one
//!   lineage (transitively, via union-find). Aggressive: collapses
//!   near-duplicate content into a single evidence source, same as declared
//!   shared ancestry already does.
//! - [`CorrelationMode::SoftDiscount`]: lineages are left unmerged (the
//!   lineage count — the real diagnostic per the arbitration module's own
//!   docs — is preserved), but a lineage with a detected near-duplicate
//!   elsewhere has its weight contribution multiplied by `1 - strength`.
//!   Conservative: softens the score without asserting "these are now one
//!   source".
//!
//! # Why this stays downgrade-only
//!
//! A merged lineage's weight is `max(constituent lineage weights)`, and each
//! original lineage weight is already `max_member(confidence × reliability ×
//! freshness)` — so the merged weight equals the max over the union of
//! members, never more. A soft-discounted lineage's weight is multiplied by
//! a factor in `[1 - strength, 1] ⊆ [0, 1]`. Both transforms can only reduce
//! `arbitrated_confidence` relative to the declared-lineage baseline, which
//! was itself already ≤ `naive_confidence`. [`arbitrate_with_content_correlation`]
//! re-checks this at runtime and fails loud (never clamps) if violated,
//! matching the posture of [`crate::arbitration`].
//!
//! # Known limitation (read before use)
//!
//! Near-duplicate detection here is **embedding-quality-bound**: it catches
//! verbatim or near-verbatim re-assertion (the astroturf shape the
//! arbitration module names), not paraphrase, translation, or semantically
//! equivalent claims phrased differently with a dissimilar embedding. It is
//! a second, independent signal layered on top of declared ancestry, not a
//! replacement for authenticated citation at ingest.
//!
//! # Complexity (read before scaling up)
//!
//! Both variants compute the full pairwise cross-lineage cosine-similarity
//! matrix: `O(L² · M̄²)` similarity evaluations where `L` is the number of
//! declared lineages and `M̄` the average members per lineage, which is
//! `O(N²)` in the total memory count `N` when lineages are maximally
//! fragmented (the parentless-astroturf case this module targets is exactly
//! that worst case: `L ≈ N`). See
//! `examples/content_correlation_bench.rs` for measured wall-clock at
//! `N` up to 1 000 and the acceptance threshold this was benchmarked against.
//! An ANN/LSH-accelerated near-duplicate prefilter (e.g. `ruvector-mincut`'s
//! k-NN graph, or an HNSW index over the embeddings) is the natural next
//! step past that ceiling and is explicitly out of scope here.

use std::collections::BTreeMap;

use crate::arbitration::{
    arbitrate, ArbitrationConfig, ArbitrationError, ArbitrationOutcome, ArbitrationVerdict,
    EvidenceLineage,
};
use crate::fusion::{CausalEpisodicGraph, NodeRef};
use crate::observation::ObservationId;
use crate::scoring::cosine_sim;

/// How a detected content near-duplicate affects arbitration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationMode {
    /// Merge near-duplicate lineages into one (transitively).
    HardMerge,
    /// Keep lineages separate; discount a lineage's weight by `strength`
    /// (must be finite in `[0, 1]`) when it has a near-duplicate elsewhere.
    SoftDiscount { strength: f32 },
}

/// Configuration for the content-correlation pass.
#[derive(Debug, Clone)]
pub struct ContentCorrelationConfig {
    /// Cosine-similarity threshold in `[-1, 1]` above which two memories are
    /// treated as content near-duplicates. Near 1.0 (e.g. `0.95`+) targets
    /// verbatim/near-verbatim re-assertion, not generic topical similarity —
    /// [`crate::scoring::coherence_score`] already exists for the latter and
    /// answers a different question (alignment with a context window, not
    /// pairwise duplication).
    pub similarity_threshold: f32,
    /// Which transform to apply once near-duplicates are found.
    pub mode: CorrelationMode,
}

impl ContentCorrelationConfig {
    fn validate(&self) -> Result<(), ContentCorrelationError> {
        if !self.similarity_threshold.is_finite()
            || !(-1.0..=1.0).contains(&self.similarity_threshold)
        {
            return Err(ContentCorrelationError::InvalidConfig(
                "similarity_threshold must be finite in [-1, 1]",
            ));
        }
        if let CorrelationMode::SoftDiscount { strength } = self.mode {
            if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
                return Err(ContentCorrelationError::InvalidConfig(
                    "SoftDiscount strength must be finite in [0, 1]",
                ));
            }
        }
        Ok(())
    }
}

/// Errors from [`arbitrate_with_content_correlation`]. Wraps
/// [`ArbitrationError`] from the underlying declared-ancestry pass rather
/// than duplicating its variants.
#[derive(Debug)]
pub enum ContentCorrelationError {
    /// The underlying declared-ancestry arbitration pass failed.
    Arbitration(ArbitrationError),
    /// A memory in the arbitrated verdict has no entry in the supplied
    /// embedding map. Refused rather than silently skipped (a memory with no
    /// embedding is, for this module's purposes, unknown risk, not "assume
    /// independent").
    MissingEmbedding(NodeRef),
    /// Configuration failed validation.
    InvalidConfig(&'static str),
    /// The downgrade-only invariant was violated after a content-correlation
    /// transform — a bug in this module, surfaced loudly rather than
    /// clamped, matching [`crate::arbitration`]'s posture.
    DowngradeInvariantViolated { before: f64, after: f64 },
}

impl From<ArbitrationError> for ContentCorrelationError {
    fn from(e: ArbitrationError) -> Self {
        ContentCorrelationError::Arbitration(e)
    }
}

impl std::fmt::Display for ContentCorrelationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentCorrelationError::Arbitration(e) => write!(f, "arbitration failed: {e}"),
            ContentCorrelationError::MissingEmbedding(node) => {
                write!(f, "no embedding supplied for memory {node:?}")
            }
            ContentCorrelationError::InvalidConfig(msg) => {
                write!(f, "invalid content-correlation config: {msg}")
            }
            ContentCorrelationError::DowngradeInvariantViolated { before, after } => write!(
                f,
                "downgrade-only invariant violated: {after} > {before} after content correlation"
            ),
        }
    }
}

impl std::error::Error for ContentCorrelationError {}

/// Arbitrate `memories` against declared ancestry (via
/// [`crate::arbitration::arbitrate`]), then apply a content-similarity
/// near-duplicate pass over the resulting lineages.
///
/// `embeddings` must carry one vector per memory appearing in the arbitrated
/// verdict's lineages (every memory in `memories`, after dedup); a missing
/// entry is a hard error (see [`ContentCorrelationError::MissingEmbedding`]).
pub fn arbitrate_with_content_correlation(
    graph: &CausalEpisodicGraph,
    memories: &[NodeRef],
    embeddings: &BTreeMap<NodeRef, Vec<f32>>,
    config: &ArbitrationConfig,
    content: &ContentCorrelationConfig,
) -> Result<ArbitrationOutcome, ContentCorrelationError> {
    content.validate()?;

    let outcome = arbitrate(graph, memories, config)?;
    let verdict = outcome.verdict().clone();

    for lineage in &verdict.lineages {
        for member in &lineage.members {
            if !embeddings.contains_key(member) {
                return Err(ContentCorrelationError::MissingEmbedding(*member));
            }
        }
    }

    let before = verdict.arbitrated_confidence;
    let merged = match content.mode {
        CorrelationMode::HardMerge => {
            merge_lineages_by_content(verdict, embeddings, content.similarity_threshold)
        }
        CorrelationMode::SoftDiscount { strength } => discount_lineages_by_content(
            verdict,
            embeddings,
            content.similarity_threshold,
            strength,
        ),
    };
    let after = merged.arbitrated_confidence;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(after <= before + 1e-9) {
        return Err(ContentCorrelationError::DowngradeInvariantViolated { before, after });
    }

    Ok(finalize(merged, config.sufficiency_threshold))
}

/// Re-derive the [`ArbitrationOutcome`] sufficiency split for a
/// (possibly re-clustered) verdict, matching
/// [`crate::arbitration::arbitrate`]'s tail-end logic exactly.
fn finalize(verdict: ArbitrationVerdict, sufficiency_threshold: usize) -> ArbitrationOutcome {
    if verdict.independent_lineages() >= sufficiency_threshold {
        ArbitrationOutcome::Sufficient(verdict)
    } else {
        let over_represented_roots: Vec<ObservationId> = verdict
            .lineages
            .iter()
            .filter(|l| l.members.len() > 1)
            .flat_map(|l| l.roots.iter().copied())
            .collect();
        ArbitrationOutcome::InsufficientIndependentEvidence {
            verdict,
            required: sufficiency_threshold,
            over_represented_roots,
        }
    }
}

/// The maximum cosine similarity between any member of `a` and any member of
/// `b`. `O(|a| · |b|)`; embeddings are assumed present (checked by the
/// caller before this runs).
fn max_cross_similarity(
    a: &EvidenceLineage,
    b: &EvidenceLineage,
    embeddings: &BTreeMap<NodeRef, Vec<f32>>,
) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for ma in &a.members {
        let va = &embeddings[ma];
        for mb in &b.members {
            let vb = &embeddings[mb];
            let s = cosine_sim(va, vb);
            if s > best {
                best = s;
            }
        }
    }
    best
}

/// Minimal union-find with path compression + union by size (independent
/// copy of [`crate::arbitration`]'s private one — kept local so this module
/// never needs to touch that file).
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

/// [`CorrelationMode::HardMerge`]: union any two lineages with a cross-pair
/// cosine similarity ≥ `threshold`, transitively, then rebuild lineages.
fn merge_lineages_by_content(
    verdict: ArbitrationVerdict,
    embeddings: &BTreeMap<NodeRef, Vec<f32>>,
    threshold: f32,
) -> ArbitrationVerdict {
    let n = verdict.lineages.len();
    if n <= 1 {
        return verdict;
    }
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if max_cross_similarity(&verdict.lineages[i], &verdict.lineages[j], embeddings)
                >= threshold
            {
                uf.union(i, j);
            }
        }
    }

    let mut by_component: BTreeMap<usize, EvidenceLineage> = BTreeMap::new();
    for (idx, lineage) in verdict.lineages.into_iter().enumerate() {
        let comp = uf.find(idx);
        let entry = by_component.entry(comp).or_insert_with(|| EvidenceLineage {
            roots: Vec::new(),
            members: Vec::new(),
            weight: 0.0,
        });
        entry.roots.extend(lineage.roots);
        entry.members.extend(lineage.members);
        entry.weight = entry.weight.max(lineage.weight);
    }
    let mut lineages: Vec<EvidenceLineage> = by_component.into_values().collect();
    for l in &mut lineages {
        l.roots.sort_unstable();
        l.roots.dedup();
        l.members.sort_unstable();
        l.members.dedup();
    }
    lineages.sort_by_key(|l| l.roots.first().copied());

    let arbitrated: f64 = lineages.iter().map(|l| l.weight).sum();
    ArbitrationVerdict {
        naive_confidence: verdict.naive_confidence,
        arbitrated_confidence: arbitrated,
        lineages,
    }
}

/// [`CorrelationMode::SoftDiscount`]: keep lineages as declared, but
/// multiply a lineage's weight by `1 - strength` if it has a near-duplicate
/// (similarity ≥ `threshold`) in any other lineage.
fn discount_lineages_by_content(
    verdict: ArbitrationVerdict,
    embeddings: &BTreeMap<NodeRef, Vec<f32>>,
    threshold: f32,
    strength: f32,
) -> ArbitrationVerdict {
    let n = verdict.lineages.len();
    let mut flagged = vec![false; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if max_cross_similarity(&verdict.lineages[i], &verdict.lineages[j], embeddings)
                >= threshold
            {
                flagged[i] = true;
                flagged[j] = true;
            }
        }
    }

    let mut lineages = verdict.lineages;
    for (l, is_flagged) in lineages.iter_mut().zip(flagged.iter()) {
        if *is_flagged {
            l.weight *= (1.0 - strength) as f64;
        }
    }
    let arbitrated: f64 = lineages.iter().map(|l| l.weight).sum();
    ArbitrationVerdict {
        naive_confidence: verdict.naive_confidence,
        arbitrated_confidence: arbitrated,
        lineages,
    }
}
