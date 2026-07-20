//! # ruvector-calyx
//!
//! An **association-native memory layer** for ruvector. It is the practical,
//! Rust reference implementation of the architecture thesis in Chris Royse's
//! *Calyx* white paper (2026): one input should not collapse into one flattened
//! vector — it should become a **constellation**, the same object measured
//! through many frozen **lenses**, with each lens stored as a distinct typed
//! **slot**. Relationships are then *derived between* slots, grounded against
//! real-world anchors, and gated by a fail-closed guard.
//!
//! ## The four verbs
//!
//! | Verb | Module | What it does |
//! |------|--------|--------------|
//! | **measure** | [`lens`], [`constellation`] | Run an object through many frozen, content-addressed lenses; keep slots distinct (never flattened). |
//! | **count** | [`loom`] | Derive cross-lens relationships — agreement *and* the informative disagreements. |
//! | **differentiate** | [`assay`] | Estimate how much signal (in bits) each lens adds, per unit cost. |
//! | **compose** | [`fusion`], [`ledger`], [`ward`], [`engine`] | Fuse per-lens rankings (RRF), resolve grounding, adjudicate a guarded answer, and record a replayable provenance path. |
//!
//! [`anneal`] closes the loop: it reversibly tunes the fusion weights to
//! maximize accepted-answers-per-cost.
//!
//! ## Three trust principles (enforced, not aspirational)
//!
//! 1. **Grounding is mandatory** — [`ward::GuardProfile::require_grounding`].
//! 2. **No flattening** — slots live in [`constellation::Constellation::slots`];
//!    [`constellation::Constellation::flatten`] exists *only* to build the
//!    lossy single-embedding baseline this layer is designed to beat.
//! 3. **Fail closed** — unknown lens, shape mismatch, low agreement, or missing
//!    grounding yield structured refusals ([`constellation::StoreError`],
//!    [`ward::RefusalReason`]), never a silent wrong answer.
//!
//! ## Relation to ruvector
//!
//! Each per-lens ranking ([`constellation::ConstellationStore::search_lens`]) is
//! exact brute force here; in a deployment it is backed by ruvector's HNSW / IVF
//! indexes. The cross-lens graph ([`loom`]) maps onto ruvector's graph/min-cut
//! association edges. See `docs/adr/ADR-272` for the full mapping and the
//! benchmark acceptance targets exercised by the `calyx-bench` binary.
//!
//! ## References
//!
//! - Royse 2026, "Calyx: An Association-Native Database and Its Path to
//!   Planetary-Scale Grounded Intelligence" (ResearchGate preprint;
//!   <https://github.com/ChrisRoyse/Calyx>).
//! - Royse 2026, "The Calculus of Association: A Formula for Artificial General
//!   Intelligence" (ResearchGate preprint) — frozen embedders as designable
//!   measurement instruments, derived-data abundance, teleological constellations.
//! - Cormack, Clarke & Buettcher 2009, "Reciprocal Rank Fusion Outperforms
//!   Condorcet and Individual Rank Learning Methods" (SIGIR).

pub mod anneal;
pub mod assay;
pub mod calibrate;
pub mod conformal;
pub mod constellation;
pub mod corpus;
pub mod disagreement;
pub mod engine;
pub mod fusion;
pub mod ledger;
pub mod ledger_policy;
pub mod lens;
pub mod loom;
pub mod routing;
pub mod rng;
pub mod scoring;
pub mod ward;

pub use anneal::{anneal_weights, AnnealConfig, AnnealOutcome};
pub use assay::{mutual_information_bits, signal_density, LensSignal, Observation};
pub use calibrate::{
    abstention_quality, expected_calibration_error, raw_confidence, AbstentionQuality, Calibrator,
    ConfidenceSignals,
};
pub use routing::{route, RoutingConfig, RoutingMode, RoutingOutcome, Witness};
pub use conformal::{
    calibrate as conformal_calibrate, coverage, empirical_risk, selective_error, ConformalSelector,
};
pub use disagreement::{
    find_conflicts, intrinsic_disagreement, neighbor_set, query_disagreement,
};
pub use ledger_policy::{learn as learn_policy, Action, Episode, LearnedPolicy};
pub use corpus::{load as load_corpus, save as save_corpus, Corpus, RealQuery};
pub use constellation::{Constellation, ConstellationStore, StoreError};
pub use engine::{CalyxEngine, CalyxResult, Query};
pub use fusion::{rrf, weighted_rrf, DEFAULT_RRF_K};
pub use ledger::{grounded_path, Anchor, AnchorKind, GroundingPath, Ledger, LedgerEntry};
pub use lens::{LensKind, LensManifest};
pub use loom::{agreement, mean_agreement, min_agreement, top_dissenter};
pub use scoring::{cosine_sim, fnv1a64, hash_vec, normalize};
pub use ward::{adjudicate, Candidate, GuardDecision, GuardProfile, RefusalReason};

/// Recall@k: fraction of `truth` ids present in `retrieved`.
pub fn recall_at_k(truth: &[u64], retrieved: &[u64]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let hits = truth
        .iter()
        .filter(|t| retrieved.contains(t))
        .count();
    hits as f32 / truth.len() as f32
}
