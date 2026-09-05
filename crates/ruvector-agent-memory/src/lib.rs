//! # ruvector-agent-memory
//!
//! Coherence-weighted agent memory compaction for ruvector.
//!
//! Agent memories decay in relevance over time.  This crate provides three
//! compaction policies that retain the most important entries when the memory
//! store exceeds a target capacity:
//!
//! | Policy | Signal | Novel? |
//! |--------|--------|--------|
//! | `LruPolicy` | Recency (`last_accessed_at`) | No — classical |
//! | `LfuPolicy` | Frequency (`access_count`) | No — classical |
//! | `CoherencePolicy` | Weighted score: recency + frequency + context coherence | **Yes** |
//!
//! The `CoherencePolicy` is the core research contribution: it scores each stored
//! memory vector against a *context window* — the embeddings of recent agent
//! queries — and preferentially retains memories that are semantically aligned
//! with the agent's current reasoning thread.
//!
//! ## References
//!
//! The `ledger` and `ops` modules add the TARL (Transaction-Aware Reliable
//! Ledgers, arXiv:2608.03699) transactional memory ledger (ADR-307, PIR WP4):
//! five executable memory operations over accepted/pending/rejected states,
//! with witness-record emission (ADR-134 schema) and proof-gated acceptance.
//!
//! The `observation` and `fusion` modules add the cross-source causal fusion
//! layer (ADR-320, PIR WP18): source-tagged [`AtomicObservation`]s fuse into a
//! [`CausalEpisodicGraph`] with provenance preserved back to each atomic
//! source. Informed by MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`) and
//! explicitly distinct from the unrelated `memfuse/memfuse` OSS project. It
//! reuses this crate's WP4 ledger for governed admission and `rvf-types`'
//! SHA-256/Ed25519 for content addressing and per-observation signatures — no
//! new hash or signature scheme is introduced.
//!
//! The `arbitration` module adds correlation-aware memory arbitration
//! (ADR-330, PIR WP27), informed by CAMA (arXiv:2608.19701): retrieved
//! memories are clustered into independent evidence lineages by causal
//! ancestry, and effective confidence is scored per lineage
//! (downgrade-only relative to a naive per-memory vote), so N memories
//! repeating one origin count as one observation, not N.
//!
//! - Park et al. 2023, "Generative Agents" (arXiv:2304.03442)
//! - Zhong et al. 2023, "MemoryBank" (arXiv:2305.10250)
//! - Xu 2026, "Self-Aware Vector Embeddings for RAG" (arXiv:2604.20598)
//! - Karhade 2026, "Not All Memories Age the Same" (arXiv:2604.26970)
//! - Survey 2026, "From Storage to Experience" (arXiv:2605.06716)

pub mod arbitration;
pub mod compaction;
pub mod diagnostic;
pub mod fusion;
#[cfg(feature = "mincut-forget")]
pub mod graph_forget;
pub mod ledger;
pub mod memory;
pub mod observation;
pub mod ops;
pub mod scoring;
pub mod witnessed_compaction;

pub use arbitration::{
    arbitrate, ArbitrationConfig, ArbitrationError, ArbitrationOutcome, ArbitrationVerdict,
    EvidenceLineage, ReliabilityModel,
};
pub use compaction::{
    weighted_importance, CoherencePolicy, CoherenceWeights, CompactionPolicy, LfuPolicy, LruPolicy,
};
pub use diagnostic::{
    apply_gated_promotion, diagnostic_coverage, evaluate_promotion, gate_and_apply,
    localized_stage, BlockReason, DiagnosticError, DiagnosticTrace, GatedPromotion, MemoryPolicy,
    MemoryStage, PairedEvaluation, PairedOutcome, PolicySet, PromotionDecision,
    ProtectedSliceResult, RetrievalStrategy, StageSignal, LOCALIZATION_THRESHOLD,
};
pub use fusion::{CausalEpisodicGraph, ClusterId, FusedCluster, FusionError, NodeRef};
#[cfg(feature = "mincut-forget")]
pub use graph_forget::{ForgetMode, MincutGatedForgetting};
#[cfg(feature = "proof-gate")]
pub use ledger::WriteGateAdapter;
pub use ledger::{replay_history, AlwaysAdmitGate, LedgerEntry, ProofGate, TransactionalLedger};
pub use memory::{MemoryEntry, MemoryStore, SearchResult};
pub use observation::{
    AtomicObservation, ObservationError, ObservationId, ObservationSource, SourceKind, Tenant,
};
pub use ops::{
    AcceptanceReceipt, EvidenceGrade, LedgerError, LedgerState, LedgerWitnessRecord, MemoryOp,
    MemoryWitnessLog, NoopWitnessSink, TransitionKind, TransitionRecord, WitnessSink,
};
pub use scoring::{coherence_score, cosine_sim, normalize};
pub use witnessed_compaction::{compact_witnessed, EvictionWitnessChain};

/// Compact `store` in-place using `policy`, retaining `target_size` entries.
///
/// `context_window` is a slice of recent query embeddings used by
/// `CoherencePolicy` to score semantic alignment.  Pass an empty slice when
/// context is unavailable; `LruPolicy` and `LfuPolicy` ignore it.
///
/// # Panics
/// Panics if `target_size > store.len()`.
pub fn compact(
    store: &mut MemoryStore,
    policy: &dyn CompactionPolicy,
    target_size: usize,
    context_window: &[Vec<f32>],
) {
    assert!(
        target_size <= store.len(),
        "target_size ({}) must be ≤ store.len() ({})",
        target_size,
        store.len()
    );
    let entries = store.entries();
    let survivor_indices = policy.select_survivors(entries, target_size, context_window);
    let mut survivors: Vec<MemoryEntry> = survivor_indices
        .into_iter()
        .map(|i| entries[i].clone())
        .collect();
    survivors.sort_unstable_by_key(|e| e.id);
    store.replace_entries(survivors);
}

/// Recall@K: fraction of true top-K neighbors found in candidate set.
///
/// `truth` and `candidates` are sets of entry ids.  K = `truth.len()`.
pub fn recall_at_k(truth: &[u64], candidates: &[u64]) -> f32 {
    let k = truth.len();
    if k == 0 {
        return 1.0;
    }
    let hits = truth.iter().filter(|id| candidates.contains(id)).count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_reduces_store_size() {
        let mut store = MemoryStore::new(4);
        for _ in 0..20 {
            store.insert(vec![1.0, 0.0, 0.0, 0.0]);
        }
        compact(&mut store, &LruPolicy, 10, &[]);
        assert_eq!(store.len(), 10);
    }

    #[test]
    fn recall_perfect() {
        let truth = vec![0, 1, 2, 3, 4];
        assert!((recall_at_k(&truth, &truth) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        let truth = vec![0, 1, 2];
        let cands = vec![5, 6, 7];
        assert!(recall_at_k(&truth, &cands) < 1e-6);
    }
}
