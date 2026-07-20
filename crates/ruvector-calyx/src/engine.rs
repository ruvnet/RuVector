//! The Calyx query engine — the four verbs wired together.
//!
//! `measure → count → differentiate → compose`:
//!
//! 1. **measure** — the query arrives already measured through the same frozen
//!    lenses as the store (slots in, never flattened).
//! 2. **count** — per-lens rankings ([`ConstellationStore::search_lens`]) and
//!    cross-lens agreement ([`crate::loom`]).
//! 3. **differentiate** — lens weights (tuned offline by [`crate::anneal`]) bias
//!    the fusion toward high-signal lenses.
//! 4. **compose** — [`crate::fusion`] fuses the rankings, [`crate::ledger`]
//!    resolves grounding, and [`crate::ward`] adjudicates a guarded, replayable
//!    decision recorded in the provenance ledger.

use std::collections::BTreeMap;

use crate::constellation::ConstellationStore;
use crate::fusion::{weighted_rrf, DEFAULT_RRF_K};
use crate::ledger::{grounded_path, Ledger};
use crate::loom::min_agreement;
use crate::scoring::{fnv1a64, hash_vec, mix_hash};
use crate::ward::{adjudicate, Candidate, GuardDecision, GuardProfile};

/// A query: one object measured through (a subset of) the registry's lenses.
#[derive(Debug, Clone, Default)]
pub struct Query {
    pub slots: BTreeMap<String, Vec<f32>>,
}

impl Query {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_slot(mut self, lens: impl Into<String>, vector: Vec<f32>) -> Self {
        self.slots.insert(lens.into(), vector);
        self
    }
}

/// The outcome of a guarded query, plus the chain hash that makes it replayable.
#[derive(Debug, Clone)]
pub struct CalyxResult {
    pub decision: GuardDecision,
    /// Fused candidate list (best-first), for inspection / RAG context.
    pub fused: Vec<(u64, f32)>,
    /// Chain hash of the ledger entry recording this query.
    pub ledger_hash: u64,
}

/// The association-native engine: a store, per-lens fusion weights, a guard
/// profile, and an append-only provenance ledger.
pub struct CalyxEngine {
    store: ConstellationStore,
    /// Fusion weight per lens, in registry order.
    weights: Vec<f32>,
    guard: GuardProfile,
    ledger: Ledger,
    rrf_k: f32,
    top_k: usize,
}

impl CalyxEngine {
    pub fn new(store: ConstellationStore) -> Self {
        let n = store.lenses().len();
        Self {
            store,
            weights: vec![1.0; n],
            guard: GuardProfile::default(),
            ledger: Ledger::new(),
            rrf_k: DEFAULT_RRF_K,
            top_k: 10,
        }
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), self.store.lenses().len(), "weights/lenses");
        self.weights = weights;
        self
    }

    pub fn with_guard(mut self, guard: GuardProfile) -> Self {
        self.guard = guard;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }

    pub fn store(&self) -> &ConstellationStore {
        &self.store
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Run one guarded, logged query.
    pub fn query(&mut self, q: &Query) -> CalyxResult {
        let lenses = self.store.lenses();

        // count: one ranked list per registry lens the query measured.
        let mut rankings: Vec<Vec<(u64, f32)>> = Vec::with_capacity(lenses.len());
        let mut weights: Vec<f32> = Vec::with_capacity(lenses.len());
        for (i, m) in lenses.iter().enumerate() {
            match q.slots.get(&m.name) {
                Some(qv) => {
                    rankings.push(self.store.search_lens(&m.name, qv, self.top_k));
                    weights.push(self.weights[i]);
                }
                None => {
                    rankings.push(Vec::new());
                    weights.push(0.0);
                }
            }
        }

        // compose: fuse, then evaluate the top candidate's grounding + agreement.
        let fused = weighted_rrf(&rankings, &weights, self.rrf_k);
        let candidate = fused.first().map(|&(id, score)| {
            let agreement = self
                .store
                .get(id)
                .map(|rec| min_agreement(&q.slots, rec))
                .unwrap_or(0.0);
            let grounding = self.store.get(id).and_then(|rec| {
                grounded_path(id, &rec.anchors, self.guard.min_grounding_weight)
            });
            Candidate {
                record_id: id,
                fusion_score: score,
                cross_lens_agreement: agreement,
                grounding,
            }
        });

        let decision = adjudicate(candidate, &self.guard);

        // record: hash input, lenses consulted, retrieval, output → ledger.
        let input_hash = hash_query(&q.slots);
        let lens_hash = lenses
            .iter()
            .fold(0u64, |h, m| mix_hash(h, m.fingerprint));
        let retrieval_hash = fused
            .iter()
            .fold(0u64, |h, (id, _)| mix_hash(h, *id));
        let output_hash = hash_decision(&decision);
        let ledger_hash = self
            .ledger
            .append(input_hash, lens_hash, retrieval_hash, output_hash);

        CalyxResult {
            decision,
            fused,
            ledger_hash,
        }
    }
}

fn hash_query(slots: &BTreeMap<String, Vec<f32>>) -> u64 {
    let mut h = 0u64;
    for (name, v) in slots {
        h = mix_hash(h, fnv1a64(name.as_bytes()));
        h = mix_hash(h, hash_vec(v));
    }
    h
}

fn hash_decision(d: &GuardDecision) -> u64 {
    match d {
        GuardDecision::Answer {
            record_id,
            fusion_score,
            ..
        } => mix_hash(*record_id, fnv1a64(&fusion_score.to_bits().to_le_bytes())),
        GuardDecision::Refuse(r) => fnv1a64(format!("refuse:{r:?}").as_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constellation::Constellation;
    use crate::ledger::{Anchor, AnchorKind};
    use crate::lens::{LensKind, LensManifest};

    fn registry() -> Vec<LensManifest> {
        vec![
            LensManifest::new("sem", "v1", LensKind::DenseSemantic, 2, 1.0, 1.0),
            LensManifest::new("str", "v1", LensKind::Structural, 2, 1.0, 1.0),
        ]
    }

    #[test]
    fn answers_grounded_and_logs_replayable() {
        let mut store = ConstellationStore::new(registry());
        store
            .insert(
                Constellation::new(1)
                    .with_slot("sem", vec![1.0, 0.0])
                    .with_slot("str", vec![1.0, 0.0])
                    .with_anchor(Anchor::new(AnchorKind::PassedTest, "t1", 0.9)),
            )
            .unwrap();
        store
            .insert(
                Constellation::new(2)
                    .with_slot("sem", vec![1.0, 0.0]) // semantic twin
                    .with_slot("str", vec![0.0, 1.0]), // structural dissenter, ungrounded
            )
            .unwrap();
        let mut eng = CalyxEngine::new(store);
        let q = Query::new()
            .with_slot("sem", vec![1.0, 0.0])
            .with_slot("str", vec![1.0, 0.0]);
        let r = eng.query(&q);
        assert_eq!(r.decision.answered_id(), Some(1));
        assert!(eng.ledger().verify());
        assert_eq!(eng.ledger().len(), 1);
    }
}
