//! Triples, dense-id vocabulary, filtered-ranking adjacency and frozen,
//! instance-disjoint train/valid/test splits.
//!
//! Nothing here logs a label (ADR-005): [`Vocab`] stores the strings the
//! bindings intern but its `Debug` prints counts only, and errors never carry
//! label text. Input limits are enforced as typed [`KgeError::Limit`] errors,
//! never silent truncation.

use crate::{EntityId, KgeError, RelationId, Result, Triple};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Max entities per table — ADR-005 (index-build budget).
pub const MAX_ENTITIES: usize = 1_000_000;
/// Max relations per table — ADR-005.
pub const MAX_RELATIONS: usize = 100_000;
/// Max triples per store. Not fixed by ADR-005; this is our own ceiling,
/// generous enough for YAGO3-10 (~1M) with headroom.
pub const MAX_TRIPLES: usize = 50_000_000;

mod rng;
pub(crate) use rng::Rng;

// ---------------------------------------------------------------------------
// Vocab — string labels ↔ dense ids. Debug redacts labels (ADR-005).
// ---------------------------------------------------------------------------

/// Maps interned string labels to dense `u32` ids for entities and relations.
/// The core never prints the strings; only the bindings own that.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(into = "VocabRepr", from = "VocabRepr")]
pub struct Vocab {
    entities: Vec<String>,
    relations: Vec<String>,
    entity_index: HashMap<String, EntityId>,
    relation_index: HashMap<String, RelationId>,
}

/// Serde surface for [`Vocab`]: the label lists only. The lookup indices are
/// rebuilt from them on load.
#[derive(Serialize, Deserialize)]
struct VocabRepr {
    entities: Vec<String>,
    relations: Vec<String>,
}

impl From<Vocab> for VocabRepr {
    fn from(v: Vocab) -> Self {
        Self {
            entities: v.entities,
            relations: v.relations,
        }
    }
}

impl From<VocabRepr> for Vocab {
    fn from(r: VocabRepr) -> Self {
        let mut v = Vocab {
            entities: r.entities,
            relations: r.relations,
            entity_index: HashMap::new(),
            relation_index: HashMap::new(),
        };
        v.rebuild_index();
        v
    }
}

impl std::fmt::Debug for Vocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print labels (ADR-005): counts only.
        f.debug_struct("Vocab")
            .field("entities", &self.entities.len())
            .field("relations", &self.relations.len())
            .finish()
    }
}

impl Vocab {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an entity label, returning its dense id (stable across calls).
    pub fn intern_entity(&mut self, label: &str) -> Result<EntityId> {
        if let Some(&id) = self.entity_index.get(label) {
            return Ok(id);
        }
        if self.entities.len() >= MAX_ENTITIES {
            return Err(KgeError::Limit("entities"));
        }
        let id = self.entities.len() as EntityId;
        self.entities.push(label.to_owned());
        self.entity_index.insert(label.to_owned(), id);
        Ok(id)
    }

    /// Intern a relation label, returning its dense id.
    pub fn intern_relation(&mut self, label: &str) -> Result<RelationId> {
        if let Some(&id) = self.relation_index.get(label) {
            return Ok(id);
        }
        if self.relations.len() >= MAX_RELATIONS {
            return Err(KgeError::Limit("relations"));
        }
        let id = self.relations.len() as RelationId;
        self.relations.push(label.to_owned());
        self.relation_index.insert(label.to_owned(), id);
        Ok(id)
    }

    pub fn entity_id(&self, label: &str) -> Option<EntityId> {
        self.entity_index.get(label).copied()
    }
    pub fn relation_id(&self, label: &str) -> Option<RelationId> {
        self.relation_index.get(label).copied()
    }
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Rebuild the skipped indices after `serde` deserialization.
    fn rebuild_index(&mut self) {
        self.entity_index = self
            .entities
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as EntityId))
            .collect();
        self.relation_index = self
            .relations
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as RelationId))
            .collect();
    }
}

// ---------------------------------------------------------------------------
// TripleStore — deduped triples + filtered-ranking adjacency.
// ---------------------------------------------------------------------------

/// A knowledge graph as dense-id triples, deduplicated, with the adjacency
/// needed for filtered ranking (all true objects per `(s, r)`, all true
/// subjects per `(r, o)`) computed over every triple in the store.
///
/// Only the triples and counts are serialized; the tuple-keyed adjacency
/// (which JSON cannot represent as a map) is rebuilt on load via [`StoreRepr`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "StoreRepr", from = "StoreRepr")]
pub struct TripleStore {
    triples: Vec<Triple>,
    num_entities: usize,
    num_relations: usize,
    /// `(s, r) -> {o}` for tail filtering.
    tails: BTreeMap<(EntityId, RelationId), BTreeSet<EntityId>>,
    /// `(r, o) -> {s}` for head filtering.
    heads: BTreeMap<(RelationId, EntityId), BTreeSet<EntityId>>,
}

/// Serde surface for [`TripleStore`]: triples and counts only. Adjacency is
/// recomputed deterministically from the triples on the way back in.
#[derive(Serialize, Deserialize)]
struct StoreRepr {
    triples: Vec<Triple>,
    num_entities: usize,
    num_relations: usize,
}

impl From<TripleStore> for StoreRepr {
    fn from(s: TripleStore) -> Self {
        Self {
            triples: s.triples,
            num_entities: s.num_entities,
            num_relations: s.num_relations,
        }
    }
}

impl From<StoreRepr> for TripleStore {
    fn from(r: StoreRepr) -> Self {
        let mut store = TripleStore {
            triples: r.triples,
            num_entities: r.num_entities,
            num_relations: r.num_relations,
            tails: BTreeMap::new(),
            heads: BTreeMap::new(),
        };
        store.rebuild_adjacency();
        store
    }
}

impl TripleStore {
    /// Build a store from raw triples. Deduplicates, derives entity/relation
    /// counts from the max id seen (unless overridden), enforces input limits,
    /// and builds filtered-ranking adjacency.
    pub fn new(triples: Vec<Triple>) -> Result<Self> {
        Self::with_counts(triples, None, None)
    }

    /// As [`TripleStore::new`], but with explicit entity/relation counts (e.g.
    /// from a [`Vocab`]) so isolated ids that never appear as a max are sized.
    pub fn with_counts(
        triples: Vec<Triple>,
        num_entities: Option<usize>,
        num_relations: Option<usize>,
    ) -> Result<Self> {
        if triples.len() > MAX_TRIPLES {
            return Err(KgeError::Limit("triples"));
        }
        // Dedup while preserving deterministic (sorted) order.
        let mut sorted: Vec<Triple> = triples;
        sorted.sort_by_key(|t| (t.s, t.r, t.o));
        sorted.dedup();

        let mut max_e: usize = 0;
        let mut max_r: usize = 0;
        for t in &sorted {
            max_e = max_e.max(t.s as usize).max(t.o as usize);
            max_r = max_r.max(t.r as usize);
        }
        let num_entities = num_entities.unwrap_or(if sorted.is_empty() { 0 } else { max_e + 1 });
        let num_relations = num_relations.unwrap_or(if sorted.is_empty() { 0 } else { max_r + 1 });

        if num_entities > MAX_ENTITIES {
            return Err(KgeError::Limit("entities"));
        }
        if num_relations > MAX_RELATIONS {
            return Err(KgeError::Limit("relations"));
        }
        // Counts must cover the ids actually present.
        if !sorted.is_empty() && (max_e + 1 > num_entities) {
            return Err(KgeError::Limit("entities"));
        }
        if !sorted.is_empty() && (max_r + 1 > num_relations) {
            return Err(KgeError::Limit("relations"));
        }

        let mut store = Self {
            triples: sorted,
            num_entities,
            num_relations,
            tails: BTreeMap::new(),
            heads: BTreeMap::new(),
        };
        store.rebuild_adjacency();
        Ok(store)
    }

    fn rebuild_adjacency(&mut self) {
        self.tails.clear();
        self.heads.clear();
        for t in &self.triples {
            self.tails.entry((t.s, t.r)).or_default().insert(t.o);
            self.heads.entry((t.r, t.o)).or_default().insert(t.s);
        }
    }

    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }
    pub fn len(&self) -> usize {
        self.triples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
    pub fn num_entities(&self) -> usize {
        self.num_entities
    }
    pub fn num_relations(&self) -> usize {
        self.num_relations
    }

    /// All true objects for `(s, r)` — the filter set for a tail query.
    pub fn true_tails(&self, s: EntityId, r: RelationId) -> Option<&BTreeSet<EntityId>> {
        self.tails.get(&(s, r))
    }
    /// All true subjects for `(r, o)` — the filter set for a head query.
    pub fn true_heads(&self, r: RelationId, o: EntityId) -> Option<&BTreeSet<EntityId>> {
        self.heads.get(&(r, o))
    }

    /// Frozen, instance-disjoint split into train/valid/test (see [`split4`]
    /// for the 4-way variant with a transfer holdout). Deterministic in
    /// `seed`: the same seed and ratios always produce the same partition, so a
    /// reloaded store re-splits identically. Each ratio must be finite and
    /// non-negative and their sum positive.
    ///
    /// [`split4`]: TripleStore::split4
    pub fn split(&self, seed: u64, ratios: [f64; 3]) -> Result<Split> {
        let sum = validate_ratios(&ratios)?;
        let idx = self.shuffled_indices(seed);
        let n = idx.len();
        let n_train = frac(ratios[0], sum, n).min(n);
        let n_valid = frac(ratios[1], sum, n).min(n - n_train);
        Ok(Split {
            train: self.pick(&idx[..n_train]),
            valid: self.pick(&idx[n_train..n_train + n_valid]),
            test: self.pick(&idx[n_train + n_valid..]),
        })
    }

    /// Frozen, instance-disjoint 4-way split with a **transfer holdout** whose
    /// relation mix differs from the others, for the ADR-004 transfer check.
    ///
    /// Rule: a seeded subset of *whole relations* — sized from `ratios[2]`, at
    /// least one when that fraction is positive, never all of them (and none
    /// when there are fewer than two relations) — is held out; every triple of
    /// those relations goes to `transfer`, so those relations appear in no
    /// other split and `transfer` measures cross-relation generalization. The
    /// remaining triples are split into train/valid/test by
    /// `[ratios[0], ratios[1], ratios[3]]`. `ratios` is
    /// `[train, valid, transfer, test]`.
    pub fn split4(&self, seed: u64, ratios: [f64; 4]) -> Result<Split4> {
        validate_ratios(&ratios)?;
        // Held-out relations: a deterministic shuffle of relation ids, take k.
        let nr = self.num_relations;
        let mut rel_ids: Vec<u32> = (0..nr as u32).collect();
        fisher_yates(&mut rel_ids, seed ^ 0x7b1e_0c4d_9a62_f3e5);
        let want = ((ratios[2] / ratios.iter().sum::<f64>()) * nr as f64).round() as usize;
        let n_hold = if ratios[2] > 0.0 && nr >= 2 {
            want.clamp(1, nr - 1)
        } else {
            0
        };
        let hold: BTreeSet<u32> = rel_ids.into_iter().take(n_hold).collect();

        // Partition triple indices into the transfer relations vs the rest.
        let mut rest: Vec<usize> = Vec::new();
        let mut transfer_idx: Vec<usize> = Vec::new();
        for (i, tr) in self.triples.iter().enumerate() {
            if hold.contains(&tr.r) {
                transfer_idx.push(i);
            } else {
                rest.push(i);
            }
        }
        // Shuffle the rest, then slice by the three non-transfer ratios.
        fisher_yates(&mut rest, seed ^ 0x5f3d_a1c9_2b47_e618);
        let m = rest.len();
        let denom = ratios[0] + ratios[1] + ratios[3];
        let n_train = frac(ratios[0], denom.max(f64::MIN_POSITIVE), m).min(m);
        let n_valid = frac(ratios[1], denom.max(f64::MIN_POSITIVE), m).min(m - n_train);
        Ok(Split4 {
            train: self.pick(&rest[..n_train]),
            valid: self.pick(&rest[n_train..n_train + n_valid]),
            transfer: self.pick(&transfer_idx),
            test: self.pick(&rest[n_train + n_valid..]),
        })
    }

    fn shuffled_indices(&self, seed: u64) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.triples.len()).collect();
        fisher_yates(&mut idx, seed ^ 0x5f3d_a1c9_2b47_e618);
        idx
    }

    fn pick(&self, range: &[usize]) -> Vec<Triple> {
        let mut v: Vec<Triple> = range.iter().map(|&i| self.triples[i]).collect();
        v.sort_by_key(|t| (t.s, t.r, t.o));
        v
    }
}

/// Validate split ratios (finite, non-negative, positive sum); return the sum.
fn validate_ratios(ratios: &[f64]) -> Result<f64> {
    for &r in ratios {
        if !r.is_finite() || r < 0.0 {
            return Err(KgeError::Invalid(
                "split ratio must be finite and >= 0".into(),
            ));
        }
    }
    let sum: f64 = ratios.iter().sum();
    if sum <= 0.0 {
        return Err(KgeError::Invalid("split ratios must sum to > 0".into()));
    }
    Ok(sum)
}

/// `round(part / sum * n)` as a count.
fn frac(part: f64, sum: f64, n: usize) -> usize {
    ((part / sum) * n as f64).round() as usize
}

/// Deterministic in-place Fisher-Yates shuffle seeded through [`Rng`].
fn fisher_yates<T>(v: &mut [T], seed: u64) {
    let mut rng = Rng::seeded(seed);
    for i in (1..v.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        v.swap(i, j);
    }
}

/// A frozen train/valid/test partition. Instance-disjoint by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    pub train: Vec<Triple>,
    pub valid: Vec<Triple>,
    pub test: Vec<Triple>,
}

impl Split {
    /// Verify no triple appears in more than one part. Returns a typed error
    /// (not a panic) so callers gate proposals on it (ADR-006 §1).
    pub fn assert_disjoint(&self) -> Result<()> {
        let mut seen: HashSet<Triple> = HashSet::new();
        for part in [&self.train, &self.valid, &self.test] {
            for &t in part {
                if !seen.insert(t) {
                    return Err(KgeError::Invalid("splits are not instance-disjoint".into()));
                }
            }
        }
        Ok(())
    }

    pub fn total(&self) -> usize {
        self.train.len() + self.valid.len() + self.test.len()
    }
}

/// A frozen 4-way partition with a transfer holdout (see [`TripleStore::split4`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split4 {
    pub train: Vec<Triple>,
    pub valid: Vec<Triple>,
    pub transfer: Vec<Triple>,
    pub test: Vec<Triple>,
}

impl Split4 {
    /// Verify no triple appears in more than one part (typed, not a panic).
    pub fn assert_disjoint(&self) -> Result<()> {
        let mut seen: HashSet<Triple> = HashSet::new();
        for part in [&self.train, &self.valid, &self.transfer, &self.test] {
            for &t in part {
                if !seen.insert(t) {
                    return Err(KgeError::Invalid("splits are not instance-disjoint".into()));
                }
            }
        }
        Ok(())
    }

    pub fn total(&self) -> usize {
        self.train.len() + self.valid.len() + self.transfer.len() + self.test.len()
    }

    /// The relations held out into `transfer` — absent from every other part.
    pub fn transfer_relations(&self) -> BTreeSet<RelationId> {
        self.transfer.iter().map(|t| t.r).collect()
    }
}

#[cfg(test)]
mod tests;
