//! The atomic record: a **constellation** — one object measured through many
//! lenses, stored as distinct typed slots that are *never flattened*.

use std::collections::BTreeMap;

use crate::ledger::Anchor;
use crate::lens::LensManifest;
use crate::scoring::{cosine_sim, normalize};

/// One object measured through many lenses.
///
/// Slots are keyed by lens name and kept separate.  The deliberately lossy
/// [`Constellation::flatten`] exists only to build the single-embedding RAG
/// *baseline* the association-native design is meant to beat.
#[derive(Debug, Clone)]
pub struct Constellation {
    /// Stable identifier.
    pub id: u64,
    /// Optional human-readable label (debugging only).
    pub label: Option<String>,
    /// Per-lens slot vectors, keyed by lens name. Never collapsed into one.
    pub slots: BTreeMap<String, Vec<f32>>,
    /// Grounding anchors tying this record to real-world outcomes.
    pub anchors: Vec<Anchor>,
}

impl Constellation {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            label: None,
            slots: BTreeMap::new(),
            anchors: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Attach a slot vector for the given lens.
    pub fn with_slot(mut self, lens: impl Into<String>, vector: Vec<f32>) -> Self {
        self.slots.insert(lens.into(), vector);
        self
    }

    /// Attach a grounding anchor.
    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.anchors.push(anchor);
        self
    }

    pub fn slot(&self, lens: &str) -> Option<&[f32]> {
        self.slots.get(lens).map(|v| v.as_slice())
    }

    /// The deliberately lossy baseline: L2-normalize each slot (so no single
    /// lens dominates by magnitude) then concatenate in manifest order into one
    /// flat vector. This is exactly the "semantic fog" Calyx argues against —
    /// a discriminative minority lens (e.g. 16 of 96 dims) is drowned out.
    pub fn flatten(&self, lenses: &[LensManifest]) -> Vec<f32> {
        let mut out = Vec::new();
        for m in lenses {
            match self.slots.get(&m.name) {
                Some(v) => out.extend_from_slice(&normalize(v)),
                None => out.resize(out.len() + m.dims, 0.0),
            }
        }
        out
    }
}

/// Errors a store raises rather than silently returning a wrong answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// A record referenced a lens not declared in the registry.
    UnknownLens(String),
    /// A slot vector did not match its lens's declared dimensionality.
    ShapeMismatch {
        lens: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::UnknownLens(l) => write!(f, "unknown lens: {l}"),
            StoreError::ShapeMismatch {
                lens,
                expected,
                got,
            } => write!(
                f,
                "shape mismatch on lens {lens}: expected {expected} dims, got {got}"
            ),
        }
    }
}

/// A registry of lenses plus the constellations measured through them.
///
/// Search is exact (brute force); the focus of this crate is the *association*
/// layer, not the index. A production deployment would back each per-lens
/// ranking with HNSW/IVF (ruvector already ships these).
#[derive(Debug, Default, Clone)]
pub struct ConstellationStore {
    lenses: Vec<LensManifest>,
    records: Vec<Constellation>,
}

impl ConstellationStore {
    pub fn new(lenses: Vec<LensManifest>) -> Self {
        Self {
            lenses,
            records: Vec::new(),
        }
    }

    pub fn lenses(&self) -> &[LensManifest] {
        &self.lenses
    }

    pub fn records(&self) -> &[Constellation] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn lens(&self, name: &str) -> Option<&LensManifest> {
        self.lenses.iter().find(|m| m.name == name)
    }

    /// Insert a constellation after validating every slot against the registry.
    /// Fails closed: an unknown lens or a wrong-shaped slot is rejected.
    pub fn insert(&mut self, c: Constellation) -> Result<(), StoreError> {
        for (name, v) in &c.slots {
            let m = self
                .lens(name)
                .ok_or_else(|| StoreError::UnknownLens(name.clone()))?;
            if v.len() != m.dims {
                return Err(StoreError::ShapeMismatch {
                    lens: name.clone(),
                    expected: m.dims,
                    got: v.len(),
                });
            }
        }
        self.records.push(c);
        Ok(())
    }

    pub fn get(&self, id: u64) -> Option<&Constellation> {
        self.records.iter().find(|c| c.id == id)
    }

    /// Rank records by cosine similarity of one lens slot to `query`.
    /// Returns `(id, score)` pairs sorted by descending score, length `top_k`.
    /// Records missing the lens slot are skipped.
    pub fn search_lens(&self, lens: &str, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .records
            .iter()
            .filter_map(|c| c.slot(lens).map(|s| (c.id, cosine_sim(s, query))))
            .collect();
        sort_desc(&mut scored);
        scored.truncate(top_k);
        scored
    }

    /// Rank records by cosine similarity in the flattened baseline space.
    pub fn search_flat(&self, query_flat: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .records
            .iter()
            .map(|c| (c.id, cosine_sim(&c.flatten(&self.lenses), query_flat)))
            .collect();
        sort_desc(&mut scored);
        scored.truncate(top_k);
        scored
    }
}

/// Stable descending sort by score, ties broken by ascending id for
/// determinism.
pub(crate) fn sort_desc(scored: &mut [(u64, f32)]) {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::LensKind;

    fn registry() -> Vec<LensManifest> {
        vec![
            LensManifest::new("sem", "v1", LensKind::DenseSemantic, 2, 1.0, 1.0),
            LensManifest::new("str", "v1", LensKind::Structural, 2, 1.0, 1.0),
        ]
    }

    #[test]
    fn rejects_unknown_lens_and_bad_shape() {
        let mut s = ConstellationStore::new(registry());
        let bad = Constellation::new(1).with_slot("nope", vec![1.0, 0.0]);
        assert!(matches!(s.insert(bad), Err(StoreError::UnknownLens(_))));
        let wrong = Constellation::new(2).with_slot("sem", vec![1.0]);
        assert!(matches!(
            s.insert(wrong),
            Err(StoreError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn per_lens_search_ranks_by_cosine() {
        let mut s = ConstellationStore::new(registry());
        s.insert(Constellation::new(1).with_slot("sem", vec![1.0, 0.0]))
            .unwrap();
        s.insert(Constellation::new(2).with_slot("sem", vec![0.0, 1.0]))
            .unwrap();
        let r = s.search_lens("sem", &[0.9, 0.1], 2);
        assert_eq!(r[0].0, 1);
    }
}
