//! Core value types. Entities and relations are dense u32 ids; labels are
//! kept outside the core (the bindings map strings to ids) so nothing in here
//! can log or leak text (ADR-005).

use serde::{Deserialize, Serialize};

pub type EntityId = u32;
pub type RelationId = u32;

/// One (subject, relation, object) fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    pub s: EntityId,
    pub r: RelationId,
    pub o: EntityId,
}

impl Triple {
    pub const fn new(s: EntityId, r: RelationId, o: EntityId) -> Self {
        Self { s, r, o }
    }
}

/// Which slot a query leaves open: `(?, r, o)` or `(s, r, ?)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Head,
    Tail,
}

/// Which scorer a model was trained with (ADR-002).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScorerKind {
    /// Circular correlation via FFT — the default, "holographic" scorer.
    Hole,
    /// Rotation in complex space — the composition-capable scorer.
    Rotate,
}

/// A ranked candidate for an open slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub entity: EntityId,
    pub score: f32,
}
