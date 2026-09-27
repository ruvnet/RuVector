//! Reciprocal relations (Lacroix et al. 2018, arXiv:1806.07297; ADR-007 §3).
//!
//! **Id convention** (shared with evaluation and the bindings): a table
//! trained with `reciprocal: true` has `2·R` relation rows, `R =
//! tables.num_relations() / 2`; row `r ∈ [0, R)` is the base relation and row
//! `r + R` is its inverse `r⁻¹`. Every training triple `(s, r, o)` is joined
//! by `(o, r⁻¹, s)`, both are shuffled together, and each example is trained
//! as a **tail** query only. A head query `(?, r, o)` is then answered as the
//! tail query `(o, r⁻¹, ?)`, filtered with `true_heads(r, o)`.

use crate::data::TripleStore;
use crate::{KgeError, RelationId, Result, Tables, Triple};

/// Number of base relations `R` in a reciprocal table (`num_relations / 2`).
pub fn base_relations(tables: &Tables) -> usize {
    tables.num_relations() / 2
}

/// The inverse row `r⁻¹ = r + R` for base relation `r` of a reciprocal table.
pub fn inverse_relation(tables: &Tables, r: RelationId) -> Result<RelationId> {
    let base = base_relations(tables);
    if (r as usize) >= base {
        return Err(KgeError::UnknownRelation(r));
    }
    Ok(r + base as RelationId)
}

/// The reciprocal triple `(o, r⁻¹, s)`.
pub fn reciprocal_triple(tables: &Tables, t: Triple) -> Result<Triple> {
    Ok(Triple::new(t.o, inverse_relation(tables, t.r)?, t.s))
}

/// `triples` followed by their reciprocals (in the same order).
pub fn augment(tables: &Tables, triples: &[Triple]) -> Result<Vec<Triple>> {
    let mut out = Vec::with_capacity(triples.len() * 2);
    out.extend_from_slice(triples);
    for &t in triples {
        out.push(reciprocal_triple(tables, t)?);
    }
    Ok(out)
}

/// Reject a table that cannot hold the inverse rows for `store`: the relation
/// row count must be even and `≥ 2 · store.num_relations()`.
pub(crate) fn validate(tables: &Tables, store: &TripleStore) -> Result<()> {
    let nr = tables.num_relations();
    if !nr.is_multiple_of(2) || nr / 2 < store.num_relations() {
        return Err(KgeError::Invalid(format!(
            "reciprocal training needs an even number of relation rows >= 2 x {} \
             (row r + R is r's inverse), got {nr}",
            store.num_relations()
        )));
    }
    Ok(())
}
