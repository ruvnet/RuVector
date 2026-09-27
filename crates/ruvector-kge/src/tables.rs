//! Entity and relation embedding tables: dense row-major `Vec<f32>`,
//! deterministic initialisation from a seed (no `rand`, no `RandomState`).
//!
//! Size limits (ADR-005, F1): a table is rejected with a typed
//! [`KgeError::Limit`] when `dims` is outside `1..=MAX_DIMS` or the total f32
//! storage `(entities + relations) * dims * 4` exceeds [`MAX_TABLE_BYTES`].
//! [`Tables::try_new`] rejects before allocating; deserialization rejects
//! before the table is used (the parsed arrays are bounded by the input, and
//! the bindings run an allocation-free header check before parsing at all).
//! The infallible [`Tables::new`] is for callers that already validated.

use crate::adversarial::MAX_DIMS;
use crate::{EntityId, KgeError, RelationId, Result};
use serde::{Deserialize, Serialize};

/// Hard ceiling on one model's embedding storage: 2 GiB of f32. Callers may
/// pass a *lower* cap to [`check_table_size`]; a higher one is clamped to this.
/// Chosen so a single call can never exhaust a 32-bit (wasm32) address space
/// and stays well inside a desktop's RAM.
pub const MAX_TABLE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The effective byte cap: `requested` clamped to [`MAX_TABLE_BYTES`]
/// (configurable only downward).
pub fn effective_max_bytes(requested: Option<u64>) -> u64 {
    requested.map_or(MAX_TABLE_BYTES, |r| r.min(MAX_TABLE_BYTES))
}

/// Validate a prospective table shape before allocating it. Arithmetic is in
/// `u64` with overflow checks (wasm32 `usize` is 32-bit).
pub fn check_table_size(
    num_entities: usize,
    num_relations: usize,
    dims: usize,
    max_bytes: Option<u64>,
) -> Result<()> {
    if dims == 0 || dims > MAX_DIMS {
        return Err(KgeError::Limit("dims out of range (1..=4096)"));
    }
    let bytes = (num_entities as u64)
        .checked_add(num_relations as u64)
        .and_then(|rows| rows.checked_mul(dims as u64))
        .and_then(|n| n.checked_mul(4));
    match bytes {
        Some(b) if b <= effective_max_bytes(max_bytes) => Ok(()),
        _ => Err(KgeError::Limit(
            "embedding tables would exceed the byte cap",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TablesWire")]
pub struct Tables {
    dims: usize,
    entities: Vec<f32>,
    relations: Vec<f32>,
}

/// Unvalidated wire shape; converted into [`Tables`] only if it is well-formed
/// and within the limits (a `dims: 0` payload would otherwise divide by zero).
#[derive(Deserialize)]
struct TablesWire {
    dims: usize,
    entities: Vec<f32>,
    relations: Vec<f32>,
}

impl TryFrom<TablesWire> for Tables {
    type Error = KgeError;
    fn try_from(w: TablesWire) -> Result<Self> {
        if w.dims == 0
            || !w.entities.len().is_multiple_of(w.dims)
            || !w.relations.len().is_multiple_of(w.dims)
        {
            return Err(KgeError::Invalid(
                "table length is not a multiple of dims".to_string(),
            ));
        }
        check_table_size(
            w.entities.len() / w.dims,
            w.relations.len() / w.dims,
            w.dims,
            None,
        )?;
        Ok(Self {
            dims: w.dims,
            entities: w.entities,
            relations: w.relations,
        })
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// splitmix64 finaliser: spreads adjacent seeds into unrelated, non-zero states.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z | 1
}

/// Uniform in [-bound, bound), Xavier-style bound = sqrt(6 / (fan_in + fan_out)).
fn fill_uniform(v: &mut [f32], bound: f32, seed: u64) {
    let mut state = splitmix64(seed);
    for x in v.iter_mut() {
        let u = (xorshift64(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
        *x = (2.0 * u - 1.0) * bound;
    }
}

impl Tables {
    /// Validating constructor: rejects (without allocating) a shape that
    /// fails [`check_table_size`] under `max_bytes` (clamped to
    /// [`MAX_TABLE_BYTES`]).
    pub fn try_new(
        num_entities: usize,
        num_relations: usize,
        dims: usize,
        seed: u64,
        max_bytes: Option<u64>,
    ) -> Result<Self> {
        check_table_size(num_entities, num_relations, dims, max_bytes)?;
        Ok(Self::new(num_entities, num_relations, dims, seed))
    }

    /// Unchecked constructor for already-validated shapes (see [`Self::try_new`]).
    pub fn new(num_entities: usize, num_relations: usize, dims: usize, seed: u64) -> Self {
        let bound = (6.0 / (dims as f32 + dims as f32)).sqrt();
        let mut entities = vec![0.0; num_entities * dims];
        let mut relations = vec![0.0; num_relations * dims];
        fill_uniform(&mut entities, bound, seed ^ 0x9e37_79b9_7f4a_7c15);
        fill_uniform(&mut relations, bound, seed ^ 0xbf58_476d_1ce4_e5b9);
        Self {
            dims,
            entities,
            relations,
        }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
    pub fn num_entities(&self) -> usize {
        self.entities.len() / self.dims
    }
    pub fn num_relations(&self) -> usize {
        self.relations.len() / self.dims
    }

    pub fn entity(&self, id: EntityId) -> Result<&[f32]> {
        let i = id as usize;
        if i >= self.num_entities() {
            return Err(KgeError::UnknownEntity(id));
        }
        Ok(&self.entities[i * self.dims..(i + 1) * self.dims])
    }
    pub fn relation(&self, id: RelationId) -> Result<&[f32]> {
        let i = id as usize;
        if i >= self.num_relations() {
            return Err(KgeError::UnknownRelation(id));
        }
        Ok(&self.relations[i * self.dims..(i + 1) * self.dims])
    }
    pub fn entity_mut(&mut self, id: EntityId) -> Result<&mut [f32]> {
        let i = id as usize;
        if i >= self.num_entities() {
            return Err(KgeError::UnknownEntity(id));
        }
        let d = self.dims;
        Ok(&mut self.entities[i * d..(i + 1) * d])
    }
    pub fn relation_mut(&mut self, id: RelationId) -> Result<&mut [f32]> {
        let i = id as usize;
        if i >= self.num_relations() {
            return Err(KgeError::UnknownRelation(id));
        }
        let d = self.dims;
        Ok(&mut self.relations[i * d..(i + 1) * d])
    }
    pub fn entities_raw(&self) -> &[f32] {
        &self.entities
    }
    pub fn relations_raw(&self) -> &[f32] {
        &self.relations
    }
    pub fn entities_raw_mut(&mut self) -> &mut [f32] {
        &mut self.entities
    }
    pub fn relations_raw_mut(&mut self) -> &mut [f32] {
        &mut self.relations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_bounded() {
        let a = Tables::new(10, 3, 8, 42);
        let b = Tables::new(10, 3, 8, 42);
        assert_eq!(a, b);
        assert_ne!(a, Tables::new(10, 3, 8, 43));
        let bound = (6.0f32 / 16.0).sqrt();
        assert!(a.entities_raw().iter().all(|x| x.abs() <= bound));
        assert!(a.entity(10).is_err());
        assert_eq!(a.relation(2).unwrap().len(), 8);
    }

    #[test]
    fn f1_dims_cap_enforced() {
        assert!(matches!(
            Tables::try_new(10, 3, 50_000, 1, None),
            Err(KgeError::Limit(_))
        ));
        assert!(matches!(
            Tables::try_new(10, 3, 0, 1, None),
            Err(KgeError::Limit(_))
        ));
        assert!(Tables::try_new(10, 3, MAX_DIMS, 1, None).is_ok());
    }

    #[test]
    fn f1_byte_cap_enforced_and_only_downward() {
        // 1M entities x 4096 dims x 4 B = 16 GiB: rejected before allocation.
        assert!(matches!(
            check_table_size(1_000_000, 1, 4096, None),
            Err(KgeError::Limit(_))
        ));
        // Exactly at the cap is allowed; one row over is not.
        let rows = (MAX_TABLE_BYTES / (4 * 256)) as usize;
        assert!(check_table_size(rows - 1, 1, 256, None).is_ok());
        assert!(check_table_size(rows, 1, 256, None).is_err());
        // A lower cap applies; a higher one is clamped to MAX_TABLE_BYTES.
        assert!(check_table_size(100, 0, 8, Some(100 * 8 * 4 - 1)).is_err());
        assert!(check_table_size(rows, 1, 256, Some(u64::MAX)).is_err());
        // Overflowing arithmetic is a limit error, not a wrap.
        assert!(check_table_size(usize::MAX, usize::MAX, 4096, None).is_err());
    }

    #[test]
    fn f1_deserialize_validates_shape_and_size() {
        let ok = Tables::new(4, 2, 8, 3);
        let json = serde_json::to_string(&ok).unwrap();
        assert_eq!(serde_json::from_str::<Tables>(&json).unwrap(), ok);
        for bad in [
            r#"{"dims":0,"entities":[],"relations":[]}"#,
            r#"{"dims":50000,"entities":[],"relations":[]}"#,
            r#"{"dims":4,"entities":[1.0,2.0,3.0],"relations":[]}"#,
        ] {
            assert!(serde_json::from_str::<Tables>(bad).is_err(), "{bad}");
        }
    }
}
