//! Entity and relation embedding tables: dense row-major `Vec<f32>`,
//! deterministic initialisation from a seed (no `rand`, no `RandomState`).

use crate::{EntityId, KgeError, RelationId, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tables {
    dims: usize,
    entities: Vec<f32>,
    relations: Vec<f32>,
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
}
