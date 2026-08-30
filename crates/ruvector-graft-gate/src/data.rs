//! Deterministic seeded data generation: cluster centroids, organic
//! ("legit") corpus points, target queries, and the synthetic poison
//! attack model.
//!
//! ## Attack model
//!
//! A real embedding-optimization poisoning attack (e.g. PoisonedRAG-style)
//! crafts a document whose embedding is optimized to maximize similarity
//! to one or more anticipated target queries, rather than being drawn from
//! a topic's natural corpus statistics. This crate formalizes that as:
//!
//! `poison = normalize(alpha * target_query + (1 - alpha) * random_direction)`
//!
//! with `alpha` close to 1 (`config::POISON_ALPHA = 0.7`). This is
//! deliberately *not* a full reproduction of an LLM-embedding attack (that
//! would require a live embedding model and network access, out of scope
//! for this offline, Rust-only nightly) — it is an explicit, falsifiable
//! formalization of the property real optimization-based attacks are
//! documented to have: concentrated similarity toward one target rather
//! than membership in a natural covariance-structured cluster. See the
//! nightly README's "Why This Attack Model" section for the caveat this
//! implies for external validity.

use crate::rng::Xorshift64;
use crate::vector::{normalize_in_place, Vector};

pub fn gen_centroids(rng: &mut Xorshift64, k: usize, dim: usize) -> Vec<Vector> {
    (0..k)
        .map(|_| {
            let mut v: Vector = (0..dim).map(|_| rng.next_gaussian()).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect()
}

/// A point drawn from an isotropic Gaussian ball around `centroid`,
/// re-normalized to unit length — used for both organic corpus points and
/// target queries (both represent genuine topical content).
pub fn gen_ball_point(rng: &mut Xorshift64, centroid: &[f32], sigma: f32) -> Vector {
    let mut v: Vector = centroid
        .iter()
        .map(|&c| c + sigma * rng.next_gaussian())
        .collect();
    normalize_in_place(&mut v);
    v
}

/// See module-level "Attack model" docs.
pub fn gen_poison_vector(rng: &mut Xorshift64, target_query: &[f32], alpha: f32) -> Vector {
    let dim = target_query.len();
    let mut rdir: Vector = (0..dim).map(|_| rng.next_gaussian()).collect();
    normalize_in_place(&mut rdir);
    let mut v: Vector = target_query
        .iter()
        .zip(rdir.iter())
        .map(|(&q, &r)| alpha * q + (1.0 - alpha) * r)
        .collect();
    normalize_in_place(&mut v);
    v
}

/// Deterministic Fisher-Yates shuffle of `0..n`.
pub fn shuffle_indices(rng: &mut Xorshift64, n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    if n < 2 {
        return idx;
    }
    for i in (1..n).rev() {
        let j = rng.next_below(i + 1);
        idx.swap(i, j);
    }
    idx
}

/// One item in the interleaved insertion stream used by the benchmark:
/// either organic corpus growth or an attacker-crafted insertion attempt.
#[derive(Clone)]
pub enum InsertItem {
    Legit {
        content_id: u64,
        vector: Vector,
    },
    Poison {
        content_id: u64,
        target_id: usize,
        vector: Vector,
    },
}

impl InsertItem {
    pub fn vector(&self) -> &[f32] {
        match self {
            InsertItem::Legit { vector, .. } => vector,
            InsertItem::Poison { vector, .. } => vector,
        }
    }

    pub fn content_id(&self) -> u64 {
        match self {
            InsertItem::Legit { content_id, .. } => *content_id,
            InsertItem::Poison { content_id, .. } => *content_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ball_points_are_unit_length_and_near_centroid() {
        let mut rng = Xorshift64::new(1);
        let mut centroid = vec![1.0, 0.0, 0.0, 0.0];
        normalize_in_place(&mut centroid);
        let p = gen_ball_point(&mut rng, &centroid, 0.1);
        let n: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
        let sim: f32 = p.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
        assert!(
            sim > 0.8,
            "expected point to remain near centroid, sim={sim}"
        );
    }

    #[test]
    fn poison_vector_is_biased_toward_query() {
        let mut rng = Xorshift64::new(2);
        let mut query = vec![0.0, 1.0, 0.0, 0.0];
        normalize_in_place(&mut query);
        let poison = gen_poison_vector(&mut rng, &query, 0.9);
        let sim: f32 = poison.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        assert!(
            sim > 0.7,
            "poison should be strongly aligned with query, sim={sim}"
        );
    }

    #[test]
    fn shuffle_is_a_permutation_and_deterministic() {
        let mut r1 = Xorshift64::new(99);
        let mut r2 = Xorshift64::new(99);
        let a = shuffle_indices(&mut r1, 50);
        let b = shuffle_indices(&mut r2, 50);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }
}
