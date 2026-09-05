//! Deterministic synthetic embedding generation for agent-memory PoC graphs.

use rand::rngs::StdRng;
use rand::Rng;

/// A synthetic memory embedding: a low-dimensional vector drawn from one of
/// `K` topic clusters plus isotropic noise, so ground-truth topic membership
/// is known exactly (needed to score associative-retrieval recall honestly).
#[derive(Debug, Clone)]
pub struct Embedding {
    pub vec: Vec<f32>,
    pub topic: usize,
}

pub struct TopicSpace {
    pub dim: usize,
    pub centroids: Vec<Vec<f32>>,
}

impl TopicSpace {
    /// Builds `n_topics` centroids spread on the unit hypersphere via a fixed
    /// deterministic RNG so the whole PoC is reproducible from one seed.
    pub fn new(dim: usize, n_topics: usize, rng: &mut StdRng) -> Self {
        let mut centroids = Vec::with_capacity(n_topics);
        for _ in 0..n_topics {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            normalize(&mut v);
            centroids.push(v);
        }
        Self { dim, centroids }
    }

    pub fn sample(&self, topic: usize, noise: f32, rng: &mut StdRng) -> Embedding {
        let mut v = self.centroids[topic].clone();
        for x in v.iter_mut() {
            *x += rng.gen_range(-noise..noise);
        }
        normalize(&mut v);
        Embedding { vec: v, topic }
    }
}

pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn centroids_are_unit_norm() {
        let mut rng = StdRng::seed_from_u64(7);
        let space = TopicSpace::new(16, 8, &mut rng);
        for c in &space.centroids {
            let n: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn same_topic_more_similar_than_different_topic() {
        let mut rng = StdRng::seed_from_u64(7);
        let space = TopicSpace::new(16, 8, &mut rng);
        let a = space.sample(0, 0.05, &mut rng);
        let b = space.sample(0, 0.05, &mut rng);
        let c = space.sample(1, 0.05, &mut rng);
        assert!(cosine(&a.vec, &b.vec) > cosine(&a.vec, &c.vec));
    }
}
