//! Deterministic synthetic insert workload alternating calm and bursty
//! coherence-drift phases.
//!
//! Calm phases insert noisy samples around a fixed cluster centroid (low
//! drift). Burst phases linearly walk the centroid to a fresh random
//! direction over the phase (high, sustained drift) — modelling an agent
//! whose working context suddenly shifts topic.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// One insert event: the embedding vector written to the store.
#[derive(Debug, Clone)]
pub struct WorkloadEvent {
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    pub dims: usize,
    pub n_events: usize,
    pub calm_phase_len: usize,
    pub burst_phase_len: usize,
    pub noise: f32,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            dims: 32,
            n_events: 4000,
            calm_phase_len: 300,
            burst_phase_len: 60,
            noise: 0.05,
        }
    }
}

fn random_unit_vector(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    raw.into_iter().map(|x| x / norm).collect()
}

/// Generate a deterministic event stream for a given `seed`.
///
/// The stream alternates: calm phase (events cluster tightly around a fixed
/// centroid) then burst phase (the centroid is linearly interpolated toward
/// a freshly sampled direction, one step per event) — repeating until
/// `n_events` events have been produced.
pub fn generate_workload(cfg: &WorkloadConfig, seed: u64) -> Vec<WorkloadEvent> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut events = Vec::with_capacity(cfg.n_events);

    let mut centroid = random_unit_vector(&mut rng, cfg.dims);
    let mut i = 0usize;
    while i < cfg.n_events {
        // Calm phase: sample noisily around the current centroid.
        let calm_len = cfg.calm_phase_len.min(cfg.n_events - i);
        for _ in 0..calm_len {
            let v: Vec<f32> = centroid
                .iter()
                .map(|c| c + rng.gen_range(-cfg.noise..cfg.noise))
                .collect();
            events.push(WorkloadEvent { vector: v });
        }
        i += calm_len;
        if i >= cfg.n_events {
            break;
        }

        // Burst phase: walk the centroid toward a new random direction.
        let target = random_unit_vector(&mut rng, cfg.dims);
        let burst_len = cfg.burst_phase_len.min(cfg.n_events - i);
        for step in 0..burst_len {
            let t = (step + 1) as f32 / burst_len as f32;
            let walk: Vec<f32> = centroid
                .iter()
                .zip(target.iter())
                .map(|(c, t2)| c + (t2 - c) * t)
                .collect();
            let v: Vec<f32> = walk
                .iter()
                .map(|c| c + rng.gen_range(-cfg.noise..cfg.noise))
                .collect();
            events.push(WorkloadEvent { vector: v });
        }
        centroid = target;
        i += burst_len;
    }

    events.truncate(cfg.n_events);
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_exact_event_count() {
        let cfg = WorkloadConfig {
            dims: 8,
            n_events: 137,
            calm_phase_len: 20,
            burst_phase_len: 5,
            noise: 0.01,
        };
        let events = generate_workload(&cfg, 42);
        assert_eq!(events.len(), 137);
        for e in &events {
            assert_eq!(e.vector.len(), 8);
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let cfg = WorkloadConfig::default();
        let a = generate_workload(&cfg, 7);
        let b = generate_workload(&cfg, 7);
        assert_eq!(a.len(), b.len());
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert_eq!(ea.vector, eb.vector);
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let cfg = WorkloadConfig::default();
        let a = generate_workload(&cfg, 1);
        let b = generate_workload(&cfg, 2);
        assert_ne!(a[0].vector, b[0].vector);
    }
}
