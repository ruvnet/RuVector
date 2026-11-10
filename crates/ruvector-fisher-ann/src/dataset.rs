//! Deterministic synthetic task/query generation.
//!
//! Real Fisher diagonals and LoRA weight deltas are not available inside
//! this sandboxed benchmark (they require a running fine-tuning loop), so
//! the dataset is synthetic. It is built to reproduce one structural
//! property continual-learning literature reports for Fisher information:
//! importance concentrates on a small subset of dimensions per task, and
//! unrelated dimensions vary with high nuisance noise. That asymmetry is
//! exactly what makes a Fisher-weighted metric diverge from plain L2 — see
//! `docs/research/nightly/2026-10-26-fisher-weighted-ewc-ann/README.md`
//! for why this is a faithful, falsifiable stand-in rather than a number
//! tuned to make the hypothesis win.

use crate::types::TaskRecord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// One synthetic "task family": a center in weight-space, a fixed set of
/// dimensions that matter for that family, and the family's overall
/// importance.
struct Family {
    center: Vec<f32>,
    important_dims: Vec<bool>,
    importance: f32,
}

fn make_family(dim: usize, important_frac: f32, rng: &mut StdRng) -> Family {
    let center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let important_dims: Vec<bool> = (0..dim)
        .map(|_| rng.gen::<f32>() < important_frac)
        .collect();
    let importance = rng.gen_range(0.5..2.0);
    Family {
        center,
        important_dims,
        importance,
    }
}

fn sample_task(id: usize, family: &Family, rng: &mut StdRng) -> TaskRecord {
    let dim = family.center.len();
    let mut weights = Vec::with_capacity(dim);
    let mut fisher = Vec::with_capacity(dim);
    for d in 0..dim {
        if family.important_dims[d] {
            // Important dims: tight cluster around the family center (this
            // is the signal), high Fisher.
            weights.push(family.center[d] + rng.gen_range(-0.05..0.05));
            fisher.push(rng.gen_range(5.0..20.0));
        } else {
            // Unimportant dims: large nuisance noise that dominates raw
            // L2 but should be discounted by a Fisher-weighted metric.
            weights.push(family.center[d] + rng.gen_range(-2.0..2.0));
            fisher.push(rng.gen_range(0.0..0.05));
        }
    }
    TaskRecord {
        id,
        weights,
        fisher,
        importance: family.importance,
    }
}

/// Generate `n` deterministic synthetic tasks across `n_families` clusters
/// at dimension `dim`, seeded by `seed`. Same `seed` always reproduces the
/// same dataset byte-for-byte (`StdRng` is a deterministic PRNG).
pub fn generate_tasks(n: usize, dim: usize, n_families: usize, seed: u64) -> Vec<TaskRecord> {
    let mut rng = StdRng::seed_from_u64(seed);
    let families: Vec<Family> = (0..n_families.max(1))
        .map(|_| make_family(dim, 0.1, &mut rng))
        .collect();
    (0..n)
        .map(|i| {
            let family = &families[i % families.len()];
            sample_task(i, family, &mut rng)
        })
        .collect()
}

/// Generate `n_queries` query vectors by perturbing an existing task's
/// weights: a small amount of additional drift on the important dims (the
/// query is "close to, but not identical to" a task the system has seen
/// before) and fresh large nuisance noise on the unimportant dims (a new,
/// irrelevant context). Returns `(query_weights, source_task_id)` pairs;
/// `source_task_id` is diagnostic only — ground truth for recall is always
/// the brute-force baseline's own output, not this label.
pub fn generate_queries(
    tasks: &[TaskRecord],
    n_queries: usize,
    seed: u64,
) -> Vec<(Vec<f32>, usize)> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xF15E_12A9);
    let dim = tasks[0].weights.len();
    (0..n_queries)
        .map(|_| {
            let src = &tasks[rng.gen_range(0..tasks.len())];
            let mut q = Vec::with_capacity(dim);
            for d in 0..dim {
                let important = src.fisher[d] > 1.0;
                if important {
                    q.push(src.weights[d] + rng.gen_range(-0.1..0.1));
                } else {
                    q.push(src.weights[d] + rng.gen_range(-2.0..2.0));
                }
            }
            (q, src.id)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed_same_output() {
        let a = generate_tasks(50, 16, 5, 42);
        let b = generate_tasks(50, 16, 5, 42);
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!(ta.weights, tb.weights);
            assert_eq!(ta.fisher, tb.fisher);
        }
    }

    #[test]
    fn different_seed_different_output() {
        let a = generate_tasks(50, 16, 5, 1);
        let b = generate_tasks(50, 16, 5, 2);
        assert_ne!(a[0].weights, b[0].weights);
    }

    #[test]
    fn important_dims_have_higher_mean_fisher() {
        let tasks = generate_tasks(200, 32, 4, 7);
        let mut important_sum = 0.0f32;
        let mut important_n = 0u32;
        let mut other_sum = 0.0f32;
        let mut other_n = 0u32;
        for t in &tasks {
            for &f in &t.fisher {
                if f > 1.0 {
                    important_sum += f;
                    important_n += 1;
                } else {
                    other_sum += f;
                    other_n += 1;
                }
            }
        }
        assert!(important_n > 0 && other_n > 0);
        assert!(important_sum / important_n as f32 > 10.0 * (other_sum / other_n as f32));
    }
}
