//! Minimal deterministic k-means (k-means++ init + Lloyd refinement).
//! Pure Rust, no unsafe. Suitable for IVF centroid training in this PoC.

use rand::{rngs::StdRng, Rng, SeedableRng};

#[inline]
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// k-means++ seeding: deterministic for a given `seed`.
pub fn kmeans_pp_init(vectors: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(!vectors.is_empty());
    assert!(k <= vectors.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..vectors.len());
    centers.push(vectors[first].clone());

    let mut min_d2 = vec![f32::INFINITY; vectors.len()];
    for (i, v) in vectors.iter().enumerate() {
        min_d2[i] = sq_l2(v, &centers[0]);
    }

    while centers.len() < k {
        let total: f32 = min_d2.iter().sum();
        if total <= 0.0 {
            // duplicates everywhere — pad with the first vector
            centers.push(vectors[0].clone());
            continue;
        }
        let mut t = rng.gen::<f32>() * total;
        let mut chosen = vectors.len() - 1;
        for (i, &d2) in min_d2.iter().enumerate() {
            t -= d2;
            if t <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.push(vectors[chosen].clone());
        let new_c = centers.last().unwrap();
        for (i, v) in vectors.iter().enumerate() {
            let d2 = sq_l2(v, new_c);
            if d2 < min_d2[i] {
                min_d2[i] = d2;
            }
        }
    }

    centers
}

/// Lloyd's algorithm. Mutates `centers` in place. Stops on `max_iters` or
/// when no centroid moves more than 1e-6 squared-L2.
pub fn lloyd_refine(vectors: &[Vec<f32>], centers: &mut [Vec<f32>], max_iters: usize) {
    let dim = vectors[0].len();
    let k = centers.len();
    let mut sums = vec![vec![0.0_f32; dim]; k];
    let mut counts = vec![0usize; k];

    for _iter in 0..max_iters {
        for s in &mut sums {
            for x in s.iter_mut() {
                *x = 0.0;
            }
        }
        for c in counts.iter_mut() {
            *c = 0;
        }

        for v in vectors {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (ci, c) in centers.iter().enumerate() {
                let d = sq_l2(v, c);
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            for (s, x) in sums[best].iter_mut().zip(v.iter()) {
                *s += *x;
            }
            counts[best] += 1;
        }

        let mut max_shift = 0.0_f32;
        for ci in 0..k {
            if counts[ci] == 0 {
                continue;
            }
            let inv = 1.0 / counts[ci] as f32;
            let mut shift = 0.0_f32;
            for d in 0..dim {
                let new_v = sums[ci][d] * inv;
                let diff = new_v - centers[ci][d];
                shift += diff * diff;
                centers[ci][d] = new_v;
            }
            if shift > max_shift {
                max_shift = shift;
            }
        }
        if max_shift < 1e-6 {
            break;
        }
    }
}
