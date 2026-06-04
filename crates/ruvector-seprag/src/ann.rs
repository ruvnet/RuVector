//! Minimal Vamana-style approximate-nearest-neighbour engine + metric-drift
//! helpers, shared by the BET-1 experiments (ADR-200).
//!
//! Vectors are plain `Vec<f32>` and the metric is squared-L2; metric *drift* is
//! modelled by transforming the vectors (a re-metrization `M = AᵀA` is L2 in the
//! transformed space), so the ANN code itself never needs a weight vector.
//!
//! The search is a standard two-heap greedy beam search (frontier min-heap +
//! bounded result max-heap), which scales to n≥10⁵ where the earlier
//! linear-scan beam did not.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub type Vec32 = Vec<f32>;

/// Squared-L2 distance.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Total-order wrapper for f32 so it can live in a `BinaryHeap`.
#[derive(Clone, Copy, PartialEq)]
struct F(f32);
impl Eq for F {}
impl PartialOrd for F {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for F {
    fn cmp(&self, o: &Self) -> Ordering { self.0.total_cmp(&o.0) }
}

pub struct AnnParams { pub r: usize, pub l: usize, pub alpha: f32, pub k: usize }

/// Brute-force exact top-k (the ground-truth oracle), excluding `q` itself.
#[must_use]
pub fn brute_topk(vecs: &[Vec32], q: usize, k: usize) -> Vec<u32> {
    let mut d: Vec<(f32, u32)> = (0..vecs.len())
        .filter(|&j| j != q)
        .map(|j| (l2(&vecs[q], &vecs[j]), j as u32))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.truncate(k);
    d.into_iter().map(|(_, n)| n).collect()
}

#[must_use]
pub fn recall(got: &[u32], truth: &[u32]) -> f64 {
    use std::collections::HashSet;
    let t: HashSet<u32> = truth.iter().copied().collect();
    got.iter().filter(|g| t.contains(g)).count() as f64 / truth.len().max(1) as f64
}

#[must_use]
pub fn medoid(vecs: &[Vec32]) -> u32 {
    let dim = vecs[0].len();
    let mut c = vec![0.0f32; dim];
    for v in vecs {
        for i in 0..dim { c[i] += v[i]; }
    }
    for x in &mut c { *x /= vecs.len() as f32; }
    (0..vecs.len()).min_by(|&a, &b| l2(&vecs[a], &c).total_cmp(&l2(&vecs[b], &c))).unwrap() as u32
}

/// Two-heap greedy beam search. Returns (top-k, visited nodes, #distance evals).
#[must_use]
pub fn search(graph: &[Vec<u32>], vecs: &[Vec32], entry: u32, query: &[f32], beam: usize, k: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let mut visited = vec![false; vecs.len()];
    let mut frontier: BinaryHeap<std::cmp::Reverse<(F, u32)>> = BinaryHeap::new(); // nearest first
    let mut result: BinaryHeap<(F, u32)> = BinaryHeap::new(); // worst (max) on top, capped at beam
    let d0 = l2(&vecs[entry as usize], query);
    visited[entry as usize] = true;
    frontier.push(std::cmp::Reverse((F(d0), entry)));
    result.push((F(d0), entry));
    let mut visited_list = vec![entry];
    let mut evals = 1usize;

    while let Some(std::cmp::Reverse((F(d), u))) = frontier.pop() {
        if result.len() >= beam && d > result.peek().unwrap().0 .0 {
            break;
        }
        for &v in &graph[u as usize] {
            if visited[v as usize] {
                continue;
            }
            visited[v as usize] = true;
            visited_list.push(v);
            let dv = l2(&vecs[v as usize], query);
            evals += 1;
            if result.len() < beam || dv < result.peek().unwrap().0 .0 {
                frontier.push(std::cmp::Reverse((F(dv), v)));
                result.push((F(dv), v));
                if result.len() > beam {
                    result.pop();
                }
            }
        }
    }
    let mut out: Vec<(f32, u32)> = result.into_iter().map(|(F(d), n)| (d, n)).collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.truncate(k);
    (out.into_iter().map(|(_, n)| n).collect(), visited_list, evals)
}

/// Vamana RobustPrune: keep up to R diverse neighbours.
fn robust_prune(p: u32, cands: &[u32], vecs: &[Vec32], alpha: f32, r: usize) -> Vec<u32> {
    let mut pool: Vec<(f32, u32)> = cands.iter().filter(|&&c| c != p).map(|&c| (l2(&vecs[p as usize], &vecs[c as usize]), c)).collect();
    pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<u32> = Vec::new();
    while !pool.is_empty() && out.len() < r {
        let (_, pstar) = pool[0];
        out.push(pstar);
        pool.retain(|&(dq, q)| alpha * l2(&vecs[pstar as usize], &vecs[q as usize]) > dq && q != pstar);
    }
    out
}

/// Build a Vamana-lite graph. Returns (graph, #distance evals, wall-build helper deferred to caller).
#[must_use]
pub fn build(vecs: &[Vec32], p: &AnnParams, seed: u64) -> Vec<Vec<u32>> {
    let n = vecs.len();
    let mut rng = Rng::new(seed);
    let mut graph: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut s = std::collections::HashSet::new();
            while s.len() < p.r.min(n - 1) {
                let j = rng.below(n);
                if j != i { s.insert(j as u32); }
            }
            s.into_iter().collect()
        })
        .collect();
    let med = medoid(vecs);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() { order.swap(i, rng.below(i + 1)); }
    for &node in &order {
        let (_, visited, _) = search(&graph, vecs, med, &vecs[node], p.l, p.k);
        let nbrs = robust_prune(node as u32, &visited, vecs, p.alpha, p.r);
        graph[node] = nbrs.clone();
        for q in nbrs {
            let qi = q as usize;
            if !graph[qi].contains(&(node as u32)) {
                graph[qi].push(node as u32);
                if graph[qi].len() > p.r {
                    let cand = graph[qi].clone();
                    graph[qi] = robust_prune(q, &cand, vecs, p.alpha, p.r);
                }
            }
        }
    }
    graph
}

// ---------------- deterministic RNG ----------------
pub struct Rng(u64);
impl Rng {
    #[must_use]
    pub fn new(s: u64) -> Self { Rng(s) }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn f32(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// ---------------- metric-drift transforms ----------------
pub fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = (rng.f32() as f64).max(1e-9);
    let u2 = rng.f32() as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

pub fn random_rotation(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut m: Vec<Vec<f32>> = (0..dim).map(|_| (0..dim).map(|_| gaussian(rng)).collect()).collect();
    for i in 0..dim {
        for j in 0..i {
            let dot: f32 = (0..dim).map(|k| m[i][k] * m[j][k]).sum();
            for k in 0..dim { m[i][k] -= dot * m[j][k]; }
        }
        let norm: f32 = m[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for k in 0..dim { m[i][k] /= norm; }
    }
    m.into_iter().flatten().collect()
}

pub fn identity(dim: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim { a[i * dim + i] = 1.0; }
    a
}

pub fn target_diag(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim { a[i * dim + i] = (0.2 + 2.8 * rng.f32()).sqrt(); }
    a
}

pub fn target_rot(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let r = random_rotation(dim, rng);
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim {
        let s = (0.2 + 2.8 * rng.f32()).sqrt();
        for j in 0..dim { a[i * dim + j] = s * r[i * dim + j]; }
    }
    a
}

pub fn lerp_mat(a0: &[f32], a1: &[f32], t: f32) -> Vec<f32> {
    a0.iter().zip(a1).map(|(x, y)| x * (1.0 - t) + y * t).collect()
}

pub fn apply_linear(a: &[f32], vecs: &[Vec32], dim: usize) -> Vec<Vec32> {
    vecs.iter().map(|v| {
        (0..dim).map(|i| {
            let row = &a[i * dim..(i + 1) * dim];
            row.iter().zip(v).map(|(x, y)| x * y).sum()
        }).collect()
    }).collect()
}

pub fn apply_nonlin(w: &[f32], vecs: &[Vec32], s: f32, dim: usize) -> Vec<Vec32> {
    vecs.iter().map(|v| {
        (0..dim).map(|i| {
            let row = &w[i * dim..(i + 1) * dim];
            let u: f32 = row.iter().zip(v).map(|(x, y)| x * y).sum();
            v[i] + s * u.tanh()
        }).collect()
    }).collect()
}

/// Read up to `n` comma-separated f32 rows from a CSV.
#[must_use]
pub fn read_vectors(path: &str, n: usize) -> Vec<Vec32> {
    let data = std::fs::read_to_string(path).expect("read features");
    data.lines().take(n)
        .map(|l| l.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect())
        .filter(|v: &Vec32| !v.is_empty())
        .collect()
}
