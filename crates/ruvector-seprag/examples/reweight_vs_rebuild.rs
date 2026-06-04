//! BET 1 (ADR-198, decoupled from CCH): does a FIXED proximity-graph topology +
//! cheap re-weighting absorb metric drift as well as a full rebuild?
//!
//! Self-learning systems change their relevance metric over time. A flat ANN
//! index (HNSW/Vamana) is built *for* a metric; when the metric drifts its graph
//! becomes suboptimal and the textbook fix is a costly rebuild. This harness
//! tests whether reusing the old topology under the new metric ("re-weight",
//! zero build cost) keeps recall close to a rebuild — and quantifies how much
//! drift fixed topology tolerates before a rebuild is actually required.
//!
//! Three strategies, recall@10 measured vs brute-force truth under the CURRENT
//! (drifted) metric:
//!   A re-weight : graph built under w0, searched under w_t   (build cost: 0)
//!   B rebuild   : graph rebuilt under w_t, searched under w_t (build cost: full)
//!   C stale     : graph built under w0, searched under w0     (ignores drift; floor)
//!
//! Pre-registered gate — WIN: recall(A) within 2% of recall(B) across the drift
//! sweep. KILL: recall(A) drops >2% below B even at small drift.
//!
//! Run: cargo run --release -p ruvector-seprag --example reweight_vs_rebuild -- <feat.csv> <N>

use std::collections::HashSet;
use std::time::Instant;

type Vec32 = Vec<f32>;

// ----- deterministic RNG (SplitMix64) -----
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// ----- weighted squared-L2 metric -----
#[inline]
fn dist(a: &[f32], b: &[f32], w: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += w[i] * d * d;
    }
    s
}

fn brute_topk(vecs: &[Vec32], w: &[f32], q: usize, k: usize) -> Vec<u32> {
    let mut d: Vec<(f32, u32)> = (0..vecs.len())
        .filter(|&j| j != q)
        .map(|j| (dist(&vecs[q], &vecs[j], w), j as u32))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.truncate(k);
    d.into_iter().map(|(_, n)| n).collect()
}

// ----- Vamana-lite proximity graph -----
struct Params { r: usize, l: usize, alpha: f32, k: usize }

fn medoid(vecs: &[Vec32], w: &[f32]) -> u32 {
    let dim = vecs[0].len();
    let mut c = vec![0.0f32; dim];
    for v in vecs {
        for i in 0..dim { c[i] += v[i]; }
    }
    for x in &mut c { *x /= vecs.len() as f32; }
    (0..vecs.len()).min_by(|&a, &b| dist(&vecs[a], &c, w).partial_cmp(&dist(&vecs[b], &c, w)).unwrap()).unwrap() as u32
}

/// Greedy beam search. Returns (top-k, set of all visited nodes, #distance evals).
fn greedy(graph: &[Vec<u32>], vecs: &[Vec32], w: &[f32], entry: u32, q: &[f32], beam: usize, k: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut expanded: HashSet<u32> = HashSet::new();
    let mut pool: Vec<(f32, u32)> = vec![(dist(&vecs[entry as usize], q, w), entry)];
    seen.insert(entry);
    let mut evals = 1usize;
    loop {
        let next = pool.iter().filter(|(_, n)| !expanded.contains(n)).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).copied();
        let (_, u) = match next { Some(x) => x, None => break };
        expanded.insert(u);
        for &v in &graph[u as usize] {
            if seen.insert(v) {
                pool.push((dist(&vecs[v as usize], q, w), v));
                evals += 1;
            }
        }
        pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pool.truncate(beam);
    }
    pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let topk = pool.iter().take(k).map(|&(_, n)| n).collect();
    let visited = seen.into_iter().collect();
    (topk, visited, evals)
}

/// RobustPrune: keep up to R diverse neighbours (Vamana alpha-pruning).
fn robust_prune(p: u32, cands: &[u32], vecs: &[Vec32], w: &[f32], alpha: f32, r: usize) -> Vec<u32> {
    let mut pool: Vec<(f32, u32)> = cands.iter().filter(|&&c| c != p).map(|&c| (dist(&vecs[p as usize], &vecs[c as usize], w), c)).collect();
    pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut out: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < pool.len() && out.len() < r {
        let (_, pstar) = pool[i];
        out.push(pstar);
        pool.retain(|&(dq, q)| alpha * dist(&vecs[pstar as usize], &vecs[q as usize], w) > dq);
        i = 0; // pool shrank; restart scan from front of remaining
        // skip already-chosen
        pool.retain(|&(_, q)| !out.contains(&q));
    }
    out
}

/// Build a Vamana-lite graph under metric `w`. Returns (graph, #distance evals).
fn build(vecs: &[Vec32], w: &[f32], p: &Params, seed: u64) -> (Vec<Vec<u32>>, usize) {
    let n = vecs.len();
    let mut rng = Rng::new(seed);
    // init: random R-regular
    let mut graph: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut s = HashSet::new();
            while s.len() < p.r.min(n - 1) {
                let j = rng.below(n);
                if j != i { s.insert(j as u32); }
            }
            s.into_iter().collect()
        })
        .collect();
    let med = medoid(vecs, w);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() { order.swap(i, rng.below(i + 1)); } // shuffle
    let mut evals = 0usize;
    for &node in &order {
        let (_, visited, e) = greedy(&graph, vecs, w, med, &vecs[node], p.l, p.k);
        evals += e;
        let nbrs = robust_prune(node as u32, &visited, vecs, w, p.alpha, p.r);
        graph[node] = nbrs.clone();
        for q in nbrs {
            let qi = q as usize;
            if !graph[qi].contains(&(node as u32)) {
                graph[qi].push(node as u32);
                if graph[qi].len() > p.r {
                    let cand = graph[qi].clone();
                    graph[qi] = robust_prune(q, &cand, vecs, w, p.alpha, p.r);
                }
            }
        }
    }
    (graph, evals)
}

fn recall(got: &[u32], truth: &[u32]) -> f64 {
    let t: HashSet<u32> = truth.iter().copied().collect();
    got.iter().filter(|g| t.contains(g)).count() as f64 / truth.len() as f64
}

fn read_vectors(path: &str, n: usize) -> Vec<Vec32> {
    let data = std::fs::read_to_string(path).expect("read features");
    data.lines().take(n)
        .map(|l| l.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect())
        .filter(|v: &Vec32| !v.is_empty())
        .collect()
}

// ---- metric drift modelled as a vector-space transform A (row-major dim*dim) ----
// metric M = A^T A; equivalently transform vectors by A and use plain L2.
fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = (rng.f32() as f64).max(1e-9);
    let u2 = rng.f32() as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

fn random_rotation(dim: usize, rng: &mut Rng) -> Vec<f32> {
    // Gram-Schmidt on a Gaussian matrix → orthonormal rows.
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

fn identity(dim: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim { a[i * dim + i] = 1.0; }
    a
}

/// Diagonal drift target: A = diag(sqrt(scale)), scale in [0.2, 3.0].
fn target_diag(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim { a[i * dim + i] = (0.2 + 2.8 * rng.f32()).sqrt(); }
    a
}

/// Dense/rotational drift target: A = diag(sqrt(scale)) · R (anisotropic scaling
/// along rotated axes — a general Mahalanobis metric; the adversarial case).
fn target_rot(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let r = random_rotation(dim, rng);
    let mut a = vec![0.0f32; dim * dim];
    for i in 0..dim {
        let s = (0.2 + 2.8 * rng.f32()).sqrt();
        for j in 0..dim { a[i * dim + j] = s * r[i * dim + j]; }
    }
    a
}

fn lerp_mat(a0: &[f32], a1: &[f32], t: f32) -> Vec<f32> {
    a0.iter().zip(a1).map(|(x, y)| x * (1.0 - t) + y * t).collect()
}

fn apply(a: &[f32], vecs: &[Vec32], dim: usize) -> Vec<Vec32> {
    vecs.iter().map(|v| {
        (0..dim).map(|i| {
            let row = &a[i * dim..(i + 1) * dim];
            row.iter().zip(v).map(|(x, y)| x * y).sum()
        }).collect()
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-2000.csv".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let vecs = read_vectors(&path, n);
    let n = vecs.len();
    let dim = vecs[0].len();
    let p = Params { r: 24, l: 64, alpha: 1.2, k: 10 };

    // Query set: 100 sampled nodes (their own vectors as queries; self excluded).
    let mut qrng = Rng::new(999);
    let queries: Vec<usize> = (0..100.min(n)).map(|_| qrng.below(n)).collect();
    let ones = vec![1.0f32; dim];

    eprintln!("[bet1] {n} vectors x {dim} dims; Vamana R={} L={} alpha={} k={}", p.r, p.l, p.alpha, p.k);
    // Base graph built once in the ORIGINAL space (drift t=0 == identity transform).
    let t0 = Instant::now();
    let (g0, e0) = build(&vecs, &ones, &p, 7);
    let med0 = medoid(&vecs, &ones);
    eprintln!("[bet1] base graph built once in {:.2}s ({e0} dist evals)\n", t0.elapsed().as_secs_f64());

    run_mode("DIAGONAL drift (per-axis rescale)", &vecs, &g0, med0, &queries, &p, dim, target_diag(dim, &mut Rng::new(12345)));
    run_mode("ROTATIONAL drift (anisotropic scale on rotated axes — adversarial)", &vecs, &g0, med0, &queries, &p, dim, target_rot(dim, &mut Rng::new(54321)));

    println!("\nGate: WIN if A within 2% of B across the sweep; KILL if A drops >2% below B.");
    println!("A's rebuild cost is 0 (topology reused); B pays a full rebuild per drift step.");
}

#[allow(clippy::too_many_arguments)]
fn run_mode(label: &str, vecs: &[Vec32], g0: &[Vec<u32>], med0: u32, queries: &[usize], p: &Params, dim: usize, a_target: Vec<f32>) {
    let ones = vec![1.0f32; dim];
    let id = identity(dim);
    let truth0: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(vecs, &ones, q, p.k)).collect();

    println!("=== BET 1: {label} ===");
    println!("{:>5} {:>10} | {:>9} {:>9} {:>9} | {:>9} {:>8}", "t", "churn", "A reweit", "B rebuild", "C stale", "B build s", "A-B");
    println!("{}", "-".repeat(74));

    for &t in &[0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
        let at = lerp_mat(&id, &a_target, t);
        let vt = apply(&at, vecs, dim); // vectors in the drifted metric space
        let truth: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vt, &ones, q, p.k)).collect();
        let churn: f64 = truth.iter().zip(&truth0).map(|(a, b)| 1.0 - recall(a, b)).sum::<f64>() / queries.len() as f64;

        // A: reuse the original-space graph, but compute distances in the drifted space.
        let ra: f64 = queries.iter().zip(&truth).map(|(&q, tr)| {
            let (got, _, _) = greedy(g0, &vt, &ones, med0, &vt[q], p.l, p.k);
            recall(&got, tr)
        }).sum::<f64>() / queries.len() as f64;

        // B: rebuild the graph in the drifted space.
        let tb = Instant::now();
        let (gt, _) = build(&vt, &ones, p, 7);
        let bt = tb.elapsed().as_secs_f64();
        let medt = medoid(&vt, &ones);
        let rb: f64 = queries.iter().zip(&truth).map(|(&q, tr)| {
            let (got, _, _) = greedy(&gt, &vt, &ones, medt, &vt[q], p.l, p.k);
            recall(&got, tr)
        }).sum::<f64>() / queries.len() as f64;

        // C: stale — search the original graph in the ORIGINAL space, score vs drifted truth.
        let rc: f64 = queries.iter().zip(&truth).map(|(&q, tr)| {
            let (got, _, _) = greedy(g0, vecs, &ones, med0, &vecs[q], p.l, p.k);
            recall(&got, tr)
        }).sum::<f64>() / queries.len() as f64;

        println!("{:>5.2} {:>9.0}% | {:>8.1}% {:>8.1}% {:>8.1}% | {:>9.2} {:>+7.1}%", t, churn * 100.0, ra * 100.0, rb * 100.0, rc * 100.0, bt, (ra - rb) * 100.0);
    }
    println!();
}
