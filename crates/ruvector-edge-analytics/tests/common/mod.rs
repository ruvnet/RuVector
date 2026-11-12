//! Deterministic graph generators shared by native and wasm32 tests. No
//! `rand`: a fixed splitmix64 keeps every target on the same edge list.
#![allow(dead_code)]

use std::collections::HashSet;

/// splitmix64.
pub struct Mix(pub u64);

impl Mix {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Sparse external ids, so varints and the id space are exercised.
pub fn vid(i: u64) -> u64 {
    1_000_003 + i * 7_919
}

fn push(
    out: &mut Vec<(u64, u64, f64)>,
    seen: &mut HashSet<(u64, u64)>,
    a: u64,
    b: u64,
    w: f64,
) -> bool {
    if a == b {
        return false;
    }
    let key = if a < b { (a, b) } else { (b, a) };
    if !seen.insert(key) {
        return false;
    }
    out.push((vid(a), vid(b), w));
    true
}

/// Connected sparse graph: random spanning tree plus random extra edges,
/// integer weights 1..=4 (exact in f64 in any summation order).
pub fn sparse_graph(n: u64, m: usize, seed: u64) -> Vec<(u64, u64, f64)> {
    let mut r = Mix(seed);
    let mut out = Vec::with_capacity(m);
    let mut seen = HashSet::with_capacity(m);
    for i in 1..n {
        let p = r.below(i);
        let w = (r.below(4) + 1) as f64;
        push(&mut out, &mut seen, i, p, w);
    }
    while out.len() < m {
        let (a, b) = (r.below(n), r.below(n));
        let w = (r.below(4) + 1) as f64;
        push(&mut out, &mut seen, a, b, w);
    }
    out
}

/// Two dense clusters of `k` vertices joined by `cross` unit edges; intra
/// weights 2..=5. No bridges and minimum degree far above 2 x lightest, so
/// the solver's certificate fails and full Stoer-Wagner runs.
pub fn two_clusters(k: u64, m: usize, cross: usize, seed: u64) -> Vec<(u64, u64, f64)> {
    let mut r = Mix(seed);
    let mut out = Vec::with_capacity(m);
    let mut seen = HashSet::with_capacity(m);
    for c in 0..2u64 {
        let base = c * k;
        // Ring first so each cluster is 2-edge-connected.
        for i in 0..k {
            let w = (r.below(4) + 2) as f64;
            push(&mut out, &mut seen, base + i, base + (i + 1) % k, w);
        }
    }
    let mut added = 0;
    while added < cross {
        let (a, b) = (r.below(k), k + r.below(k));
        if push(&mut out, &mut seen, a, b, 1.0) {
            added += 1;
        }
    }
    while out.len() < m {
        let base = r.below(2) * k;
        let (a, b) = (base + r.below(k), base + r.below(k));
        let w = (r.below(4) + 2) as f64;
        push(&mut out, &mut seen, a, b, w);
    }
    out
}

/// Sparse ring plus chords, unit weights: every vertex has degree >= 3 and
/// there are no bridges, so the certificate fails (sparse Stoer-Wagner).
pub fn ring_chords(n: u64, m: usize, seed: u64) -> Vec<(u64, u64, f64)> {
    let mut r = Mix(seed);
    let mut out = Vec::with_capacity(m);
    let mut seen = HashSet::with_capacity(m);
    for i in 0..n {
        push(&mut out, &mut seen, i, (i + 1) % n, 1.0);
    }
    for i in 0..n {
        while !push(&mut out, &mut seen, i, r.below(n), 1.0) {}
    }
    while out.len() < m {
        push(&mut out, &mut seen, r.below(n), r.below(n), 1.0);
    }
    out
}

/// `(value, (S, T), cut edges)` as the native crate reports them.
pub type NativeCut = (f64, (Vec<u64>, Vec<u64>), Vec<(u64, u64, f64)>);

/// What `ruvector-mincut` returns when called directly, the way a native
/// user would: default `MinCutBuilder` config, edges in generation order.
pub fn native_exact(edges: &[(u64, u64, f64)]) -> NativeCut {
    let mc = ruvector_mincut::MinCutBuilder::new()
        .exact()
        .with_edges(edges.to_vec())
        .build()
        .expect("native build");
    let mut cut: Vec<(u64, u64, f64)> = mc
        .cut_edges()
        .into_iter()
        .map(|e| {
            let (u, v) = e.canonical_endpoints();
            (u, v, e.weight)
        })
        .collect();
    cut.sort_by_key(|e| (e.0, e.1));
    (mc.min_cut_value(), mc.partition(), cut)
}

/// Digests of the two 50k-edge answers (`CutReport::digest_hex`), computed
/// natively in `tests/equivalence.rs` and asserted again on wasm32 in
/// `tests/wasm.rs`: the cross-target "equals native" evidence.
pub const SPARSE_50K_DIGEST: &str =
    "ef547f9bb36ce701bdde91854d0ca32e0e2bea337a37ed525c504b339dcf3d7c";
/// See [`SPARSE_50K_DIGEST`].
pub const CLUSTERS_50K_DIGEST: &str =
    "5b328099c131d156e74196b079e6834c5624ddac8a88b7a0ca5a271b626841b4";
/// Generator parameters for the two graphs, shared with wasm.
pub fn sparse_50k() -> Vec<(u64, u64, f64)> {
    sparse_graph(10_000, 50_000, 0x5eed)
}
/// See [`sparse_50k`].
pub fn clusters_50k() -> Vec<(u64, u64, f64)> {
    two_clusters(250, 50_000, 5, 0xc1)
}
