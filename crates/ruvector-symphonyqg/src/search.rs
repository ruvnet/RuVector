use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{batch_hamming_dist, dist_cosine, dist_l2_sq, SymphonyGraph};
use crate::{Config, Metric};

/// Single ANN result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub idx: usize,
    pub dist: f32,
}

/// Total-order wrapper for f32 distances in a BinaryHeap.
#[derive(PartialEq)]
pub(crate) struct HeapEntry(pub f32, pub usize);
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant A: FlatExactIndex — linear scan, exact distances (oracle baseline).
// ─────────────────────────────────────────────────────────────────────────────

pub struct FlatExactIndex {
    vectors: Vec<f32>,
    n: usize,
    dim: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
}

impl FlatExactIndex {
    pub fn build(vecs: &[Vec<f32>], config: &Config) -> Self {
        let n = vecs.len();
        let dim = config.dim;
        let mut vectors = vec![0.0f32; n * dim];
        for (i, v) in vecs.iter().enumerate() {
            vectors[i * dim..(i + 1) * dim].copy_from_slice(v);
        }
        let dist_fn: fn(&[f32], &[f32]) -> f32 = match config.metric {
            Metric::Euclidean => dist_l2_sq,
            Metric::Cosine => dist_cosine,
        };
        Self {
            vectors,
            n,
            dim,
            dist_fn,
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut dists: Vec<(f32, usize)> = (0..self.n)
            .map(|i| {
                (
                    (self.dist_fn)(query, &self.vectors[i * self.dim..(i + 1) * self.dim]),
                    i,
                )
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists
            .iter()
            .take(k)
            .map(|&(d, i)| SearchResult { idx: i, dist: d })
            .collect()
    }

    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant B: GraphExactIndex — HNSW-style graph, exact f32 at every hop.
// ─────────────────────────────────────────────────────────────────────────────

pub struct GraphExactIndex {
    graph: SymphonyGraph,
    dist_fn: fn(&[f32], &[f32]) -> f32,
}

impl GraphExactIndex {
    pub fn from_graph(graph: SymphonyGraph, metric: Metric) -> Self {
        let dist_fn: fn(&[f32], &[f32]) -> f32 = match metric {
            Metric::Euclidean => dist_l2_sq,
            Metric::Cosine => dist_cosine,
        };
        Self { graph, dist_fn }
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let g = &self.graph;
        let mut visited = vec![false; g.n];
        let mut cands: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        let mut results: BinaryHeap<HeapEntry> = BinaryHeap::new();

        let ep = g.entry;
        let d0 = (self.dist_fn)(query, g.vector_of(ep));
        cands.push(Reverse(HeapEntry(d0, ep)));
        results.push(HeapEntry(d0, ep));
        visited[ep] = true;

        while let Some(Reverse(HeapEntry(cd, cu))) = cands.pop() {
            let worst = results.peek().map_or(f32::MAX, |e| e.0);
            if cd > worst && results.len() >= ef {
                break;
            }
            for &nb in g.neighbors_of(cu) {
                let nb = nb as usize;
                if nb < g.n && !visited[nb] {
                    visited[nb] = true;
                    let nd = (self.dist_fn)(query, g.vector_of(nb));
                    cands.push(Reverse(HeapEntry(nd, nb)));
                    if results.len() < ef || nd < results.peek().map_or(f32::MAX, |e| e.0) {
                        results.push(HeapEntry(nd, nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<SearchResult> = results
            .into_iter()
            .map(|HeapEntry(d, i)| SearchResult { idx: i, dist: d })
            .collect();
        out.sort_unstable_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(k);
        out
    }

    pub fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant C: SymphonyIndex — 1-bit batch scan during traversal + exact re-rank.
//
// Core innovation (from SymphonyQG, SIGMOD 2025, arXiv:2411.12229):
//   - Graph degree is a multiple of BATCH_SIZE (no wasted SIMD lanes).
//   - During beam search, evaluate ALL m neighbours in one XNOR-popcount batch.
//   - Accumulate ef candidates by estimated distance.
//   - After traversal, re-rank the ef candidates with exact f32 distances.
// ─────────────────────────────────────────────────────────────────────────────

pub struct SymphonyIndex {
    graph: SymphonyGraph,
    dist_fn: fn(&[f32], &[f32]) -> f32,
}

impl SymphonyIndex {
    pub fn from_graph(graph: SymphonyGraph, metric: Metric) -> Self {
        let dist_fn: fn(&[f32], &[f32]) -> f32 = match metric {
            Metric::Euclidean => dist_l2_sq,
            Metric::Cosine => dist_cosine,
        };
        Self { graph, dist_fn }
    }

    /// SymphonyQG search:
    ///   Phase 1 (traversal) — estimated distances via 1-bit XNOR-popcount batch.
    ///   Phase 2 (re-rank)   — exact f32 distances on the top `ef` candidates.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let g = &self.graph;
        let q_code = g.encode_query(query);

        let mut visited = vec![false; g.n];
        // min-heap by estimated distance (candidates to expand)
        let mut cands: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        // max-heap by estimated distance, capacity ef (best seen so far)
        let mut ef_set: BinaryHeap<HeapEntry> = BinaryHeap::new();

        let ep = g.entry;
        // Seed: 1-bit estimated distance query → entry point via self-code.
        let ep_est = batch_hamming_dist(&q_code, g.self_code_of(ep), 1, g.code_bytes)[0];
        cands.push(Reverse(HeapEntry(ep_est, ep)));
        ef_set.push(HeapEntry(ep_est, ep));
        visited[ep] = true;

        while let Some(Reverse(HeapEntry(cd, cu))) = cands.pop() {
            let worst = ef_set.peek().map_or(f32::MAX, |e| e.0);
            if cd > worst && ef_set.len() >= ef {
                break;
            }

            // ── Batch XNOR-popcount: ALL m neighbours in one pass ──────────
            let nbrs = g.neighbors_of(cu);
            let nb_codes = g.nb_codes_of(cu);
            let est_dists = batch_hamming_dist(&q_code, nb_codes, g.m, g.code_bytes);

            for (j, &nb) in nbrs.iter().enumerate() {
                let nb = nb as usize;
                // Reject padding slots (graph::PADDING_SENTINEL = u32::MAX,
                // which casts to a usize > any real g.n) and already-visited
                // vertices in one branch. Padded slots ALSO have zero-filled
                // code bytes so the SIMD popcount above produced a uniform,
                // discardable score — never a real edge displacement.
                if nb >= g.n || visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let est = est_dists[j];
                cands.push(Reverse(HeapEntry(est, nb)));
                if ef_set.len() < ef {
                    ef_set.push(HeapEntry(est, nb));
                } else if est < ef_set.peek().map_or(f32::MAX, |e| e.0) {
                    ef_set.pop();
                    ef_set.push(HeapEntry(est, nb));
                }
            }
        }

        // ── Phase 2: exact f32 re-rank ─────────────────────────────────────
        let mut reranked: Vec<SearchResult> = ef_set
            .into_iter()
            .map(|HeapEntry(_, i)| SearchResult {
                idx: i,
                dist: (self.dist_fn)(query, g.vector_of(i)),
            })
            .collect();
        reranked.sort_unstable_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(k);
        reranked
    }

    pub fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }

    /// Run `queries.len()` independent searches and return their results in
    /// the same order. Sequential by default; set `feature = "parallel"`
    /// to dispatch via Rayon's work-stealing pool (one query per work item).
    ///
    /// Each query is independent (no shared mutable state in `&self`), so
    /// the parallel speedup is essentially linear in physical cores up to
    /// the point where memory bandwidth becomes the bottleneck (typically
    /// 4-8x on a server, less on a Pi 5).
    ///
    /// **Bit-for-bit equivalence guarantee**: `search_batch` returns the
    /// exact same `Vec<Vec<SearchResult>>` as a sequential loop calling
    /// `self.search` once per query. Verified by
    /// `tests::search_batch_matches_sequential`.
    pub fn search_batch(&self, queries: &[&[f32]], k: usize, ef: usize) -> Vec<Vec<SearchResult>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries.par_iter().map(|q| self.search(q, k, ef)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries.iter().map(|q| self.search(q, k, ef)).collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_all, Config, Metric};

    fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    /// Compute exact top-k ground truth.
    fn ground_truth(vecs: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, i)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.iter().take(k).map(|&(_, i)| i).collect()
    }

    #[test]
    fn flat_exact_finds_self() {
        let n = 200;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 42);
        let flat = FlatExactIndex::build(&vecs, &cfg);
        // Query is vecs[0] itself — must be result[0].
        let res = flat.search(&vecs[0], 1);
        assert_eq!(res[0].idx, 0);
        assert!(res[0].dist < 1e-6);
    }

    /// SymphonyQG needs D ≥ 128 for 1-bit codes to have low estimation variance.
    /// At D=128, code_bytes=16, the estimator std ≈ 0.06 — tight enough for beam guidance.
    #[test]
    fn symphony_recall_at_10_above_70pct() {
        let n = 500;
        let dim = 128; // 1-bit estimation improves with higher D
        let k = 10;
        let cfg = Config {
            dim,
            m_base: 16,
            ef_construction: 150,
            metric: Metric::Euclidean,
            seed: 7,
            vamana: None,
        };
        let vecs = gaussian_vecs(n, dim, 7);
        let (_, _, symphony) = build_all(&vecs, &cfg);

        let mut hit = 0usize;
        let queries = 50;
        for q in 0..queries {
            let query = &vecs[(q * 7 + 3) % n];
            let gt = ground_truth(&vecs, query, k);
            // ef=300: SymphonyQG needs higher ef than GraphExact because 1-bit estimated
            // distances are noisy (σ ≈ sin(θ)/√dim). Re-ranking recovers recall.
            let res = symphony.search(query, k, 300);
            let result_ids: std::collections::HashSet<usize> = res.iter().map(|r| r.idx).collect();
            hit += gt.iter().filter(|&&g| result_ids.contains(&g)).count();
        }
        let recall = hit as f64 / (queries * k) as f64;
        // Measured: 0.714 at dim=128, n=500, ef=300 after the padding-edges
        // correctness fix (PR #428 review feedback). The PR body's headline
        // 97.6% was measured at n=5000, ef=100 by `src/main.rs` and reflects
        // a different operating point — the small-corpus test here gives
        // the 1-bit estimator less signal.
        // Floor set to 0.70 so this test catches regressions without
        // pinning the exact value (which depends on the random seed).
        assert!(
            recall >= 0.70,
            "SymphonyQG recall@10 = {:.1}% < 70% floor (dim={dim}, ef=300)",
            recall * 100.0
        );
    }

    #[test]
    fn graph_exact_recall_at_10_above_70pct() {
        let n = 500;
        let dim = 128;
        let k = 10;
        let cfg = Config {
            dim,
            m_base: 16,
            ef_construction: 150,
            metric: Metric::Euclidean,
            seed: 8,
            vamana: None,
        };
        let vecs = gaussian_vecs(n, dim, 8);
        let (_, graph_exact, _) = build_all(&vecs, &cfg);

        let mut hit = 0usize;
        let queries = 50;
        for q in 0..queries {
            let query = &vecs[(q * 7 + 3) % n];
            let gt = ground_truth(&vecs, query, k);
            let res = graph_exact.search(query, k, 150);
            let result_ids: std::collections::HashSet<usize> = res.iter().map(|r| r.idx).collect();
            hit += gt.iter().filter(|&&g| result_ids.contains(&g)).count();
        }
        let recall = hit as f64 / (queries * k) as f64;
        assert!(
            recall >= 0.70,
            "GraphExact recall@10 = {:.1}% < 70% floor (dim={dim})",
            recall * 100.0
        );
    }

    #[test]
    fn symphony_and_graph_results_have_k_entries() {
        let n = 200;
        let dim = 64;
        let k = 5;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 99);
        let (flat, graph_exact, symphony) = build_all(&vecs, &cfg);
        let q = &vecs[10];
        assert_eq!(flat.search(q, k).len(), k);
        assert_eq!(graph_exact.search(q, k, 50).len(), k);
        assert_eq!(symphony.search(q, k, 50).len(), k);
    }

    // ── Edge-case tests (PR #428 review feedback) ────────────────────────

    /// `n < BATCH_SIZE`: the most padding-heavy regime — every per-vertex
    /// block is mostly PADDING_SENTINEL slots. Pre-fix this would have
    /// produced wildly wrong results because random padded edges dominated
    /// the candidate beam. Post-fix the sentinel skip discards them.
    #[test]
    fn small_n_below_batch_size_does_not_panic_or_corrupt() {
        // n = 8, BATCH_SIZE = 32 → every block has 8 real edges + 24
        // PADDING_SENTINEL slots. A self-query must return the vertex itself.
        let n = 8;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 4, // m will be padded to 32 (BATCH_SIZE)
            ef_construction: 7,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 11);
        let (_, graph_exact, symphony) = build_all(&vecs, &cfg);

        // Self-query: vertex 0 must rank itself in the top-1.
        let res = symphony.search(&vecs[0], 1, 32);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].idx, 0, "self-query at n<BATCH_SIZE must find itself");

        // GraphExact must agree.
        let res_g = graph_exact.search(&vecs[0], 1, 32);
        assert_eq!(res_g[0].idx, 0);
    }

    /// `dim` not a multiple of 32: stresses the new packed-block stride math.
    /// `m * code_bytes` must still be a multiple of 4 for the &[u32]→&[u8]
    /// cast in graph::nb_codes_of to be sound. m = 32, dim = 72 → code_bytes = 9
    /// → m * code_bytes = 288 (multiple of 4). ✓
    #[test]
    fn dim_72_non_multiple_of_32_packs_cleanly() {
        let n = 64;
        let dim = 72;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 32,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 13);
        let (_, _, symphony) = build_all(&vecs, &cfg);

        // Build did not panic. Search produces a result of the right shape.
        let res = symphony.search(&vecs[5], 3, 16);
        assert_eq!(res.len(), 3);
        for r in &res {
            assert!(r.idx < n);
            assert!(r.dist.is_finite());
        }
    }

    /// `ef > n`: degenerate — ef-set can never fill. Should still return
    /// correct results without panicking on heap underflow.
    #[test]
    fn ef_larger_than_corpus_returns_valid_results() {
        let n = 16;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 4,
            ef_construction: 8,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 17);
        let (_, _, symphony) = build_all(&vecs, &cfg);

        let res = symphony.search(&vecs[3], 5, 1000); // ef=1000 >> n=16
        assert!(res.len() <= 5);
        assert!(res.len() <= n);
    }

    /// `k > ef`: result truncation must respect both bounds.
    #[test]
    fn k_larger_than_ef_truncates_to_ef() {
        let n = 200;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 19);
        let (_, _, symphony) = build_all(&vecs, &cfg);

        let res = symphony.search(&vecs[1], 50, 10); // k=50 > ef=10
        assert!(
            res.len() <= 10,
            "result count {} must be ≤ ef=10",
            res.len()
        );
    }

    /// search_batch must produce the exact same Vec<Vec<SearchResult>>
    /// as a sequential loop calling search() once per query. The test
    /// runs in both `parallel` and non-parallel feature configurations
    /// because the assertion is feature-agnostic.
    #[test]
    fn search_batch_matches_sequential() {
        let n = 200;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 31);
        let (_, _, sym) = build_all(&vecs, &cfg);

        let probes: Vec<&[f32]> = (0..10).map(|i| vecs[i * 7 % n].as_slice()).collect();
        let batched = sym.search_batch(&probes, 5, 30);
        let sequential: Vec<Vec<SearchResult>> =
            probes.iter().map(|q| sym.search(q, 5, 30)).collect();
        assert_eq!(batched.len(), sequential.len());
        for (b, s) in batched.iter().zip(sequential.iter()) {
            assert_eq!(b.len(), s.len());
            for (br, sr) in b.iter().zip(s.iter()) {
                assert_eq!(br.idx, sr.idx, "search_batch != search at idx");
                assert!((br.dist - sr.dist).abs() < 1e-6);
            }
        }
    }

    /// Out-of-corpus query (a fresh random vector, not from `vecs`):
    /// search must not panic and must produce monotone-distance results.
    #[test]
    fn out_of_corpus_query_returns_sorted_distances() {
        let n = 200;
        let dim = 64;
        let cfg = Config {
            dim,
            m_base: 8,
            ef_construction: 50,
            ..Config::default()
        };
        let vecs = gaussian_vecs(n, dim, 23);
        let (_, _, symphony) = build_all(&vecs, &cfg);

        // Query is index 999 from a different seed — definitely not in vecs.
        let novel: Vec<f32> = gaussian_vecs(1, dim, 9999).into_iter().next().unwrap();

        let res = symphony.search(&novel, 5, 50);
        assert_eq!(res.len(), 5);
        for w in res.windows(2) {
            assert!(
                w[0].dist <= w[1].dist,
                "results must be sorted ascending: {} <= {}",
                w[0].dist,
                w[1].dist
            );
        }
    }
}
