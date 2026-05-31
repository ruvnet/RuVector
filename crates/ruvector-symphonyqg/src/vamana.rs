//! Vamana α-pruning refinement for SymphonyQG graphs.
//!
//! After the initial sampled-greedy graph build (`build::build`), graph
//! quality is mediocre at large `n` because each vertex's neighbours are
//! all locally close — there are no long-range "highway" edges. The
//! beam search ends up wandering the local cluster instead of jumping
//! across regions, which destroys recall.
//!
//! Vamana (DiskANN, NeurIPS 2019, [arXiv:1907.06146](https://arxiv.org/abs/1907.06146))
//! addresses this with **α-pruning**: when picking M neighbours from a
//! pool of candidates, after each pick the algorithm removes any
//! remaining candidate that's "shadowed" by the pick — specifically, any
//! candidate `v` where `α · dist(p*, v) ≤ dist(p, v)` (i.e. the just-picked
//! `p*` is α-times closer to `v` than `p` is). This forces the kept
//! neighbours to span different regions of space.
//!
//! Per the SymphonyQG paper §4.2 this complements (rather than replaces)
//! their inline-codes layout — the storage format is unchanged; only the
//! adjacency selection rule is upgraded.
//!
//! Empirical impact (PR #428 iter-5 on cognitum-v0 x86_64):
//!   n=50K, ef=100 — recall@10 climbs from 31.3% (sampled-greedy) to ≥50%.
//!   Headline n=5K, ef=100 — minimal change (graph was already near-optimal).

use crate::build;
use crate::graph::{batch_pad, dist_l2_sq, encode, SymphonyGraph};
use crate::search::{GraphExactIndex, SearchResult};
use crate::{Config, Metric};

/// Tuning knobs for Vamana refinement. Defaults follow the DiskANN paper.
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// α factor for the pruning rule. 1.0 = greedy (no diversity), 1.2 = DiskANN
    /// recommendation (mild diversity), 1.5 = aggressive diversity. Above 2.0
    /// the rule becomes degenerate (effectively keeps everything).
    pub alpha: f32,
    /// Number of refinement passes. 1 is usually enough; 2 squeezes a few
    /// percentage points more recall. Linear cost in `n × ef × m`.
    pub passes: usize,
    /// Beam width during refinement search. Larger = better candidates but
    /// slower refinement build. Use `Config::ef_construction` if unsure.
    pub beam_ef: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            alpha: 1.2,
            passes: 1,
            beam_ef: 200,
        }
    }
}

/// Robust α-prune: from `candidates` (each `(idx, dist_to_p)`, may be unsorted),
/// select up to `m` indices that are **diverse** in the α-pruning sense.
///
/// Algorithm (DiskANN §3.3):
/// ```text
/// while pool != ∅ and |kept| < m:
///   p* = argmin_{v ∈ pool} dist(p, v)
///   kept.push(p*); pool.remove(p*)
///   for v in pool:
///     if α · dist(p*, v) ≤ dist(p, v):
///       pool.remove(v)
/// ```
///
/// Excludes `p_idx` from the candidate set (a vertex can't be its own neighbour).
/// Cost: O(|candidates| · m) distance evaluations against `vectors`.
pub fn robust_prune(
    p_idx: usize,
    p_vec: &[f32],
    mut candidates: Vec<(usize, f32)>,
    vectors: &[f32],
    dim: usize,
    m: usize,
    alpha: f32,
) -> Vec<usize> {
    candidates.retain(|&(idx, _)| idx != p_idx);
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<usize> = Vec::with_capacity(m);

    while !candidates.is_empty() && kept.len() < m {
        // p* = closest remaining candidate (front of sorted vec).
        let (p_star, _d_star) = candidates.remove(0);
        kept.push(p_star);

        let p_star_vec = &vectors[p_star * dim..(p_star + 1) * dim];

        // Drop every remaining candidate v that p_star α-dominates:
        // α · dist(p_star, v) ≤ dist(p, v).
        candidates.retain(|&(v_idx, d_pv)| {
            let v_vec = &vectors[v_idx * dim..(v_idx + 1) * dim];
            let d_star_v = dist_l2_sq(p_star_vec, v_vec);
            // dist values here are squared L2 — alpha applies to the comparison
            // between two such distances; semantically equivalent for ranking.
            alpha * d_star_v > d_pv
        });
    }

    kept
}

/// One refinement pass: forward α-prune for every vertex, then back-edge
/// propagation (DiskANN §3.3 lines 12-13). Without back-edges, α-pruning
/// destroys the cross-cluster bridging edges that beam search needs for
/// out-of-corpus queries on clustered data — verified empirically at
/// PR #428 iter-7. Adding back-edges restores the property that
/// `(p, p*) is an edge ⇒ (p*, p) is also an edge`, after which the
/// graph stays connected enough for navigability while still benefiting
/// from the diversity property of α-pruning.
fn refine_pass(graph: &SymphonyGraph, cfg: &Config, vcfg: &VamanaConfig) -> Vec<Vec<usize>> {
    let n = graph.n;
    let dim = graph.dim;
    let m = batch_pad(cfg.m_base);

    // Beam search uses exact f32 distances over the existing graph topology.
    let searcher = GraphExactIndex::from_graph(graph.clone(), cfg.metric);

    // ── 1. Forward α-prune ────────────────────────────────────────────────
    let mut adj: Vec<Vec<usize>> = Vec::with_capacity(n);
    for v in 0..n {
        let v_vec = graph.vector_of(v);
        let res: Vec<SearchResult> = searcher.search(v_vec, vcfg.beam_ef, vcfg.beam_ef);
        let candidates: Vec<(usize, f32)> = res.into_iter().map(|r| (r.idx, r.dist)).collect();
        adj.push(robust_prune(
            v,
            v_vec,
            candidates,
            &graph.vectors,
            dim,
            m,
            vcfg.alpha,
        ));
    }

    // ── 2. Back-edge collection ───────────────────────────────────────────
    // For every forward edge (p → p*), record p as a back-edge candidate at
    // p*. After this pass, any vertex whose neighbour list now exceeds `m`
    // gets re-pruned in step 3.
    let mut back: Vec<Vec<usize>> = vec![Vec::new(); n];
    for p in 0..n {
        for &p_star in &adj[p] {
            back[p_star].push(p);
        }
    }

    // ── 3. Merge + re-prune overflow ──────────────────────────────────────
    for v in 0..n {
        if back[v].is_empty() {
            continue;
        }
        // Combine existing forward neighbours with back-edge proposals,
        // dropping duplicates. Use a small inline dedup since lists are O(m).
        let mut combined: Vec<usize> = adj[v].clone();
        for &b in &back[v] {
            if b != v && !combined.contains(&b) {
                combined.push(b);
            }
        }

        if combined.len() <= m {
            adj[v] = combined;
            continue;
        }

        // Overflow → re-prune. Compute exact distances from v to each
        // combined candidate and run robust_prune again to select the
        // most diverse `m`.
        let v_vec = graph.vector_of(v);
        let cands: Vec<(usize, f32)> = combined
            .iter()
            .map(|&u| {
                let u_vec = &graph.vectors[u * dim..(u + 1) * dim];
                (u, dist_l2_sq(v_vec, u_vec))
            })
            .collect();
        adj[v] = robust_prune(v, v_vec, cands, &graph.vectors, dim, m, vcfg.alpha);
    }

    adj
}

/// Pack new adjacency lists into a refined SymphonyGraph, reusing the
/// existing graph's vectors / signs / perm / self_codes (those don't change
/// between refinement passes).
fn pack(graph: &SymphonyGraph, cfg: &Config, new_adj: Vec<Vec<usize>>) -> SymphonyGraph {
    let n = graph.n;
    let dim = graph.dim;
    let code_bytes = graph.code_bytes;
    let m = batch_pad(cfg.m_base);

    // Re-encode all codes from scratch — cheap (one pass over n vectors).
    let all_codes: Vec<Vec<u8>> = (0..n)
        .map(|v| encode(graph.vector_of(v), &graph.signs, &graph.perm))
        .collect();

    let stride_u32 = SymphonyGraph::block_stride_u32_for(m, code_bytes);
    let mut blocks = vec![0u32; n * stride_u32];
    for v in 0..n {
        let off = v * stride_u32;
        for slot in &mut blocks[off..off + m] {
            *slot = crate::graph::PADDING_SENTINEL;
        }
    }

    for v in 0..n {
        let off = v * stride_u32;
        let codes_u32_len = (m * code_bytes) / 4;
        for (j, &nb) in new_adj[v].iter().enumerate() {
            blocks[off + j] = nb as u32;
        }
        let codes_u32 = &mut blocks[off + m..off + m + codes_u32_len];
        // SAFETY: same as build::build — Vec<u32> is 4-byte aligned, m * code_bytes
        // is a multiple of 4 (m is multiple of BATCH_SIZE=32).
        let codes_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(codes_u32.as_mut_ptr() as *mut u8, codes_u32.len() * 4)
        };
        for (j, &nb) in new_adj[v].iter().enumerate() {
            let dst = j * code_bytes;
            codes_bytes[dst..dst + code_bytes].copy_from_slice(&all_codes[nb]);
        }
    }

    SymphonyGraph {
        n,
        dim,
        code_bytes,
        m,
        block_stride_u32: stride_u32,
        vectors: graph.vectors.clone(),
        blocks,
        self_codes: graph.self_codes.clone(),
        signs: graph.signs.clone(),
        perm: graph.perm.clone(),
        entry: graph.entry,
    }
}

/// Find the medoid — the vertex closest to the corpus centroid. Beam
/// search starting from the medoid converges faster than from vertex 0
/// because the medoid is, on average, the shortest hop count from any
/// query in space.
fn find_medoid(vectors: &[f32], n: usize, dim: usize) -> usize {
    let mut centroid = vec![0.0f32; dim];
    for v in 0..n {
        for d in 0..dim {
            centroid[d] += vectors[v * dim + d];
        }
    }
    for d in 0..dim {
        centroid[d] /= n as f32;
    }
    let mut best_v = 0usize;
    let mut best_d = f32::INFINITY;
    for v in 0..n {
        let d = dist_l2_sq(&centroid, &vectors[v * dim..(v + 1) * dim]);
        if d < best_d {
            best_d = d;
            best_v = v;
        }
    }
    best_v
}

/// Refine an existing SymphonyGraph with `passes` rounds of α-pruning,
/// then promote the **medoid** as the search entry point.
///
/// **Known limitation (PR #428 iter-7)**: this implementation skips the
/// back-edge propagation step from canonical Vamana (DiskANN §3.3 lines
/// 12-13). Without back-edges, refinement on **clustered** data tends
/// to overprune the cross-cluster bridging edges that beam search needs
/// for out-of-corpus queries; recall can regress vs the unrefined graph.
/// The medoid entry point partly mitigates this by starting from the
/// graph's geometric centre, but back-edge propagation is still TODO.
///
/// Empirically:
/// - **Pure Gaussian data, n=50K**: out-of-corpus recall@10 climbs from
///   18.0% to 35.6% (helps).
/// - **Clustered Gaussian data, n=50K** (the realistic case): recall
///   regresses without back-edges. Use only after manually verifying
///   on representative queries until iter-N adds back-edges.
///
/// Returns a fresh graph; the input is consumed but its allocations are
/// reused for the output's vectors/signs/perm/self_codes.
pub fn refine(graph: SymphonyGraph, cfg: &Config, vcfg: &VamanaConfig) -> SymphonyGraph {
    let _ = build::build; // keep build module reference for paired import
    let mut g = graph;
    for _ in 0..vcfg.passes {
        let new_adj = refine_pass(&g, cfg, vcfg);
        g = pack(&g, cfg, new_adj);
    }
    // Promote medoid as the search entry point — far better starting
    // location than the (essentially arbitrary) vertex 0.
    g.entry = find_medoid(&g.vectors, g.n, g.dim);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_all;

    fn unit_vec(theta: f32) -> Vec<f32> {
        // Deterministic 4-D probe vector along a polar angle θ in dim-0/1 plane.
        vec![theta.cos(), theta.sin(), 0.0, 0.0]
    }

    #[test]
    fn robust_prune_keeps_diverse_neighbours() {
        // Vertex p at origin. Candidates form three clusters along three axes.
        // With alpha=1.2 and m=3, robust_prune should keep one per cluster
        // (diverse), not three from the closest cluster (greedy).
        let dim = 4;
        let p_vec = vec![0.0; 4];
        // Lay out candidates: 3 close to +x axis, 3 close to +y, 3 close to +z.
        // Indices 0..9. Distances chosen so cluster A is closest.
        let mut vectors = vec![0.0f32; 10 * dim];
        let mut put = |i: usize, v: [f32; 4]| {
            for (k, x) in v.iter().enumerate() {
                vectors[i * dim + k] = *x;
            }
        };
        put(0, [1.0, 0.01, 0.0, 0.0]);
        put(1, [1.05, -0.02, 0.0, 0.0]);
        put(2, [1.1, 0.03, 0.0, 0.0]);
        put(3, [0.01, 1.0, 0.0, 0.0]);
        put(4, [-0.02, 1.05, 0.0, 0.0]);
        put(5, [0.03, 1.1, 0.0, 0.0]);
        put(6, [0.0, 0.01, 1.0, 0.0]);
        put(7, [0.0, -0.02, 1.05, 0.0]);
        put(8, [0.0, 0.03, 1.1, 0.0]);
        // p itself at index 9 (will be excluded).
        put(9, [0.0, 0.0, 0.0, 0.0]);

        let candidates: Vec<(usize, f32)> = (0..10)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                (i, dist_l2_sq(&p_vec, v))
            })
            .collect();

        let kept = robust_prune(9, &p_vec, candidates, &vectors, dim, 3, 1.2);
        assert_eq!(kept.len(), 3);

        // Each kept neighbour must be from a different cluster.
        let cluster_of = |idx: usize| match idx {
            0..=2 => 0,
            3..=5 => 1,
            6..=8 => 2,
            _ => panic!("p excluded"),
        };
        let mut seen_clusters = std::collections::HashSet::new();
        for &k in &kept {
            seen_clusters.insert(cluster_of(k));
        }
        assert_eq!(
            seen_clusters.len(),
            3,
            "expected 3 distinct clusters, got {:?} (kept = {:?})",
            seen_clusters,
            kept
        );
    }

    #[test]
    fn robust_prune_alpha_governs_diversity_aggressiveness() {
        // DiskANN α-rule: drop any candidate v that the just-picked p* is at
        // least `1/α` times as close to as p is. With colinear points along
        // the +x ray from origin, the closest one (1,0) "shadows" the others:
        // - α=1.0: strict — keeps ONLY (1,0), drops (2,0) and (3,0).
        // - α=10.0: relaxed — keeps all three because p* is not 10× closer to v.
        // This test pins the algorithm's intended sensitivity to α.
        let dim = 2;
        let p_vec = vec![0.0; 2];
        let vectors = vec![
            1.0, 0.0, // 0: (1,0) — squared dist to p = 1
            2.0, 0.0, // 1: (2,0) — squared dist to p = 4
            3.0, 0.0, // 2: (3,0) — squared dist to p = 9
        ];
        let candidates: Vec<(usize, f32)> = vec![(0, 1.0), (1, 4.0), (2, 9.0)];

        let kept_a1 = robust_prune(99, &p_vec, candidates.clone(), &vectors, dim, 3, 1.0);
        assert_eq!(
            kept_a1.len(),
            1,
            "α=1.0 with colinear ray must drop shadowed points; got {:?}",
            kept_a1
        );
        assert_eq!(kept_a1[0], 0);

        let kept_a10 = robust_prune(99, &p_vec, candidates, &vectors, dim, 3, 10.0);
        assert_eq!(
            kept_a10.len(),
            3,
            "α=10.0 should keep all 3 colinear points; got {:?}",
            kept_a10
        );
    }

    /// Integration test for Config::vamana = Some(...) — validates the
    /// end-to-end build_all path actually runs refinement and produces
    /// a measurable recall improvement at the operating point where
    /// sampled-greedy is known to underperform.
    ///
    /// Smaller scale than the n=50K example bench (3000 vecs vs 50000)
    /// so the test runs in seconds, not minutes; recall delta is
    /// proportionally smaller but still > 5pp at n=3000.
    #[test]
    fn config_vamana_integration_improves_recall() {
        let n = 3000;
        let dim = 128;
        let k = 10;
        let queries = 30;
        let ef = 50;

        // Same vectors for both runs — only the Config::vamana flag changes.
        let mut s = 0xc0ffee_u32;
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        (s as f32 / u32::MAX as f32) - 0.5
                    })
                    .collect()
            })
            .collect();

        let measure = |vamana: Option<VamanaConfig>| -> f64 {
            let cfg = Config {
                dim,
                m_base: 16,
                ef_construction: 100,
                vamana,
                ..Config::default()
            };
            let (_, _, sym) = build_all(&vecs, &cfg);
            let mut hit = 0;
            for q in 0..queries {
                let query = &vecs[(q * 7 + 3) % n];
                let res = sym.search(query, k, ef);
                let mut gt: Vec<(f32, usize)> = vecs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (dist_l2_sq(query, v), i))
                    .collect();
                gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let gt_set: std::collections::HashSet<usize> =
                    gt.iter().take(k).map(|&(_, i)| i).collect();
                hit += res.iter().filter(|r| gt_set.contains(&r.idx)).count();
            }
            hit as f64 / (queries * k) as f64
        };

        let r_off = measure(None);
        let r_on = measure(Some(VamanaConfig::default()));
        eprintln!(
            "config_vamana_integration: recall without={:.3} with={:.3} (Δ={:+.3})",
            r_off,
            r_on,
            r_on - r_off
        );
        assert!(
            r_on > r_off + 0.05,
            "Vamana must improve recall by > 5pp at n={n}: got {:.3} → {:.3} (Δ={:+.3})",
            r_off,
            r_on,
            r_on - r_off
        );
    }

    #[test]
    fn refine_preserves_or_improves_recall() {
        // Sanity check: one pass of Vamana refinement at n=300, dim=128
        // should NOT make recall worse on a small benchmark. (Improvement
        // shows up at larger n; this test guards against regression.)
        let n = 300;
        let dim = 128;
        let cfg = Config {
            dim,
            m_base: 16,
            ef_construction: 100,
            ..Config::default()
        };
        let mut rng_state = 0x12345678u32;
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 17;
                        rng_state ^= rng_state << 5;
                        (rng_state as f32 / u32::MAX as f32) - 0.5
                    })
                    .collect()
            })
            .collect();

        let (_, _, sym_before) = build_all(&vecs, &cfg);
        // Refine in place via a fresh build path.
        let g_before = crate::build::build(&vecs, &cfg);
        let g_after = refine(
            g_before,
            &cfg,
            &VamanaConfig {
                alpha: 1.2,
                passes: 1,
                beam_ef: 100,
            },
        );
        let sym_after = crate::search::SymphonyIndex::from_graph(g_after, Metric::Euclidean);

        // Measure recall before/after on a few queries.
        let measure = |idx: &crate::search::SymphonyIndex| -> f64 {
            let mut hit = 0;
            for q in 0..20 {
                let query = &vecs[(q * 7 + 3) % n];
                let res = idx.search(query, 10, 50);
                let mut gt: Vec<(f32, usize)> = vecs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (dist_l2_sq(query, v), i))
                    .collect();
                gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let gt_set: std::collections::HashSet<usize> =
                    gt.iter().take(10).map(|&(_, i)| i).collect();
                let res_set: std::collections::HashSet<usize> = res.iter().map(|r| r.idx).collect();
                hit += gt_set.intersection(&res_set).count();
            }
            hit as f64 / (20.0 * 10.0)
        };

        let r_before = measure(&sym_before);
        let r_after = measure(&sym_after);
        eprintln!(
            "recall before refine: {:.3}, after: {:.3}",
            r_before, r_after
        );
        assert!(
            r_after >= r_before - 0.05,
            "refine regressed recall by > 5pp: {:.3} → {:.3}",
            r_before,
            r_after
        );
    }
}
