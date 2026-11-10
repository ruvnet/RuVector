//! Filter-aware layer-0 traversal, after ACORN (Patel et al., "ACORN: Performant and Predicate-Agnostic Search Over
//! Vector Embeddings and Structured Data", SIGMOD 2024, arXiv:2403.04871; reference implementation in
//! `crates/ruvector-acorn`).
//!
//! Post-filtering an unfiltered top-`ef` returns fewer than `k` hits whenever fewer than `k` of the `ef` nearest nodes
//! pass. Here the traversal is predicate-agnostic instead: every node the beam admits is expanded whether or not it
//! passes, so failing nodes still serve as bridges, but only passing nodes enter the result heap, and the admission and
//! stop thresholds are taken from that heap. The search therefore keeps walking until it holds `ef` passing nodes
//! (or the frontier empties). Because failing nodes are expanded directly, ACORN-γ's 2-hop neighbour lists add nothing
//! on top of this graph.
//!
//! At low selectivity `s` that walk visits about `ef / s` nodes, so it is capped by a distance-computation budget; the
//! caller is expected to send small match sets to an exact scan instead (see `VectorDB`'s filter planner).

use super::{Cand, FlatGraph};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Result of [`FlatGraph::search_filtered`].
#[derive(Debug, Default)]
pub struct FilteredHits {
    /// `(external id, distance)` of passing nodes, ascending by distance; at most `k`.
    pub hits: Vec<(usize, f32)>,
    /// Distance computations spent on layer 0 (the upper-layer descent is not counted).
    pub distance_computations: usize,
    /// True when the walk stopped because the budget ran out rather than because it converged.
    pub budget_exhausted: bool,
}

impl FlatGraph {
    /// The `k` nearest nodes whose external id passes `pred`, with a beam of `ef` passing nodes and at most `budget`
    /// layer-0 distance computations. `pred` is evaluated at most once per node per query, and only for nodes close
    /// enough to enter the beam.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        budget: usize,
        pred: impl Fn(usize) -> bool,
    ) -> FilteredHits {
        let Some(entry) = self.entry else {
            return FilteredHits::default();
        };
        if k == 0 || budget == 0 {
            return FilteredHits::default();
        }
        let ef = ef.max(k).min(self.len());
        let q = self.prepare(query);
        let mut cur = entry;
        let mut cur_d = self.dist(&q, self.vec(cur));
        for l in (1..=self.max_level).rev() {
            (cur, cur_d) = self.greedy(&q, cur, cur_d, l);
        }

        let mut vis = self.take_visited();
        let (top_cap, cand_cap) = self.heap_caps(ef);
        let mut top: BinaryHeap<Cand> = BinaryHeap::with_capacity(top_cap);
        let mut cand: BinaryHeap<Reverse<Cand>> = BinaryHeap::with_capacity(cand_cap);
        // Distance of the worst passing node held; +inf until one is found.
        let mut lower = f32::INFINITY;
        let accept = |c: Cand, top: &mut BinaryHeap<Cand>, lower: &mut f32| {
            if pred(self.ext[c.id as usize]) {
                top.push(c);
                if top.len() > ef {
                    top.pop();
                }
                if let Some(t) = top.peek() {
                    *lower = t.dist;
                }
            }
        };

        vis.visit(cur);
        let start = Cand {
            dist: cur_d,
            id: cur,
        };
        cand.push(Reverse(start));
        accept(start, &mut top, &mut lower);
        let mut spent = 0usize;
        let mut budget_exhausted = false;
        'walk: while let Some(Reverse(c)) = cand.pop() {
            if c.dist > lower && top.len() >= ef {
                break;
            }
            let nbrs = self.list(c.id, 0);
            for &nb in nbrs {
                super::prefetch(self.data.as_ptr().wrapping_add(nb as usize * self.dim));
            }
            for &nb in nbrs {
                if !vis.visit(nb) {
                    continue;
                }
                if spent >= budget {
                    budget_exhausted = true;
                    break 'walk;
                }
                spent += 1;
                let d = self.dist(&q, self.vec(nb));
                if top.len() < ef || d < lower {
                    let n = Cand { dist: d, id: nb };
                    // Predicate-agnostic: the node is expanded later whether or not it passes.
                    cand.push(Reverse(n));
                    accept(n, &mut top, &mut lower);
                }
            }
        }
        self.give_visited(vis);
        while top.len() > k {
            top.pop();
        }
        let mut out = top.into_vec();
        out.sort_unstable();
        FilteredHits {
            hits: out
                .into_iter()
                .map(|c| (self.ext[c.id as usize], c.dist))
                .collect(),
            distance_computations: spent,
            budget_exhausted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;

    fn data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX >> 1) as f32
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn brute(base: &[Vec<f32>], q: &[f32], k: usize, pass: impl Fn(usize) -> bool) -> Vec<usize> {
        let mut d: Vec<(f32, usize)> = (0..base.len())
            .filter(|&i| pass(i))
            .map(|i| (l2(q, &base[i]), i))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        d.into_iter().take(k).map(|x| x.1).collect()
    }

    fn graph(base: &[Vec<f32>]) -> FlatGraph {
        let mut g = FlatGraph::new(base[0].len(), DistanceMetric::Euclidean, 16, 200);
        let items: Vec<(&[f32], usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_slice(), i))
            .collect();
        g.insert_batch(&items);
        g
    }

    #[test]
    fn filtered_recall_matches_brute_force_where_post_filtering_collapses() {
        let (n, dim, k, ef) = (8000, 16, 10, 64);
        let base = data(n, dim, 21);
        let queries = data(50, dim, 22);
        let g = graph(&base);
        // 2% selectivity: ~1 hit expected in an unfiltered top-64.
        let pass = |i: usize| i % 50 == 7;
        let (mut hits, mut post_hits, mut full) = (0, 0, 0);
        for q in &queries {
            let truth = brute(&base, q, k, pass);
            let r = g.search_filtered(q, k, ef, n, pass);
            assert!(r.hits.iter().all(|h| pass(h.0)), "only passing ids");
            assert!(r.hits.windows(2).all(|w| w[0].1 <= w[1].1), "sorted");
            if r.hits.len() == k {
                full += 1;
            }
            hits += r.hits.iter().filter(|h| truth.contains(&h.0)).count();
            post_hits += g
                .search(q, ef, ef)
                .into_iter()
                .filter(|h| pass(h.0))
                .take(k)
                .filter(|h| truth.contains(&h.0))
                .count();
        }
        let recall = hits as f64 / (queries.len() * k) as f64;
        let post = post_hits as f64 / (queries.len() * k) as f64;
        assert_eq!(full, queries.len(), "every query returns k hits");
        assert!(recall > 0.9, "filtered recall {recall}");
        assert!(post < 0.3, "post-filter baseline should collapse: {post}");
    }

    #[test]
    fn k_above_the_match_count_returns_every_match_and_empty_matches_return_nothing() {
        let base = data(3000, 8, 5);
        let g = graph(&base);
        let pass = |i: usize| i % 500 == 3; // 6 matches
        let r = g.search_filtered(&base[0], 50, 64, base.len(), pass);
        let mut got: Vec<usize> = r.hits.iter().map(|h| h.0).collect();
        got.sort_unstable();
        assert_eq!(got, vec![3, 503, 1003, 1503, 2003, 2503]);
        assert!(!r.budget_exhausted);
        assert!(g
            .search_filtered(&base[0], 10, 64, base.len(), |_| false)
            .hits
            .is_empty());
    }

    #[test]
    fn budget_bounds_the_walk_at_low_selectivity() {
        let base = data(6000, 8, 9);
        let g = graph(&base);
        let r = g.search_filtered(&base[0], 10, 64, 500, |i| i % 1000 == 1);
        assert!(r.distance_computations <= 500);
        assert!(r.budget_exhausted);
        let r = g.search_filtered(&base[0], 10, 64, 0, |_| true);
        assert!(r.hits.is_empty());
    }

    #[test]
    fn huge_k_and_ef_do_not_allocate_or_overflow() {
        let base = data(200, 8, 3);
        let g = graph(&base);
        let r = g.search_filtered(&base[1], usize::MAX, usize::MAX, usize::MAX, |i| i % 2 == 0);
        assert_eq!(r.hits.len(), 100);
        assert_eq!(g.search(&base[1], usize::MAX, usize::MAX).len(), 200);
        let empty = FlatGraph::new(8, DistanceMetric::Euclidean, 16, 200);
        assert!(empty
            .search_filtered(&base[1], usize::MAX, usize::MAX, usize::MAX, |_| true)
            .hits
            .is_empty());
    }
}
