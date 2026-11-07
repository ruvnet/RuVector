//! In-repo Pareto frontier selection for the Darwin evolutionary harness
//! (ADR-335 Decision §6, work package WP31).
//!
//! `crates/ruvector-sota-bench/harness/src/darwin.ts` currently gets
//! `selection: "pareto"` from the external npm package
//! `@metaharness/darwin@^0.9.1` — ADR-335 records this as a known external
//! dependency and asks a follow-up to either implement an in-repo frontier
//! this program controls, or explicitly accept the dependency. This module
//! is that in-repo implementation: a `#[no_std]`-free, dependency-free,
//! formally-defined Pareto frontier over a candidate's multi-objective
//! fitness vector (e.g. quality, latency, memory, cost — the exact
//! objective list `schemas/optimization-manifest-v1.json`'s
//! `promotion.pareto.objectives` already declares per repository).
//!
//! Three selection strategies are provided, all verified against the same
//! formal ground truth ([`brute_force_frontier`]):
//!
//! - [`brute_force_frontier`] — the definition itself: a point is on the
//!   frontier iff no other point dominates it. `O(M * N^2)` worst case, but
//!   its per-point scan exits as soon as one dominator is found, so on the
//!   sparse-frontier data a random Darwin generation typically produces, it
//!   is fast in *practice* despite the worst-case bound. Measured in
//!   `benches/darwin_pareto_bench.rs`, it beats [`nsga2_frontier`] at every
//!   measured N (see that benchmark's doc comment) and stays competitive
//!   with [`kung_frontier_2d`] below a few thousand candidates. It is the
//!   oracle every other algorithm here is checked against either way.
//! - [`nsga2_frontier`] — the classic NSGA-II (Deb et al., 2002) fast
//!   non-dominated sort, restricted to returning front 0. Included for
//!   general-M correctness (`M > 2`, where [`kung_frontier_2d`] does not
//!   apply), **not** because it measured faster in this module's benchmark:
//!   it does unconditional `N*(N-1)/2` pairwise work with no early exit, so
//!   on this crate's sparse-frontier test data it was consistently and
//!   substantially *slower* than early-exit brute force at every measured
//!   N (up to ~260x slower at N=5,000) — a result worth stating plainly
//!   rather than silently dropping, per this research process's own rule
//!   against presenting an unmeasured complexity-class argument as a
//!   practical one. Keep this as the `M > 2` fallback; do not reach for it
//!   over brute force at `M = 2`.
//! - [`kung_frontier_2d`] — Kung, Luccio & Preparata's (1975) `O(N log N)`
//!   divide-and-conquer skyline algorithm, specialized here to the M=2 case
//!   (quality vs. latency, or any two-objective slice), which is the common
//!   shape of a Darwin promotion decision per `optimization-manifest-v1`'s
//!   two-objective examples. It is the only algorithm here with a measured
//!   asymptotic edge: on this module's random uniform test data the
//!   crossover where it overtakes brute force is around N≈2,000-5,000, and
//!   it continues to scale near-linearithmically out to N=100,000 where
//!   brute force and NSGA-II were not run (would be tens of seconds to
//!   minutes at that N). `kung_frontier_2d` panics on `objectives.len() !=
//!   2` rather than silently falling back, so a caller cannot accidentally
//!   use the specialized 2D algorithm outside its proven domain — see
//!   [`select_frontier`] for the dispatcher that does fall back.
//!
//! [`select_frontier`] dispatches to the specialized 2D algorithm when
//! `objectives.len() == 2` and to NSGA-II otherwise (the only option this
//! module has for `M > 2`), which is the recommended entry point for
//! callers (including a future `darwin.ts` FFI boundary, not built this
//! run — see the companion nightly research doc).
//!
//! [`is_dominated_by_frontier`] is the literal gate function
//! `optimization-manifest-v1.json`'s `promotion.pareto.require_non_dominated`
//! describes: "a candidate dominated by the current frontier cannot be
//! promoted."

/// Which way a lower/higher raw value counts as "better" for one objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Higher raw values are better (e.g. recall, throughput).
    Maximize,
    /// Lower raw values are better (e.g. latency, memory, cost).
    Minimize,
}

impl Direction {
    /// Orient a raw value so that, for every objective, "greater or equal"
    /// under this orientation always means "at least as good".
    #[inline]
    #[must_use]
    fn orient(self, raw: f64) -> f64 {
        match self {
            Direction::Maximize => raw,
            Direction::Minimize => -raw,
        }
    }
}

/// `a` Pareto-dominates `b` iff `a` is at least as good as `b` on every
/// objective and strictly better on at least one. Ties on every objective
/// mean neither dominates the other (both may sit on the frontier).
///
/// `objectives.len()` must equal `a.len()` and `b.len()`; this is a
/// programmer error, not a runtime input-validation boundary (both slices
/// come from the same internally-constructed candidate matrix), so it
/// panics via indexing rather than returning a `Result`.
#[must_use]
pub fn dominates(objectives: &[Direction], a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(objectives.len(), a.len());
    debug_assert_eq!(objectives.len(), b.len());
    let mut strictly_better_somewhere = false;
    for (i, dir) in objectives.iter().enumerate() {
        let av = dir.orient(a[i]);
        let bv = dir.orient(b[i]);
        if av < bv {
            return false;
        }
        if av > bv {
            strictly_better_somewhere = true;
        }
    }
    strictly_better_somewhere
}

/// The formal ground truth: indices of every point not dominated by any
/// other point in `points`. `O(M * N^2)`.
///
/// Returned indices are ascending and stable (first-seen order among ties),
/// which the other algorithms in this module are tested against exactly.
#[must_use]
pub fn brute_force_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize> {
    let n = points.len();
    let mut out = Vec::new();
    for i in 0..n {
        let mut dominated = false;
        for j in 0..n {
            if i != j && dominates(objectives, points[j], points[i]) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            out.push(i);
        }
    }
    out
}

/// NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002) fast non-dominated sort,
/// returning only front 0 (the Pareto frontier) rather than the full
/// front-by-front ranking — Darwin's promotion gate only needs membership in
/// the frontier, not a complete ranking of dominated candidates.
///
/// `O(M * N^2)` worst case, same as [`brute_force_frontier`], but unlike it
/// has **no early exit**: every pair is visited unconditionally. Measured in
/// `benches/darwin_pareto_bench.rs`, this makes it consistently slower than
/// early-exit brute force on sparse-frontier data (up to ~260x at N=5,000) —
/// its purpose in this module is correctness for `M > 2` objectives, where
/// [`kung_frontier_2d`] does not apply, not a practical speedup at `M = 2`.
#[must_use]
pub fn nsga2_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize> {
    let n = points.len();
    let mut domination_count = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(objectives, points[i], points[j]) {
                domination_count[j] += 1;
            } else if dominates(objectives, points[j], points[i]) {
                domination_count[i] += 1;
            }
        }
    }
    (0..n).filter(|&i| domination_count[i] == 0).collect()
}

/// Kung, Luccio & Preparata (1975) divide-and-conquer skyline/Pareto-frontier
/// algorithm, specialized to two objectives: sort by the first objective,
/// then scan and keep every point whose second objective beats the best
/// second-objective value seen so far among points at least as good on the
/// first. `O(N log N)`.
///
/// # Panics
/// Panics if `objectives.len() != 2`. This module does not silently widen
/// the 2D algorithm to more objectives — [`select_frontier`] is the
/// dispatcher that falls back to [`nsga2_frontier`] for `M != 2`.
#[must_use]
pub fn kung_frontier_2d(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize> {
    assert_eq!(
        objectives.len(),
        2,
        "kung_frontier_2d is specialized to exactly 2 objectives; use select_frontier() to dispatch, or nsga2_frontier() directly"
    );
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let x = |i: usize| objectives[0].orient(points[i][0]);
    let y = |i: usize| objectives[1].orient(points[i][1]);

    // Sort by first objective descending (best first). Within an x-tie
    // block, relative order doesn't matter: the block is resolved as a
    // whole below.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x(b).partial_cmp(&x(a)).unwrap_or(std::cmp::Ordering::Equal));

    // `running_best_y` is the best y among every block already processed —
    // all of which have *strictly* greater x than the current block (since
    // blocks are visited in descending-x order), so any such prior point
    // with y' >= current y already dominates (strictly better on x is
    // enough to satisfy "strictly better somewhere"). A block's own
    // non-dominated members are exactly those whose y equals the block's
    // max y (points with the same x but lower y are dominated by their own
    // block's maximum).
    let mut frontier = Vec::new();
    let mut running_best_y = f64::NEG_INFINITY;
    let mut i = 0;
    while i < n {
        let xi = x(order[i]);
        let mut j = i;
        while j < n && x(order[j]) == xi {
            j += 1;
        }
        let block_max_y = order[i..j]
            .iter()
            .map(|&k| y(k))
            .fold(f64::NEG_INFINITY, f64::max);
        if block_max_y > running_best_y {
            for &k in &order[i..j] {
                if y(k) == block_max_y {
                    frontier.push(k);
                }
            }
            running_best_y = block_max_y;
        }
        i = j;
    }
    frontier.sort_unstable();
    frontier
}

/// Dispatch to the specialized O(N log N) 2D algorithm when there are
/// exactly two objectives, and to NSGA-II's fast non-dominated sort
/// otherwise. This is the recommended entry point for Darwin's promotion
/// gate.
#[must_use]
pub fn select_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize> {
    if objectives.len() == 2 {
        kung_frontier_2d(objectives, points)
    } else {
        nsga2_frontier(objectives, points)
    }
}

/// NSGA-II crowding distance within one front — used to pick a
/// diversity-preserving subset when more candidates sit on the frontier
/// than Darwin's `maximum_promotions` budget allows (STEP 18/19). Points at
/// the extremes of an objective get `f64::INFINITY` so they are never
/// discarded ahead of an interior point.
///
/// `front` must index into `points`; `points[i].len() == objectives.len()`
/// for every `i` in `front`.
#[must_use]
pub fn crowding_distance(objectives: &[Direction], points: &[&[f64]], front: &[usize]) -> Vec<f64> {
    let m = front.len();
    let mut distance = vec![0.0f64; m];
    if m <= 2 {
        return vec![f64::INFINITY; m];
    }
    for (obj_idx, dir) in objectives.iter().enumerate() {
        let mut local: Vec<usize> = (0..m).collect();
        local.sort_by(|&a, &b| {
            dir.orient(points[front[a]][obj_idx])
                .partial_cmp(&dir.orient(points[front[b]][obj_idx]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        distance[local[0]] = f64::INFINITY;
        distance[local[m - 1]] = f64::INFINITY;
        let lo = dir.orient(points[front[local[0]]][obj_idx]);
        let hi = dir.orient(points[front[local[m - 1]]][obj_idx]);
        let span = hi - lo;
        if span <= 0.0 {
            continue;
        }
        for k in 1..(m - 1) {
            if !distance[local[k]].is_finite() {
                continue;
            }
            let prev = dir.orient(points[front[local[k - 1]]][obj_idx]);
            let next = dir.orient(points[front[local[k + 1]]][obj_idx]);
            distance[local[k]] += (next - prev) / span;
        }
    }
    distance
}

/// The literal `require_non_dominated` gate from
/// `schemas/optimization-manifest-v1.json`: `true` iff `candidate` is
/// dominated by at least one point already on `frontier` (or tied with one
/// on every objective — a tie is not a strict improvement, so it does not
/// earn promotion either). A candidate that is itself non-dominated by, and
/// not dominated by, the current frontier (i.e. it opens a new region of the
/// objective space) returns `false` and is eligible for promotion.
#[must_use]
pub fn is_dominated_by_frontier(
    objectives: &[Direction],
    candidate: &[f64],
    frontier: &[&[f64]],
) -> bool {
    frontier
        .iter()
        .any(|p| dominates(objectives, p, candidate) || p.iter().eq(candidate.iter()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows<'a>(data: &'a [[f64; 2]]) -> Vec<&'a [f64]> {
        data.iter().map(|r| r.as_slice()).collect()
    }

    #[test]
    fn dominates_basic_cases() {
        let obj = [Direction::Maximize, Direction::Minimize];
        // a: higher quality, lower latency than b -> a dominates b
        assert!(dominates(&obj, &[0.9, 10.0], &[0.8, 20.0]));
        assert!(!dominates(&obj, &[0.8, 20.0], &[0.9, 10.0]));
        // tie on both -> neither dominates
        assert!(!dominates(&obj, &[0.9, 10.0], &[0.9, 10.0]));
        // mixed trade-off -> neither dominates
        assert!(!dominates(&obj, &[0.9, 20.0], &[0.8, 10.0]));
    }

    #[test]
    fn empty_input_yields_empty_frontier() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let pts: Vec<&[f64]> = Vec::new();
        assert_eq!(brute_force_frontier(&obj, &pts), Vec::<usize>::new());
        assert_eq!(nsga2_frontier(&obj, &pts), Vec::<usize>::new());
        assert_eq!(kung_frontier_2d(&obj, &pts), Vec::<usize>::new());
    }

    #[test]
    fn single_point_is_its_own_frontier() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let data = [[0.5, 5.0]];
        let pts = rows(&data);
        assert_eq!(brute_force_frontier(&obj, &pts), vec![0]);
        assert_eq!(nsga2_frontier(&obj, &pts), vec![0]);
        assert_eq!(kung_frontier_2d(&obj, &pts), vec![0]);
    }

    #[test]
    fn total_antichain_keeps_every_point() {
        // Every point strictly better on exactly one objective and worse on
        // the other -> all are mutually non-dominated.
        let obj = [Direction::Maximize, Direction::Minimize];
        let data = [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0]];
        let pts = rows(&data);
        let expect = vec![0, 1, 2, 3];
        assert_eq!(brute_force_frontier(&obj, &pts), expect);
        assert_eq!(nsga2_frontier(&obj, &pts), expect);
        assert_eq!(kung_frontier_2d(&obj, &pts), expect);
    }

    #[test]
    fn single_dominator_collapses_frontier_to_one_point() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let data = [[1.0, 1.0], [0.5, 5.0], [0.2, 9.0], [0.9, 1.5]];
        let pts = rows(&data);
        // point 0 dominates every other point
        assert_eq!(brute_force_frontier(&obj, &pts), vec![0]);
        assert_eq!(nsga2_frontier(&obj, &pts), vec![0]);
        assert_eq!(kung_frontier_2d(&obj, &pts), vec![0]);
    }

    #[test]
    fn duplicate_points_all_survive_as_a_tied_frontier() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let data = [[0.5, 5.0], [0.5, 5.0], [0.9, 1.0]];
        let pts = rows(&data);
        // the two duplicates tie each other (neither dominates), but both
        // are dominated by point 2.
        assert_eq!(brute_force_frontier(&obj, &pts), vec![2]);
        assert_eq!(nsga2_frontier(&obj, &pts), vec![2]);
        assert_eq!(kung_frontier_2d(&obj, &pts), vec![2]);
    }

    #[test]
    fn is_dominated_by_frontier_gate() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let f1: [f64; 2] = [0.9, 10.0];
        let f2: [f64; 2] = [0.7, 5.0];
        let frontier: Vec<&[f64]> = vec![&f1, &f2];
        // dominated by f1
        assert!(is_dominated_by_frontier(&obj, &[0.8, 15.0], &frontier));
        // exact tie with f2 -> not an improvement, gate blocks it
        assert!(is_dominated_by_frontier(&obj, &[0.7, 5.0], &frontier));
        // opens a new trade-off (higher quality AND higher latency than f2,
        // lower quality AND lower latency than f1) -> not dominated
        assert!(!is_dominated_by_frontier(&obj, &[0.8, 7.0], &frontier));
    }

    #[test]
    fn crowding_distance_extremes_are_infinite() {
        let obj = [Direction::Maximize, Direction::Minimize];
        let data = [[0.1, 9.0], [0.5, 5.0], [0.9, 1.0]];
        let pts = rows(&data);
        let front = [0usize, 1, 2];
        let d = crowding_distance(&obj, &pts, &front);
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[2], f64::INFINITY);
        assert!(d[1].is_finite() && d[1] > 0.0);
    }

    // --- Cross-algorithm agreement across many random instances ---------
    // This is the module's real correctness evidence: nsga2_frontier and
    // kung_frontier_2d must exactly reproduce brute_force_frontier's
    // formal ground truth on every generated instance, not just the
    // hand-picked cases above.

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_points(seed: u64, n: usize, m: usize, max_val: u32) -> Vec<Vec<f64>> {
        let mut s = seed.wrapping_mul(2_685_821_657_736_338_717).max(1);
        (0..n)
            .map(|_| {
                (0..m)
                    .map(|_| (xorshift(&mut s) % u64::from(max_val)) as f64)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn random_2d_instances_agree_with_brute_force() {
        let obj = [Direction::Maximize, Direction::Minimize];
        for seed in 0..200u64 {
            // small max_val on purpose: forces frequent ties, the case most
            // likely to expose an off-by-one in the tie-block scan.
            let n = 1 + (seed % 40) as usize;
            let max_val = 1 + (seed % 6) as u32;
            let data = random_points(seed, n, 2, max_val);
            let pts: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
            let truth = brute_force_frontier(&obj, &pts);
            assert_eq!(
                nsga2_frontier(&obj, &pts),
                truth,
                "nsga2 disagreement at seed={seed} n={n} max_val={max_val}"
            );
            assert_eq!(
                kung_frontier_2d(&obj, &pts),
                truth,
                "kung disagreement at seed={seed} n={n} max_val={max_val}"
            );
            assert_eq!(select_frontier(&obj, &pts), truth);
        }
    }

    #[test]
    fn random_3d_and_4d_instances_agree_with_brute_force() {
        for (m, dirs) in [
            (
                3usize,
                vec![
                    Direction::Maximize,
                    Direction::Minimize,
                    Direction::Maximize,
                ],
            ),
            (
                4usize,
                vec![
                    Direction::Maximize,
                    Direction::Minimize,
                    Direction::Minimize,
                    Direction::Maximize,
                ],
            ),
        ] {
            for seed in 0..100u64 {
                let n = 1 + (seed % 30) as usize;
                let max_val = 1 + (seed % 5) as u32;
                let data = random_points(seed ^ (m as u64), n, m, max_val);
                let pts: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
                let truth = brute_force_frontier(&dirs, &pts);
                assert_eq!(
                    nsga2_frontier(&dirs, &pts),
                    truth,
                    "nsga2 disagreement at m={m} seed={seed} n={n} max_val={max_val}"
                );
                assert_eq!(select_frontier(&dirs, &pts), truth);
            }
        }
    }

    #[test]
    #[should_panic(expected = "exactly 2 objectives")]
    fn kung_frontier_2d_panics_on_wrong_arity() {
        let obj = [
            Direction::Maximize,
            Direction::Minimize,
            Direction::Maximize,
        ];
        let data = [[0.1, 0.2, 0.3]];
        let pts = rows_3(&data);
        let _ = kung_frontier_2d(&obj, &pts);
    }

    fn rows_3<'a>(data: &'a [[f64; 3]]) -> Vec<&'a [f64]> {
        data.iter().map(|r| r.as_slice()).collect()
    }
}
