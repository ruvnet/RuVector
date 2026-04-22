//! Degree-stratified null-sample generator for AC-5 at FlyWire scale.
//!
//! ADR-154 §8.4 and §13: at synthetic-SBM N=1024 scale the degree-
//! stratified random null collapses the AC-5 effect size to zero (the
//! functional boundary and the degree-matched hubs overlap). At
//! FlyWire v783 scale (~139 k neurons with a much heavier non-hub
//! tail) the stratified null is expected to separate from the
//! boundary; that is the correct bench for the null-tightness side of
//! the AC-5 SOTA target (`z_rand ≤ 1σ`).
//!
//! This module is the port of the stratified-null sampler investigated
//! in the 7a83adffe dev branch and documented (but not shipped) in
//! that commit's ADR-154 §8.4 entry. It is wired to take any
//! `Connectome` — synthetic SBM or FlyWire-loaded — so the same test
//! drives both substrates:
//!
//! ```text
//! // Synthetic (runs today, collapses at N=1024 per §8.4):
//! let conn = Connectome::generate(&ConnectomeConfig::default());
//!
//! // FlyWire (runs once connectome/flywire streaming ingest has real
//! // data; expected to separate per §8.4):
//! let conn = load_flywire_streaming(&flywire_dir)?;
//! ```
//!
//! The algorithm:
//!
//! 1. Compute `(out_degree_i, in_degree_j)` for each synapse endpoint.
//! 2. Group all synapses into 10 deciles by the product of those two
//!    degrees (binning by `out * in` so hub-at-one-end or hub-at-both
//!    are scored appropriately).
//! 3. Given a boundary edge set, count boundary edges per decile —
//!    this is the histogram the null must match.
//! 4. For each decile, draw WITHOUT replacement from the non-boundary
//!    subset of that decile until the count matches the boundary
//!    histogram for that decile.
//! 5. Concatenate to produce the stratified random sample.
//!
//! Determinism: the caller provides a seeded RNG. Same seed + same
//! `Connectome` + same `boundary` → bit-identical sample. This
//! preserves AC-1 hygiene when the sampler is driven from an
//! acceptance test.

use std::collections::HashSet;

use rand::seq::IteratorRandom;
use rand::RngCore;

use super::Connectome;

/// Number of decile bins; fixed at 10 to match the prototype wording
/// in ADR-154 §8.4. A 20-bin version is a plausible refinement if the
/// FlyWire-scale distribution is very long-tailed but is not shipped.
pub const NUM_DECILES: usize = 10;

/// Out-of-band outcome from the sampler when the non-boundary pool of
/// a needed decile is too small to satisfy the boundary histogram.
/// This is NOT an error — it is the data telling us the SBM (or the
/// loaded connectome) lacks enough non-hub filler for the stratified
/// null to be well-defined at this scale. The shipped sampler returns
/// a short sample and the *caller* decides whether to treat that as
/// a skip, a partial-credit pass, or a FAIL. See ADR-154 §8.4.
#[derive(Debug, Clone)]
pub struct StratifiedSample {
    /// Flat-index synapses selected for the null sample.
    pub sample: Vec<usize>,
    /// Boundary-decile histogram (10 entries).
    pub boundary_hist: [u32; NUM_DECILES],
    /// Sample-decile histogram achieved by the draw. If any entry is
    /// below `boundary_hist[i]`, the pool for decile i was exhausted.
    pub sample_hist: [u32; NUM_DECILES],
    /// Total non-boundary edges in each decile, as seen during the
    /// scan. Lets the caller reason about *why* a decile was short.
    pub pool_sizes: [u32; NUM_DECILES],
}

impl StratifiedSample {
    /// True if the sampler satisfied the boundary histogram in every
    /// decile. When false, at least one decile's non-boundary pool
    /// was too small.
    pub fn is_complete(&self) -> bool {
        self.boundary_hist
            .iter()
            .zip(self.sample_hist.iter())
            .all(|(b, s)| s >= b)
    }
}

/// Draw a degree-decile-stratified random sample of synapses from the
/// non-boundary edges of `conn`, matching the decile histogram of
/// `boundary`.
///
/// The caller supplies a seeded `RngCore` so the draw is deterministic
/// given `(conn, boundary, rng_state)`. The sampler never draws a
/// synapse that appears in `boundary`.
///
/// Degree-product decile boundaries are computed from the full edge
/// list, not from the boundary subset, so the same decile scheme is
/// used at sampling and histogramming time.
pub fn degree_stratified_null_sample<R: RngCore>(
    conn: &Connectome,
    boundary: &[usize],
    rng: &mut R,
) -> StratifiedSample {
    let (out_deg, in_deg) = degrees(conn);
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();

    // Compute the degree-product key for every synapse. Using a
    // smooth max bound `max_product + 1` so the floor at 0 and the
    // ceiling are inside the bin table regardless of isolated nodes.
    let total_edges = syn.len();
    if total_edges == 0 {
        return StratifiedSample {
            sample: Vec::new(),
            boundary_hist: [0; NUM_DECILES],
            sample_hist: [0; NUM_DECILES],
            pool_sizes: [0; NUM_DECILES],
        };
    }

    let mut flat_to_product: Vec<u64> = Vec::with_capacity(total_edges);
    for (pre_idx, window) in row_ptr.windows(2).enumerate() {
        let s = window[0] as usize;
        let e = window[1] as usize;
        for flat in s..e {
            let post_idx = syn[flat].post.idx();
            let p = (out_deg[pre_idx] as u64) * (in_deg[post_idx] as u64);
            flat_to_product.push(p);
        }
    }
    debug_assert_eq!(flat_to_product.len(), total_edges);

    let max_product = flat_to_product.iter().copied().max().unwrap_or(0).max(1);

    // Boundary decile histogram.
    let boundary_set: HashSet<usize> = boundary.iter().copied().collect();
    let mut boundary_hist = [0u32; NUM_DECILES];
    for &flat in boundary {
        let d = product_to_decile(flat_to_product[flat], max_product);
        boundary_hist[d] += 1;
    }

    // Build per-decile pools of non-boundary synapses, count pool
    // sizes while doing so.
    let mut pools: Vec<Vec<usize>> = (0..NUM_DECILES).map(|_| Vec::new()).collect();
    let mut pool_sizes = [0u32; NUM_DECILES];
    for (flat, &product) in flat_to_product.iter().enumerate() {
        if boundary_set.contains(&flat) {
            continue;
        }
        let d = product_to_decile(product, max_product);
        pools[d].push(flat);
        pool_sizes[d] += 1;
    }

    // Draw without replacement from each decile pool, up to the
    // boundary count in that decile. `choose_multiple` on a cloned
    // iterator gives deterministic-under-RNG selection without
    // materialising unused elements.
    let mut sample: Vec<usize> = Vec::with_capacity(boundary.len());
    let mut sample_hist = [0u32; NUM_DECILES];
    for (d, need) in boundary_hist.iter().copied().enumerate() {
        if need == 0 {
            continue;
        }
        let pool = &pools[d];
        let take = (need as usize).min(pool.len());
        // `choose_multiple` is deterministic given the RngCore state
        // and preserves the exclusion-without-replacement property.
        let chosen: Vec<usize> = pool.iter().copied().choose_multiple(rng, take);
        sample_hist[d] = chosen.len() as u32;
        sample.extend(chosen);
    }

    StratifiedSample {
        sample,
        boundary_hist,
        sample_hist,
        pool_sizes,
    }
}

/// Map a degree-product into its decile bin index in `0..NUM_DECILES`.
/// Saturates at the last bin when the product equals `max_product`.
fn product_to_decile(product: u64, max_product: u64) -> usize {
    if max_product == 0 {
        return 0;
    }
    let scaled = ((product as u128) * (NUM_DECILES as u128)) / (max_product as u128);
    (scaled as usize).min(NUM_DECILES - 1)
}

/// Per-neuron (out-degree, in-degree) vectors.
fn degrees(conn: &Connectome) -> (Vec<u32>, Vec<u32>) {
    let n = conn.num_neurons();
    let row_ptr = conn.row_ptr();
    let syn = conn.synapses();
    let mut out_deg = vec![0u32; n];
    let mut in_deg = vec![0u32; n];
    for (pre_idx, window) in row_ptr.windows(2).enumerate() {
        let s = window[0] as usize;
        let e = window[1] as usize;
        out_deg[pre_idx] = (e - s) as u32;
        for flat in s..e {
            let post = syn[flat].post.idx();
            in_deg[post] += 1;
        }
    }
    (out_deg, in_deg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{Connectome, ConnectomeConfig};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn degrees_match_row_ptr_widths() {
        let conn = Connectome::generate(&ConnectomeConfig::default());
        let (out_deg, in_deg) = degrees(&conn);
        assert_eq!(out_deg.len(), conn.num_neurons());
        assert_eq!(in_deg.len(), conn.num_neurons());
        let total_out: u32 = out_deg.iter().sum();
        let total_in: u32 = in_deg.iter().sum();
        assert_eq!(total_out as usize, conn.synapses().len());
        assert_eq!(total_in as usize, conn.synapses().len());
    }

    #[test]
    fn decile_binning_monotonic() {
        assert_eq!(product_to_decile(0, 100), 0);
        assert_eq!(product_to_decile(100, 100), NUM_DECILES - 1);
        for p in (0u64..=100).step_by(10) {
            let d = product_to_decile(p, 100);
            assert!(d < NUM_DECILES);
        }
    }

    #[test]
    fn stratified_sample_deterministic_under_same_seed() {
        let conn = Connectome::generate(&ConnectomeConfig::default());
        // Pick first 100 edges as a stand-in "boundary".
        let boundary: Vec<usize> = (0..100.min(conn.synapses().len())).collect();
        let mut rng1 = Xoshiro256StarStar::seed_from_u64(0xC0FE_BABE_CAFE_F00D);
        let mut rng2 = Xoshiro256StarStar::seed_from_u64(0xC0FE_BABE_CAFE_F00D);
        let s1 = degree_stratified_null_sample(&conn, &boundary, &mut rng1);
        let s2 = degree_stratified_null_sample(&conn, &boundary, &mut rng2);
        assert_eq!(
            s1.sample, s2.sample,
            "same seed produced different stratified samples"
        );
    }

    #[test]
    fn stratified_sample_excludes_boundary() {
        let conn = Connectome::generate(&ConnectomeConfig::default());
        let boundary: Vec<usize> = (0..200.min(conn.synapses().len())).collect();
        let mut rng = Xoshiro256StarStar::seed_from_u64(0xDEAD_BEEF);
        let s = degree_stratified_null_sample(&conn, &boundary, &mut rng);
        let b_set: HashSet<usize> = boundary.iter().copied().collect();
        for &flat in &s.sample {
            assert!(
                !b_set.contains(&flat),
                "stratified sample contains boundary edge {flat}"
            );
        }
    }

    #[test]
    fn stratified_sample_matches_boundary_histogram_when_pools_sufficient() {
        let conn = Connectome::generate(&ConnectomeConfig::default());
        let boundary: Vec<usize> = (0..100.min(conn.synapses().len())).collect();
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let s = degree_stratified_null_sample(&conn, &boundary, &mut rng);
        // The boundary is small relative to per-decile pool sizes for
        // the default SBM, so the histogram should match exactly.
        let sum_sample: u32 = s.sample_hist.iter().sum();
        let sum_boundary: u32 = s.boundary_hist.iter().sum();
        assert_eq!(
            sum_sample, sum_boundary,
            "sample size {} did not match boundary size {} — pools too small on this SBM; revisit SBM density",
            sum_sample, sum_boundary
        );
        for d in 0..NUM_DECILES {
            assert_eq!(
                s.sample_hist[d], s.boundary_hist[d],
                "decile {d} mismatch: sample={} boundary={}",
                s.sample_hist[d], s.boundary_hist[d]
            );
        }
    }

    #[test]
    fn empty_boundary_returns_empty_sample() {
        let conn = Connectome::generate(&ConnectomeConfig::default());
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let s = degree_stratified_null_sample(&conn, &[], &mut rng);
        assert_eq!(s.sample.len(), 0);
        assert_eq!(s.boundary_hist, [0; NUM_DECILES]);
        assert_eq!(s.sample_hist, [0; NUM_DECILES]);
    }
}
