//! M0 — predicate families, the ρ-correlation knob, and selectivity targeting.
//!
//! A [`Predicate`] is a boolean membership mask over node ids `[0, n)`. The harness
//! passes it to ACORN / the oracle as `|id| pred.test(id)`.
//!
//! ## The ρ-knob (the controlled instrument)
//!
//! [`correlated`] builds a predicate of an *exact* target selectivity whose correlation
//! with embedding geometry is tunable: ρ=1 is a tight, structurally-clustered set (built
//! from subject-label classes, which occupy regions of the embedding space); ρ=0 is a
//! random set of the same size (ACORN's home turf, the kill control); intermediate ρ
//! replaces a fraction `1−ρ` of structured members with random non-members. Selectivity is
//! held fixed across ρ so cost differences are attributable to correlation, not set size.

use rand::seq::SliceRandom;
use rand::Rng;

/// A predicate over node ids, plus the construction parameters that produced it.
#[derive(Clone)]
pub struct Predicate {
    mask: Vec<bool>,
    /// Number of matching nodes (`mask` trues).
    pub n_match: usize,
    /// Requested selectivity (matches/n); see [`Predicate::selectivity`] for the realized value.
    pub target_sel: f64,
    /// Construction correlation knob in `[0,1]` (1 = structured, 0 = random). `NaN` for
    /// natural-family predicates where ρ is not a construction parameter.
    pub rho: f64,
}

impl Predicate {
    #[inline]
    pub fn test(&self, id: u32) -> bool {
        self.mask[id as usize]
    }

    /// `Fn(u32) -> bool` view for ACORN / oracle APIs.
    pub fn as_fn(&self) -> impl Fn(u32) -> bool + Copy + '_ {
        move |id| self.mask[id as usize]
    }

    /// Realized selectivity = matches / n.
    pub fn selectivity(&self) -> f64 {
        self.n_match as f64 / self.mask.len() as f64
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    fn from_mask(mask: Vec<bool>, target_sel: f64, rho: f64) -> Predicate {
        let n_match = mask.iter().filter(|&&b| b).count();
        Predicate { mask, n_match, target_sel, rho }
    }
}

/// Natural categorical predicate: nodes whose subject label equals `class`.
pub fn from_label(labels: &[u32], class: u32) -> Predicate {
    let mask = labels.iter().map(|&l| l == class).collect::<Vec<_>>();
    let sel = mask.iter().filter(|&&b| b).count() as f64 / labels.len() as f64;
    Predicate::from_mask(mask, sel, f64::NAN)
}

/// Natural ordinal predicate: nodes with `year >= y`.
pub fn year_ge(years: &[i32], y: i32) -> Predicate {
    let mask = years.iter().map(|&yr| yr >= y).collect::<Vec<_>>();
    let sel = mask.iter().filter(|&&b| b).count() as f64 / years.len() as f64;
    Predicate::from_mask(mask, sel, f64::NAN)
}

/// The controlled instrument: a predicate of exact selectivity `target_sel` with tunable
/// geometric correlation `rho ∈ [0,1]`.
///
/// - `seed_class_rank` selects which size-ranked label class seeds the structured set
///   (0 = largest); rotating it lets M3 average over several regions to remove
///   region-specific bias.
/// - The structured pool is the union of label classes (in size order from the seed),
///   truncated to exactly `m = round(target_sel · n)` members. `keep = round(rho · m)` of
///   those are retained; the remaining `m − keep` are random non-members, so |set| = m for
///   every ρ.
pub fn correlated(
    labels: &[u32],
    target_sel: f64,
    rho: f64,
    seed_class_rank: usize,
    rng: &mut impl Rng,
) -> Predicate {
    let n = labels.len();
    let m = ((target_sel * n as f64).round() as usize).clamp(1, n);
    let rho = rho.clamp(0.0, 1.0);

    // Label classes sorted by descending size; rotate by seed_class_rank.
    let n_classes = (labels.iter().copied().max().unwrap_or(0) as usize) + 1;
    let mut counts = vec![0usize; n_classes];
    for &l in labels {
        counts[l as usize] += 1;
    }
    let mut class_order: Vec<u32> = (0..n_classes as u32).collect();
    class_order.sort_by_key(|&c| std::cmp::Reverse(counts[c as usize]));
    if !class_order.is_empty() {
        let rot = seed_class_rank % class_order.len();
        class_order.rotate_left(rot);
    }

    // Accumulate node ids class-by-class until the pool reaches m, then truncate.
    let mut structured: Vec<u32> = Vec::with_capacity(m);
    'fill: for &c in &class_order {
        for (id, &l) in labels.iter().enumerate() {
            if l == c {
                structured.push(id as u32);
                if structured.len() >= m {
                    break 'fill;
                }
            }
        }
    }

    let keep = ((rho * m as f64).round() as usize).min(structured.len());
    let mut mask = vec![false; n];
    for &id in &structured[..keep] {
        mask[id as usize] = true;
    }

    // Fill the rest with random non-members so realized selectivity == m/n exactly.
    let need = m - keep;
    if need > 0 {
        let mut pool: Vec<u32> = (0..n as u32).filter(|&id| !mask[id as usize]).collect();
        let (picked, _) = pool.partial_shuffle(rng, need);
        for &id in picked.iter() {
            mask[id as usize] = true;
        }
    }

    Predicate::from_mask(mask, target_sel, rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn synth_labels(n: usize, n_classes: u32) -> Vec<u32> {
        // Deterministic block labels: class regions are contiguous id ranges (a proxy for
        // geometric clustering, sufficient to test the ρ mechanism).
        (0..n).map(|i| (i as u32 * n_classes / n as u32).min(n_classes - 1)).collect()
    }

    #[test]
    fn selectivity_is_exact_across_rho() {
        let labels = synth_labels(10_000, 8);
        let mut rng = StdRng::seed_from_u64(1);
        for &rho in &[0.0, 0.3, 0.7, 1.0] {
            let p = correlated(&labels, 0.05, rho, 0, &mut rng);
            assert_eq!(p.n_match, 500, "exact selectivity must hold for ρ={rho}");
            assert!((p.selectivity() - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn rho1_is_structured_rho0_is_spread() {
        // ρ=1 concentrates in few classes; ρ=0 spreads across all. Use distinct-class
        // count of the matched set as a cheap structure proxy.
        let labels = synth_labels(10_000, 8);
        let mut rng = StdRng::seed_from_u64(2);
        let distinct = |p: &Predicate| {
            let mut s = std::collections::HashSet::new();
            for id in 0..labels.len() as u32 {
                if p.test(id) {
                    s.insert(labels[id as usize]);
                }
            }
            s.len()
        };
        let p1 = correlated(&labels, 0.05, 1.0, 0, &mut rng);
        let p0 = correlated(&labels, 0.05, 0.0, 0, &mut rng);
        assert!(
            distinct(&p1) < distinct(&p0),
            "ρ=1 should span fewer classes ({}) than ρ=0 ({})",
            distinct(&p1),
            distinct(&p0)
        );
    }

    #[test]
    fn from_label_matches_count() {
        let labels = vec![0u32, 1, 1, 2, 1];
        let p = from_label(&labels, 1);
        assert_eq!(p.n_match, 3);
        // labels = [0,1,1,2,1] → ids 1,2,4 match; 0,3 do not.
        assert!(p.test(1) && p.test(2) && p.test(4));
        assert!(!p.test(0) && !p.test(3));
    }
}
