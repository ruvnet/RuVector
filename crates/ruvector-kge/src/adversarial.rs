//! Adversarial and security checks (ADR-005). Three things live here: the
//! symmetry-pattern decoy-triple generator (the primary poisoning CI case), a
//! membership-inference probe, and the input-limit guard the bindings call
//! before the FFI boundary. None of these ever logs triple text (ADR-005).

use crate::{EntityId, KgeError, RelationId, Result, Triple};
use std::collections::HashSet;

// ---- Input limits (ADR-005) ------------------------------------------------

/// ADR-005 table limits (index-build budget).
pub const MAX_ENTITIES: usize = 1_000_000;
/// ADR-005 table limits.
pub const MAX_RELATIONS: usize = 100_000;
/// Local defensive bound — ADR-005 states no explicit triple cap; this guards
/// the ingest path from an unbounded edge list.
pub const MAX_TRIPLES: usize = 100_000_000;
/// Local defensive bound — ADR-005 states no explicit dimension cap; this keeps
/// the ANN index build within budget.
pub const MAX_DIMS: usize = 4096;

/// Reject an oversized table before it reaches the scorer/index (ADR-005:
/// rejected with a typed error, never silently truncated).
pub fn check_limits(
    num_entities: usize,
    num_relations: usize,
    num_triples: usize,
    dims: usize,
) -> Result<()> {
    if num_entities > MAX_ENTITIES {
        return Err(KgeError::Limit("more than 1M entities"));
    }
    if num_relations > MAX_RELATIONS {
        return Err(KgeError::Limit("more than 100k relations"));
    }
    if num_triples > MAX_TRIPLES {
        return Err(KgeError::Limit("more than 100M triples"));
    }
    if dims == 0 || dims > MAX_DIMS {
        return Err(KgeError::Limit("dims out of range (1..=4096)"));
    }
    Ok(())
}

// ---- Symmetry-pattern decoy poisoning (Bhardwaj 2021) ----------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate up to `n` symmetry-pattern decoy triples targeting `relation`
/// (Bhardwaj et al., ACL-IJCNLP 2021, arXiv:2111.06345 — the attack that
/// generalises across every model/dataset tested).
///
/// The pattern: for a relation the model has learned as **symmetric**, a target
/// fact `(s, r, o)` is attacked *indirectly*. Adding `(o, r, s')` for a
/// distractor `s' ∉ {s, o}` makes symmetry infer `(s', r, o)` — a competitor
/// that steals rank and score mass from the target subject `s` without the
/// attacker ever touching `(s, r, o)` itself. We deliberately do **not** emit
/// the target's own inverse `(o, r, s)`: for a symmetric relation that
/// *reinforces* the target rather than poisoning it.
///
/// Decoys are deduplicated against `store` and against each other, and capped
/// at `n`. `seed` makes the distractor choice deterministic. The entity id
/// space is inferred from `store` (max id + 1).
#[must_use]
pub fn symmetry_decoys(store: &[Triple], relation: RelationId, n: usize, seed: u64) -> Vec<Triple> {
    let targets: Vec<&Triple> = store.iter().filter(|t| t.r == relation).collect();
    if targets.is_empty() || n == 0 {
        return Vec::new();
    }
    let num_entities = store
        .iter()
        .flat_map(|t| [t.s, t.o])
        .max()
        .map_or(0, |m| m as u64 + 1);
    if num_entities < 3 {
        return Vec::new();
    }
    let existing: HashSet<Triple> = store.iter().copied().collect();
    let mut seen: HashSet<Triple> = HashSet::new();
    let mut out = Vec::with_capacity(n);
    let mut state = seed | 1;

    let mut i = 0usize;
    // Bounded scan: at most n·64 attempts so a saturated graph terminates.
    let max_attempts = n.saturating_mul(64).max(n);
    let mut attempts = 0usize;
    while out.len() < n && attempts < max_attempts {
        attempts += 1;
        let target = targets[i % targets.len()];
        i += 1;
        // Pick a distractor s' ∉ {s, o}.
        let mut sprime = (xorshift64(&mut state) % num_entities) as EntityId;
        if sprime == target.s || sprime == target.o {
            sprime = ((sprime as u64 + 1) % num_entities) as EntityId;
        }
        if sprime == target.s || sprime == target.o {
            continue; // tiny graph collision; try another target/seed draw
        }
        let decoy = Triple::new(target.o, relation, sprime);
        if existing.contains(&decoy) || !seen.insert(decoy) {
            continue;
        }
        out.push(decoy);
    }
    out
}

/// Per-target confidence drop under an attack. The gate (ADR-006) is that
/// confidence **drops** under attack, not that accuracy is perfect.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackReport {
    pub n: usize,
    pub mean_before: f32,
    pub mean_after: f32,
    /// `mean(before − after)` — positive means the attack degraded confidence.
    pub mean_drop: f32,
    pub max_drop: f32,
    /// Share of targeted facts whose score decreased.
    pub fraction_dropped: f32,
    /// `true` when mean confidence dropped (the honest-degradation signal).
    pub dropped: bool,
}

/// Compare targeted-triple scores before and after an attack. `before` and
/// `after` are paired per target; the shorter length wins if they differ.
#[must_use]
pub fn attack_report(before: &[f32], after: &[f32]) -> AttackReport {
    let n = before.len().min(after.len());
    if n == 0 {
        return AttackReport {
            n: 0,
            mean_before: 0.0,
            mean_after: 0.0,
            mean_drop: 0.0,
            max_drop: 0.0,
            fraction_dropped: 0.0,
            dropped: false,
        };
    }
    let mut sum_before = 0.0f32;
    let mut sum_after = 0.0f32;
    let mut max_drop = f32::NEG_INFINITY;
    let mut dropped_count = 0usize;
    for i in 0..n {
        let (b, a) = (before[i], after[i]);
        sum_before += b;
        sum_after += a;
        let drop = b - a;
        if drop > max_drop {
            max_drop = drop;
        }
        if a < b {
            dropped_count += 1;
        }
    }
    let mean_before = sum_before / n as f32;
    let mean_after = sum_after / n as f32;
    let mean_drop = mean_before - mean_after;
    AttackReport {
        n,
        mean_before,
        mean_after,
        mean_drop,
        max_drop,
        fraction_dropped: dropped_count as f32 / n as f32,
        dropped: mean_drop > 0.0,
    }
}

// ---- Membership inference (Wang & Sun, arXiv:2104.08273) --------------------

/// Score-gap statistic between train and held-out triples. A large positive gap
/// means a query-only attacker can infer training-set membership.
#[derive(Debug, Clone, PartialEq)]
pub struct MembershipReport {
    pub train_mean: f32,
    pub holdout_mean: f32,
    /// `train_mean − holdout_mean`.
    pub gap: f32,
    pub threshold: f32,
    /// `true` when the gap exceeds `threshold` (membership is inferable).
    pub alarm: bool,
}

/// Probe membership-inference leakage: the gap between mean train and mean
/// held-out scores, alarming past `threshold` (ADR-005 medium/high threat).
#[must_use]
pub fn membership_inference_probe(
    train_scores: &[f32],
    holdout_scores: &[f32],
    threshold: f32,
) -> MembershipReport {
    let train_mean = mean(train_scores);
    let holdout_mean = mean(holdout_scores);
    let gap = train_mean - holdout_mean;
    MembershipReport {
        train_mean,
        holdout_mean,
        gap,
        threshold,
        alarm: gap > threshold,
    }
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_limits_rejects_oversized_tables() {
        assert!(check_limits(10, 10, 100, 128).is_ok());
        assert!(matches!(
            check_limits(MAX_ENTITIES + 1, 10, 100, 128),
            Err(KgeError::Limit(_))
        ));
        assert!(matches!(
            check_limits(10, MAX_RELATIONS + 1, 100, 128),
            Err(KgeError::Limit(_))
        ));
        assert!(matches!(
            check_limits(10, 10, 100, 0),
            Err(KgeError::Limit(_))
        ));
        assert!(matches!(
            check_limits(10, 10, 100, MAX_DIMS + 1),
            Err(KgeError::Limit(_))
        ));
    }

    #[test]
    fn decoys_exploit_symmetry_without_touching_the_target() {
        // A synthetic graph where relation 0 is symmetric: (s,0,o) & (o,0,s).
        let store = vec![
            Triple::new(0, 0, 1),
            Triple::new(1, 0, 0),
            Triple::new(2, 0, 3),
            Triple::new(3, 0, 2),
            Triple::new(4, 0, 5),
        ];
        let decoys = symmetry_decoys(&store, 0, 4, 12345);
        assert!(!decoys.is_empty());
        assert!(decoys.len() <= 4);
        for d in &decoys {
            // Every decoy uses the attacked relation.
            assert_eq!(d.r, 0);
            // A decoy is (o, r, s') — never a verbatim copy of an existing triple.
            assert!(!store.contains(d));
            // The distractor s' is not the object it hangs off.
            assert_ne!(d.s, d.o);
            // Structural symmetry property: the decoy's subject is the object of
            // some real target triple, so symmetry infers a competitor for it.
            assert!(store.iter().any(|t| t.r == 0 && t.o == d.s));
        }
        // Deterministic under a fixed seed.
        assert_eq!(decoys, symmetry_decoys(&store, 0, 4, 12345));
    }

    #[test]
    fn decoys_empty_for_absent_relation() {
        let store = vec![Triple::new(0, 0, 1), Triple::new(1, 0, 2)];
        assert!(symmetry_decoys(&store, 9, 4, 1).is_empty());
    }

    #[test]
    fn attack_report_shows_confidence_drop() {
        let before = [0.9, 0.8, 0.95, 0.7];
        let after = [0.5, 0.6, 0.4, 0.65];
        let r = attack_report(&before, &after);
        assert!(r.dropped);
        assert!(r.mean_drop > 0.0);
        assert_eq!(r.fraction_dropped, 1.0);
        assert!(r.max_drop >= 0.55 - 1e-6);
    }

    #[test]
    fn attack_report_no_drop_when_scores_hold() {
        let before = [0.5, 0.5];
        let after = [0.5, 0.6];
        let r = attack_report(&before, &after);
        assert!(!r.dropped);
    }

    #[test]
    fn membership_probe_alarms_on_large_gap() {
        let train = [0.95, 0.92, 0.98];
        let holdout = [0.40, 0.45, 0.38];
        let r = membership_inference_probe(&train, &holdout, 0.2);
        assert!(r.alarm);
        assert!(r.gap > 0.2);

        let tight = membership_inference_probe(&train, &train, 0.2);
        assert!(!tight.alarm);
    }
}
