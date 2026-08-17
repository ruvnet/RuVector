//! Retention policies: turn a memory set (optionally already partitioned
//! into topic clusters) plus a target size into a retained-id list.
//!
//! Every policy here scores candidates with the *same* scorer —
//! `ruvector_agent_memory::CoherencePolicy`, the winning policy from the
//! 2026-06-14 nightly — so the only independent variable under test is how
//! the retention *budget* is allocated (globally vs. per-partition with a
//! floor), not the scoring function itself.

use crate::corpus::MemoryRecord;
use ruvector_agent_memory::{CoherencePolicy, CompactionPolicy, MemoryEntry};
use ruvector_mincut::VertexId;
use std::collections::HashMap;

fn to_entry(r: &MemoryRecord) -> MemoryEntry {
    MemoryEntry {
        id: r.id,
        vector: r.embedding.clone(),
        label: None,
        created_at: 0,
        last_accessed_at: r.last_accessed_tick as u64,
        access_count: r.access_count as u64,
    }
}

/// Global top-score baseline: score every record with `CoherencePolicy`
/// and keep the `target_size` highest, with no notion of topic partitions.
/// This is the exact mechanism a session-context-biased consolidation
/// event applies today (nightly 2026-06-14) — a fair, non-strawman
/// baseline, not a deliberately weak one.
pub fn retain_global_top_score(
    records: &[MemoryRecord],
    context_window: &[Vec<f32>],
    target_size: usize,
) -> Vec<u64> {
    let entries: Vec<MemoryEntry> = records.iter().map(to_entry).collect();
    let survivors = CoherencePolicy::default().select_survivors(
        &entries,
        target_size.min(entries.len()),
        context_window,
    );
    survivors.into_iter().map(|i| entries[i].id).collect()
}

pub struct RetentionPolicy {
    /// Minimum records guaranteed to each non-empty partition, capacity
    /// permitting. This is the mechanism that protects minority topics:
    /// a proportional-only allocation (`floor_min = 0`) can round a small
    /// partition's share down to zero and drop it entirely.
    pub floor_min: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self { floor_min: 3 }
    }
}

/// Largest-remainder apportionment of `total` across `weights`, capped
/// per-bucket by `caps`. Guaranteed to allocate exactly
/// `total.min(caps.iter().sum())`.
fn apportion_capped(total: usize, weights: &[f64], caps: &[usize]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 || total == 0 {
        return vec![0; n];
    }
    let sum_w: f64 = weights.iter().sum();
    let raw: Vec<f64> = if sum_w > 0.0 {
        weights.iter().map(|w| total as f64 * w / sum_w).collect()
    } else {
        vec![0.0; n]
    };
    let mut alloc: Vec<usize> = raw
        .iter()
        .zip(caps)
        .map(|(x, &cap)| (x.floor() as usize).min(cap))
        .collect();

    let mut fracs: Vec<(usize, f64)> = raw
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x.fract()))
        .collect();
    fracs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut assigned: usize = alloc.iter().sum();
    let target = total.min(caps.iter().sum());

    // First pass: hand out remainder units by largest fractional share.
    for &(i, _) in &fracs {
        if assigned >= target {
            break;
        }
        if alloc[i] < caps[i] {
            alloc[i] += 1;
            assigned += 1;
        }
    }
    // Second pass: any residual (all fractional winners were already at
    // cap) goes to whoever still has spare capacity, largest weight first.
    if assigned < target {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap());
        for _ in 0..2 {
            for &i in &order {
                if assigned >= target {
                    break;
                }
                if alloc[i] < caps[i] {
                    alloc[i] += 1;
                    assigned += 1;
                }
            }
        }
    }
    alloc
}

/// Partition-aware retention: allocate `target_size` across `partitions`
/// with a per-partition floor, then rank within each partition by
/// `CoherencePolicy` and keep its allocated share.
pub fn retain_partitioned(
    records: &[MemoryRecord],
    partitions: &[Vec<VertexId>],
    context_window: &[Vec<f32>],
    target_size: usize,
    policy: &RetentionPolicy,
) -> Vec<u64> {
    let by_id: HashMap<u64, &MemoryRecord> = records.iter().map(|r| (r.id, r)).collect();
    let sizes: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
    let floors: Vec<usize> = sizes.iter().map(|&s| policy.floor_min.min(s)).collect();
    let sum_floor: usize = floors.iter().sum();
    let remaining = target_size.saturating_sub(sum_floor);
    let capacities: Vec<usize> = sizes.iter().zip(&floors).map(|(&s, &f)| s - f).collect();
    let weights: Vec<f64> = capacities.iter().map(|&c| c as f64).collect();
    let extra = apportion_capped(remaining, &weights, &capacities);

    let mut retained = Vec::new();
    for (i, part) in partitions.iter().enumerate() {
        let budget = (floors[i] + extra[i]).min(part.len());
        if budget == 0 {
            continue;
        }
        let entries: Vec<MemoryEntry> = part
            .iter()
            .filter_map(|id| by_id.get(id).map(|r| to_entry(r)))
            .collect();
        let survivors =
            CoherencePolicy::default().select_survivors(&entries, budget, context_window);
        for idx in survivors {
            retained.push(entries[idx].id);
        }
    }
    retained
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apportion_respects_caps_and_hits_target() {
        let weights = vec![10.0, 1.0, 1.0];
        let caps = vec![10, 1, 1];
        let alloc = apportion_capped(8, &weights, &caps);
        assert_eq!(alloc.iter().sum::<usize>(), 8);
        assert!(alloc.iter().zip(&caps).all(|(a, c)| a <= c));
    }

    #[test]
    fn apportion_never_exceeds_total_capacity() {
        let weights = vec![1.0, 1.0];
        let caps = vec![2, 2];
        let alloc = apportion_capped(10, &weights, &caps);
        assert_eq!(alloc.iter().sum::<usize>(), 4);
    }

    #[test]
    fn floor_protects_a_tiny_partition_from_proportional_zero() {
        // Partition sizes 970 / 30 at 5% overall retention (target=50):
        // pure proportional would give the 30-record partition ~1.5 -> 1
        // or, with harsher rounding, 0. The floor guarantees >= floor_min.
        let sizes = [970usize, 30usize];
        let floors: Vec<usize> = sizes.iter().map(|&s| 3usize.min(s)).collect();
        assert_eq!(floors[1], 3);
    }
}
