//! Background compaction for dead space reclamation.
//!
//! Compaction scheduling policy (from spec 10, section 7):
//! - IO budget: max 30% of IOPS (60% in emergency)
//! - Priority: queries > ingest > compaction
//! - Triggers: dead_space > 20%, segment_count > 32, time > 60s
//! - Emergency: dead_space > 70% -> preempt ingest
//!
//! Segment selection order:
//! 1. Tombstoned segments (reclaim dead space)
//! 2. Small VEC_SEGs (< 1MB, merge into larger)
//! 3. High-overlap INDEX_SEGs
//! 4. Cold OVERLAY_SEGs

/// Compaction trigger thresholds.
pub struct CompactionThresholds {
    /// Minimum dead space ratio to trigger compaction.
    pub dead_space_ratio: f64,
    /// Maximum segment count before compaction.
    pub max_segment_count: u32,
    /// Minimum seconds since last compaction.
    pub min_interval_secs: u64,
    /// Emergency dead space ratio (preempts ingest).
    pub emergency_ratio: f64,
}

impl Default for CompactionThresholds {
    fn default() -> Self {
        Self {
            dead_space_ratio: 0.20,
            max_segment_count: 32,
            min_interval_secs: 60,
            emergency_ratio: 0.70,
        }
    }
}

/// Compaction decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionDecision {
    /// No compaction needed.
    None,
    /// Normal compaction should run.
    Normal,
    /// Emergency compaction (high dead space).
    Emergency,
}

/// Evaluate whether compaction should run.
pub fn evaluate_triggers(
    dead_space_ratio: f64,
    segment_count: u32,
    secs_since_last: u64,
    thresholds: &CompactionThresholds,
) -> CompactionDecision {
    // Emergency check first.
    if dead_space_ratio > thresholds.emergency_ratio {
        return CompactionDecision::Emergency;
    }

    // Check all normal conditions.
    if secs_since_last < thresholds.min_interval_secs {
        return CompactionDecision::None;
    }

    if dead_space_ratio > thresholds.dead_space_ratio {
        return CompactionDecision::Normal;
    }

    if segment_count > thresholds.max_segment_count {
        return CompactionDecision::Normal;
    }

    CompactionDecision::None
}

/// Represents a compaction plan: which segments to compact and how.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct CompactionPlan {
    /// Segment IDs to compact (input).
    pub source_segments: Vec<u64>,
    /// Whether this is emergency compaction.
    pub emergency: bool,
    /// IO budget as a fraction (0.30 normal, 0.60 emergency).
    pub io_budget: f64,
}

impl CompactionPlan {
    /// Create a normal compaction plan.
    #[allow(dead_code)]
    pub(crate) fn normal(segments: Vec<u64>) -> Self {
        Self {
            source_segments: segments,
            emergency: false,
            io_budget: 0.30,
        }
    }

    /// Create an emergency compaction plan.
    #[allow(dead_code)]
    pub(crate) fn emergency(segments: Vec<u64>) -> Self {
        Self {
            source_segments: segments,
            emergency: true,
            io_budget: 0.60,
        }
    }
}

/// Select segments for compaction based on the tiered strategy.
///
/// Priority:
/// 1. Tombstoned segments
/// 2. Small VEC_SEGs (< threshold)
/// 3. Remaining segments by age
#[allow(dead_code)]
pub(crate) fn select_segments(
    segment_dir: &[(u64, u64, u8, bool)], // (seg_id, payload_len, seg_type, is_tombstoned)
    max_segments: usize,
) -> Vec<u64> {
    let mut selected = Vec::new();

    // Phase 1: tombstoned segments.
    for &(seg_id, _, _, tombstoned) in segment_dir {
        if tombstoned && selected.len() < max_segments {
            selected.push(seg_id);
        }
    }

    // Phase 2: small VEC_SEGs (< 1MB).
    let small_threshold = 1024 * 1024;
    for &(seg_id, payload_len, seg_type, _) in segment_dir {
        if seg_type == 0x01
            && payload_len < small_threshold
            && selected.len() < max_segments
            && !selected.contains(&seg_id)
        {
            selected.push(seg_id);
        }
    }

    // Phase 3: fill remaining with oldest segments.
    for &(seg_id, _, _, _) in segment_dir {
        if selected.len() >= max_segments {
            break;
        }
        if !selected.contains(&seg_id) {
            selected.push(seg_id);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_compaction_when_fresh() {
        let decision = evaluate_triggers(0.10, 10, 30, &CompactionThresholds::default());
        assert_eq!(decision, CompactionDecision::None);
    }

    #[test]
    fn normal_compaction_on_dead_space() {
        let decision = evaluate_triggers(0.25, 10, 120, &CompactionThresholds::default());
        assert_eq!(decision, CompactionDecision::Normal);
    }

    #[test]
    fn normal_compaction_on_segment_count() {
        let decision = evaluate_triggers(0.10, 50, 120, &CompactionThresholds::default());
        assert_eq!(decision, CompactionDecision::Normal);
    }

    #[test]
    fn emergency_compaction_on_high_dead_space() {
        let decision = evaluate_triggers(0.75, 10, 10, &CompactionThresholds::default());
        assert_eq!(decision, CompactionDecision::Emergency);
    }

    #[test]
    fn no_compaction_before_interval() {
        let decision = evaluate_triggers(0.25, 50, 30, &CompactionThresholds::default());
        // Even though dead_space and segment_count exceed thresholds,
        // interval hasn't passed.
        assert_eq!(decision, CompactionDecision::None);
    }

    // --- the policy is now REACHABLE from the store, not just from these tests ---
    //
    // Every input evaluate_triggers needs was already tracked on RvfStore
    // (dead_space_ratio and total_segments via status(), last_compaction_time stamped
    // by compact()). These drive a REAL store rather than passing numbers straight to
    // evaluate_triggers, so they lock the WIRING -- the part that was missing -- and
    // not just the arithmetic, which the tests above already cover.

    use crate::options::DistanceMetric;
    use crate::store::RvfStore;
    use crate::RvfOptions;
    use tempfile::TempDir;

    fn store_with(dir: &TempDir, name: &str, count: u64, delete: u64) -> RvfStore {
        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut store = RvfStore::create(&dir.path().join(name), options).unwrap();
        if count > 0 {
            let vecs: Vec<Vec<f32>> = (0..count).map(|i| vec![i as f32, 1.0, 0.0, 0.0]).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let ids: Vec<u64> = (0..count).collect();
            store.ingest_batch(&refs, &ids, None).unwrap();
        }
        if delete > 0 {
            let doomed: Vec<u64> = (0..delete).collect();
            store.delete(&doomed).unwrap();
        }
        store
    }

    #[test]
    fn a_store_with_no_dead_space_is_not_due() {
        let dir = TempDir::new().unwrap();
        let store = store_with(&dir, "clean.rvf", 10, 0);

        assert_eq!(store.should_compact(), CompactionDecision::None);
    }

    #[test]
    fn dead_space_over_the_threshold_is_read_from_LIVE_state() {
        // 6 of 10 deleted = 0.60: over the 0.20 normal trigger, under the 0.70 emergency.
        // This is the case that could not be reached before, because nothing called the
        // policy with the store's real numbers.
        let dir = TempDir::new().unwrap();
        let store = store_with(&dir, "dirty.rvf", 10, 6);

        assert!(
            store.status().dead_space_ratio > 0.20,
            "precondition: real dead space"
        );
        assert_eq!(store.should_compact(), CompactionDecision::Normal);
    }

    #[test]
    fn dead_space_over_the_emergency_ratio_escalates() {
        // 8 of 10 deleted = 0.80, above the 0.70 emergency ratio.
        let dir = TempDir::new().unwrap();
        let store = store_with(&dir, "emergency.rvf", 10, 8);

        assert_eq!(store.should_compact(), CompactionDecision::Emergency);
    }

    #[test]
    fn compact_if_needed_distinguishes_not_needed_from_ran_and_found_nothing() {
        let dir = TempDir::new().unwrap();
        let mut clean = store_with(&dir, "skip.rvf", 10, 0);
        let mut dirty = store_with(&dir, "run.rvf", 10, 6);

        // None => the policy declined. That must NOT look like a compaction that ran and
        // reclaimed zero, which is Some(result) -- the distinction is why this returns an
        // Option rather than a bare CompactionResult.
        assert!(clean.compact_if_needed().unwrap().is_none());
        assert!(dirty.compact_if_needed().unwrap().is_some());
    }

    #[test]
    fn custom_thresholds_are_honoured() {
        let dir = TempDir::new().unwrap();
        let store = store_with(&dir, "custom.rvf", 10, 0);

        // A store the default policy declines must flip under a stricter threshold,
        // which proves should_compact_with reads live state rather than a constant.
        assert_eq!(store.should_compact(), CompactionDecision::None);
        let eager = CompactionThresholds {
            dead_space_ratio: 0.20,
            max_segment_count: 0,
            min_interval_secs: 0,
            emergency_ratio: 0.70,
        };
        assert_eq!(
            store.should_compact_with(&eager),
            CompactionDecision::Normal
        );
    }

    #[test]
    fn select_tombstoned_first() {
        let segments = vec![
            (1, 500_000, 0x01, false),
            (2, 100_000, 0x01, true), // tombstoned
            (3, 200_000, 0x01, false),
            (4, 50_000, 0x01, true), // tombstoned
        ];
        let selected = select_segments(&segments, 3);
        // Tombstoned segments (2, 4) should come first.
        assert_eq!(selected[0], 2);
        assert_eq!(selected[1], 4);
        assert_eq!(selected.len(), 3);
    }
}
