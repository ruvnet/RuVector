//! Call-site instrumentation for diagnosing `BoundedInstance`/`MinCutWrapper`
//! latency scaling, added by the 2026-11-11 nightly research run.
//!
//! ADR-345 (`docs/adr/ADR-345-mincut-gated-forgetting.md`) measured
//! `RuVectorGraphAnalyzer::partition()` scaling from ~77ms (n=50) to ~11.4s
//! (n=400) and left the root cause unattributed. The 2026-09-11 nightly's
//! Next Research item 2 proposed `connectivity::polylog::PolylogConnectivity`
//! as a replacement backend for the connectivity checks inside
//! `instance::bounded::BoundedInstance` as the fix. This module provides the
//! counters needed to test that proposal with real call-site attribution
//! instead of guessing: every `is_connected()` call, every
//! `search_for_cuts()` oracle invocation, and every `process_instances()`
//! instance attempt increments an atomic counter here.
//!
//! Counting is gated behind the `profile-counters` Cargo feature so the
//! default production build pays zero cost (the increments compile to
//! nothing). With the feature on, each increment is a single relaxed atomic
//! add — negligible next to the work being measured, and ordering between
//! counters is not load-bearing (each is read only after the timed region
//! that increments it has fully returned, from a single thread, in the
//! `examples/profile_partition_scaling.rs` harness this module was built
//! for).

use std::sync::atomic::{AtomicU64, Ordering};

static CONNECTIVITY_CHECK_CALLS: AtomicU64 = AtomicU64::new(0);
static CONNECTIVITY_CHECK_NODES_VISITED: AtomicU64 = AtomicU64::new(0);
static ORACLE_SEARCH_CALLS: AtomicU64 = AtomicU64::new(0);
static INSTANCES_TRIED: AtomicU64 = AtomicU64::new(0);

/// Record one `BoundedInstance::is_connected()` (whole-graph BFS) call.
///
/// `nodes_visited` is the BFS visited-set size for that call, i.e. the work
/// actually done (not the graph's total vertex count, which may differ on
/// an early-disconnect path — there is none here, `is_connected` always
/// runs the BFS to completion, but the parameter stays honest about what it
/// measures rather than assuming).
#[inline]
pub fn record_connectivity_check(nodes_visited: u64) {
    #[cfg(feature = "profile-counters")]
    {
        CONNECTIVITY_CHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        CONNECTIVITY_CHECK_NODES_VISITED.fetch_add(nodes_visited, Ordering::Relaxed);
    }
    #[cfg(not(feature = "profile-counters"))]
    {
        let _ = nodes_visited;
    }
}

/// Record one `DeterministicLocalKCut::search()` invocation from
/// `BoundedInstance::search_for_cuts()`.
#[inline]
pub fn record_oracle_search_call() {
    #[cfg(feature = "profile-counters")]
    ORACLE_SEARCH_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Record one `BoundedInstance` attempted (its `.query()` invoked) inside
/// `MinCutWrapper::process_instances()`'s `start_idx..MAX_INSTANCES` loop.
#[inline]
pub fn record_instance_tried() {
    #[cfg(feature = "profile-counters")]
    INSTANCES_TRIED.fetch_add(1, Ordering::Relaxed);
}

/// Point-in-time snapshot of all counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub connectivity_check_calls: u64,
    pub connectivity_check_nodes_visited: u64,
    pub oracle_search_calls: u64,
    pub instances_tried: u64,
}

/// Read all counters without resetting them.
pub fn snapshot() -> Snapshot {
    Snapshot {
        connectivity_check_calls: CONNECTIVITY_CHECK_CALLS.load(Ordering::Relaxed),
        connectivity_check_nodes_visited: CONNECTIVITY_CHECK_NODES_VISITED.load(Ordering::Relaxed),
        oracle_search_calls: ORACLE_SEARCH_CALLS.load(Ordering::Relaxed),
        instances_tried: INSTANCES_TRIED.load(Ordering::Relaxed),
    }
}

/// Reset all counters to zero. Intended for single-threaded benchmark
/// harnesses that call this between measured runs; not safe to call
/// concurrently with in-flight `record_*` calls on another thread.
pub fn reset() {
    CONNECTIVITY_CHECK_CALLS.store(0, Ordering::Relaxed);
    CONNECTIVITY_CHECK_NODES_VISITED.store(0, Ordering::Relaxed);
    ORACLE_SEARCH_CALLS.store(0, Ordering::Relaxed);
    INSTANCES_TRIED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_then_record_then_snapshot_round_trips() {
        reset();
        record_connectivity_check(7);
        record_connectivity_check(3);
        record_oracle_search_call();
        record_instance_tried();
        record_instance_tried();

        let snap = snapshot();

        #[cfg(feature = "profile-counters")]
        {
            assert_eq!(snap.connectivity_check_calls, 2);
            assert_eq!(snap.connectivity_check_nodes_visited, 10);
            assert_eq!(snap.oracle_search_calls, 1);
            assert_eq!(snap.instances_tried, 2);
        }
        #[cfg(not(feature = "profile-counters"))]
        {
            // Without the feature, every record_* is a no-op: counters stay
            // at zero regardless of how many times they are called.
            assert_eq!(snap, Snapshot::default());
        }

        reset();
        assert_eq!(snapshot(), Snapshot::default());
    }
}
