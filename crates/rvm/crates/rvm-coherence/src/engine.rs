//! Unified coherence engine.
//!
//! The `CoherenceEngine` ties together:
//! - Graph state (from [`graph`])
//! - MinCut computation (from [`mincut`] or [`bridge`])
//! - Coherence scoring (from [`scoring`] or [`bridge`])
//! - Cut pressure (from [`pressure`])
//! - Adaptive recomputation frequency (from [`adaptive`])
//!
//! This is the single entry point that the kernel calls on each epoch.
//!
//! ## Lifecycle
//!
//! ```text
//! engine.add_partition(id)          -- register a new partition
//! engine.record_communication(a, b) -- record inter-partition traffic
//! engine.tick(cpu_load)             -- advance epoch, recompute if adaptive says so
//! engine.score(id)                  -- read the latest coherence score
//! engine.pressure(id)               -- read the latest cut pressure
//! engine.recommend()                -- get split/merge recommendation
//! ```

use rvm_types::{CoherenceScore, CutPressure, PartitionId, RvmError};

use crate::adaptive::AdaptiveCoherenceEngine;
use crate::bridge::{
    BackendMinCutResult, BuiltinCoherence, BuiltinMinCut, CoherenceBackend, MinCutBackend,
};
use crate::fennel::FennelPlacer;
use crate::graph::{CoherenceGraph, GraphError};
use crate::pressure::{self, MergeSignal, SPLIT_THRESHOLD_BP};

/// Maximum number of partitions tracked by the coherence engine.
const ENGINE_MAX_NODES: usize = 32;

/// Maximum number of directed edges tracked by the coherence engine.
const ENGINE_MAX_EDGES: usize = 128;

// -----------------------------------------------------------------------
// Split decision policy (roadmap item 5)
// -----------------------------------------------------------------------
//
// Split decisions combine BOTH pressure ratio AND cut quality:
//
// 1. pressure <= SPLIT_THRESHOLD_BP (8000): never split.
// 2. pressure >= CRITICAL_PRESSURE_BP (9500): split unconditionally.
//    A partition whose traffic is >= 95% external is pathological no
//    matter what the cut looks like (safety valve; also preserves the
//    pre-existing behavior for fully-external partitions).
// 3. SPLIT_THRESHOLD_BP < pressure < CRITICAL_PRESSURE_BP: split only
//    if the epoch mincut found a *usable, low-conductance* boundary:
//    both sides non-empty and conductance <= MAX_SPLIT_CONDUCTANCE_BP.
//    Without a coherent sub-cluster to peel off, splitting would just
//    move the traffic, not reduce it.
//
// Conductance is `cut_weight / min(vol(left), vol(right))` in basis
// points, where vol(side) is the sum of total incident edge weight of
// the side's members. 10_000 bp means the cut is as heavy as the
// lighter side's entire traffic (worst case, e.g. cutting off a leaf).

/// Pressure (basis points) above which a split fires regardless of cut
/// quality. See the policy block above.
pub const CRITICAL_PRESSURE_BP: u32 = 9_500;

/// Maximum cut conductance (basis points) for a mid-band split. A cut
/// crossing at most 50% of the lighter side's volume indicates a
/// genuinely weakly-coupled sub-cluster worth splitting off.
pub const MAX_SPLIT_CONDUCTANCE_BP: u32 = 5_000;

/// Side of a split boundary, as produced by the epoch mincut and by
/// hot-path incremental placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSide {
    /// Stays coupled to the split source partition.
    Keep,
    /// Should be re-homed to the newly created child partition.
    MoveToChild,
}

/// A concrete split boundary computed by the epoch mincut task.
///
/// `keep` is the side of the cut containing the split candidate itself
/// (its neighbors stay attached to the source after the split);
/// `move_out` is the far side (neighbors whose communication edges are
/// re-homed to the child by [`CoherenceEngine::apply_split_boundary`]).
#[derive(Debug, Clone)]
pub struct SplitPlan {
    /// The partition this plan splits.
    pub partition: PartitionId,
    /// Neighborhood members that stay with the source (includes `partition`).
    pub keep: [Option<PartitionId>; ENGINE_MAX_NODES],
    /// Number of valid entries in `keep`.
    pub keep_count: u16,
    /// Neighborhood members re-homed to the child.
    pub move_out: [Option<PartitionId>; ENGINE_MAX_NODES],
    /// Number of valid entries in `move_out`.
    pub move_count: u16,
    /// Total weight of edges crossing the cut.
    pub cut_weight: u64,
    /// Cut conductance in basis points (see policy block); 10_000 when
    /// no usable two-sided cut exists.
    pub conductance_bp: u32,
    /// Whether the mincut computation completed within budget.
    pub within_budget: bool,
    /// Engine epoch at which this plan was computed.
    pub epoch: u64,
}

impl SplitPlan {
    /// Whether this plan represents a usable two-sided boundary.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.keep_count > 0 && self.move_count > 0
    }

    /// Whether the boundary quality permits a mid-band split.
    #[must_use]
    pub const fn is_low_conductance(&self) -> bool {
        self.is_usable() && self.conductance_bp <= MAX_SPLIT_CONDUCTANCE_BP
    }
}

/// A recommendation produced by the coherence engine after an epoch tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceDecision {
    /// No split or merge action is warranted.
    NoAction,
    /// A partition should be split due to high cut pressure.
    SplitRecommended {
        /// The partition that should be split.
        partition: PartitionId,
        /// The cut pressure that triggered the recommendation.
        pressure: CutPressure,
    },
    /// Two partitions should be merged due to high mutual coherence.
    MergeRecommended {
        /// First partition to merge.
        a: PartitionId,
        /// Second partition to merge.
        b: PartitionId,
        /// Mutual coherence score.
        mutual_coherence: CoherenceScore,
    },
}

/// Per-partition cached scoring data.
#[derive(Debug, Clone, Copy)]
struct PartitionEntry {
    /// Partition ID.
    id: PartitionId,
    /// Most recently computed coherence score.
    score: CoherenceScore,
    /// Most recently computed cut pressure.
    pressure: CutPressure,
    /// Whether this slot is active.
    active: bool,
}

impl PartitionEntry {
    const EMPTY: Self = Self {
        id: PartitionId::HYPERVISOR, // sentinel; never matched when !active
        score: CoherenceScore::MAX,
        pressure: CutPressure::ZERO,
        active: false,
    };
}

/// The unified coherence engine.
///
/// Generics `MCB` and `CB` allow injecting custom mincut and coherence
/// scoring backends for testing or for the ruvector bridge.
pub struct CoherenceEngine<MCB: MinCutBackend, CB: CoherenceBackend> {
    /// The communication topology graph.
    graph: CoherenceGraph<ENGINE_MAX_NODES, ENGINE_MAX_EDGES>,
    /// Adaptive recomputation controller.
    adaptive: AdaptiveCoherenceEngine,
    /// MinCut backend.
    mincut_backend: MCB,
    /// Coherence scoring backend.
    coherence_backend: CB,
    /// Per-partition cached scores and pressures.
    entries: [PartitionEntry; ENGINE_MAX_NODES],
    /// Epoch counter (incremented on each `tick`).
    epoch: u64,
    /// Decision computed on the last recompute (or after the last graph
    /// mutation). Returned on skip ticks instead of re-running the O(n^2)
    /// recommendation pass over stale data.
    cached_decision: CoherenceDecision,
    /// Whether `cached_decision` is still valid. Invalidated by graph
    /// mutations (`record_communication`, `add_partition`,
    /// `remove_partition`).
    cache_valid: bool,
    /// EPOCH TASK output: the mincut boundary for the current best
    /// split candidate, refreshed on recompute ticks (or on demand by
    /// `split_plan_for`). `None` when no candidate exceeds the split
    /// threshold. Consumed by `apply_split_boundary`.
    split_plan: Option<SplitPlan>,
    /// HOT PATH state: Fennel placer seeded from `split_plan`. Routes
    /// newly arriving nodes to a boundary side in O(degree) without
    /// recomputing the mincut.
    placer: FennelPlacer,
}

// -----------------------------------------------------------------------
// Type alias for the default engine (built-in backends)
// -----------------------------------------------------------------------

/// Default coherence engine using built-in Stoer-Wagner and ratio scoring.
pub type DefaultCoherenceEngine =
    CoherenceEngine<BuiltinMinCut<ENGINE_MAX_NODES>, BuiltinCoherence>;

/// RuVector-backed coherence engine (available with `ruvector` feature).
#[cfg(feature = "ruvector")]
pub type RuVectorCoherenceEngine = CoherenceEngine<
    crate::bridge::RuVectorMinCut<ENGINE_MAX_NODES>,
    crate::bridge::SpectralCoherence,
>;

// -----------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------

impl DefaultCoherenceEngine {
    /// Create a new default engine with built-in backends.
    ///
    /// `max_iterations` controls the Stoer-Wagner budget per mincut
    /// computation.
    #[must_use]
    pub fn with_defaults(max_iterations: u32) -> Self {
        Self::new(
            BuiltinMinCut::new(max_iterations),
            BuiltinCoherence,
        )
    }
}

#[cfg(feature = "ruvector")]
impl RuVectorCoherenceEngine {
    /// Create a new engine with RuVector backends.
    ///
    /// `max_iterations` is passed to the fallback Stoer-Wagner until the
    /// ruvector crates gain `no_std` support.
    #[must_use]
    pub fn with_ruvector(max_iterations: u32) -> Self {
        Self::new(
            crate::bridge::RuVectorMinCut::new(max_iterations),
            crate::bridge::SpectralCoherence,
        )
    }
}

impl<MCB: MinCutBackend, CB: CoherenceBackend> CoherenceEngine<MCB, CB> {
    /// Create a new engine with the given backends.
    #[must_use]
    pub fn new(mincut_backend: MCB, coherence_backend: CB) -> Self {
        Self {
            graph: CoherenceGraph::new(),
            adaptive: AdaptiveCoherenceEngine::new(),
            mincut_backend,
            coherence_backend,
            entries: [PartitionEntry::EMPTY; ENGINE_MAX_NODES],
            epoch: 0,
            cached_decision: CoherenceDecision::NoAction,
            cache_valid: false,
            split_plan: None,
            placer: FennelPlacer::with_default_alpha(),
        }
    }

    /// Current epoch counter.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of active partitions tracked by the engine.
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.graph.node_count() as usize
    }

    /// Register a new partition in the coherence graph.
    pub fn add_partition(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.cache_valid = false;
        self.graph
            .add_node(id)
            .map_err(|e| match e {
                GraphError::DuplicateNode => RvmError::InvalidPartitionState,
                GraphError::NodeCapacityExhausted => RvmError::ResourceLimitExceeded,
                _ => RvmError::InternalError,
            })?;

        // Find a free entry slot
        for entry in self.entries.iter_mut() {
            if !entry.active {
                entry.id = id;
                entry.score = CoherenceScore::MAX;
                entry.pressure = CutPressure::ZERO;
                entry.active = true;
                return Ok(());
            }
        }
        // Shouldn't happen because the graph already accepted the node,
        // but guard against it.
        Err(RvmError::ResourceLimitExceeded)
    }

    /// Remove a partition from the coherence graph.
    pub fn remove_partition(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.cache_valid = false;
        self.placer.remove(id);
        if self
            .split_plan
            .as_ref()
            .is_some_and(|p| p.partition == id)
        {
            self.split_plan = None;
        }
        self.graph
            .remove_node(id)
            .map_err(|_| RvmError::PartitionNotFound)?;

        // Clear the entry
        for entry in self.entries.iter_mut() {
            if entry.active && entry.id == id {
                entry.active = false;
                break;
            }
        }
        Ok(())
    }

    /// Record a directed communication event between two partitions.
    ///
    /// If no edge exists yet, one is created. If an edge already exists,
    /// its weight is incremented by `weight`.
    ///
    /// Uses the graph's adjacency-matrix-backed `find_directed_edge` for
    /// O(1) existence check + O(out-degree) edge lookup instead of the
    /// previous O(E) scan over all active edges.
    pub fn record_communication(
        &mut self,
        from: PartitionId,
        to: PartitionId,
        weight: u64,
    ) -> Result<(), RvmError> {
        self.cache_valid = false;
        match self.graph.find_directed_edge(from, to) {
            Some(eidx) => {
                // Clamp to i64::MAX: a plain `as i64` cast would turn
                // weights >= 2^63 into negative deltas (weight decrease).
                let delta = weight.min(i64::MAX as u64) as i64;
                self.graph
                    .update_weight(eidx, delta)
                    .map_err(|_| RvmError::InternalError)?;
            }
            None => {
                self.graph.add_edge(from, to, weight).map_err(|e| match e {
                    GraphError::EdgeCapacityExhausted => RvmError::ResourceLimitExceeded,
                    GraphError::NodeNotFound => RvmError::PartitionNotFound,
                    _ => RvmError::InternalError,
                })?;
            }
        }
        Ok(())
    }

    /// Advance one epoch.
    ///
    /// Consults the adaptive engine to decide whether to recompute
    /// coherence scores and cut pressures. Returns the strongest
    /// split or merge recommendation found, or `NoAction`.
    /// Edge weight decay rate per epoch (basis points). 500 = 5% decay.
    const EDGE_DECAY_BP: u16 = 500;

    /// Advance one epoch.
    ///
    /// Decays edge weights by 5% per epoch to prevent stale communication
    /// patterns from dominating. Then consults the adaptive engine to
    /// decide whether to recompute scores and pressures. Returns the
    /// strongest split or merge recommendation, or `NoAction`.
    pub fn tick(&mut self, cpu_load_percent: u8) -> CoherenceDecision {
        self.epoch = self.epoch.wrapping_add(1);

        // Decay edge weights each epoch to prevent stale communication
        // patterns from dominating the graph.
        self.graph.decay_weights(Self::EDGE_DECAY_BP);

        let should_recompute = self.adaptive.tick(cpu_load_percent);
        if !should_recompute {
            // Skip tick: avoid the O(n^2) recommendation pass over stale
            // data. Reuse the cached decision unless the graph mutated
            // since it was computed.
            if !self.cache_valid {
                self.cached_decision = self.recommend();
                self.cache_valid = true;
            }
            return self.cached_decision;
        }

        // Recompute scores and pressures for all active partitions
        for entry in self.entries.iter_mut() {
            if !entry.active {
                continue;
            }
            entry.score = self.coherence_backend.compute_score(entry.id, &self.graph);

            let pr = pressure::compute_cut_pressure(entry.id, &self.graph);
            entry.pressure = pr.pressure;
        }

        // EPOCH TASK: when a split candidate exists, run the exact
        // mincut now -- off the hot path -- and cache the boundary plan
        // plus the Fennel placer seeding for subsequent hot-path
        // placements. This is the only place the mincut runs during
        // normal epoch processing (pressure-triggered).
        self.split_plan = None;
        if let Some((candidate, _)) = self.best_split_candidate() {
            let plan = self.build_split_plan(candidate);
            self.seed_placer_from_plan(&plan);
            self.split_plan = Some(plan);
        }

        self.adaptive.record_computation();
        let decision = self.recommend();
        self.cached_decision = decision;
        self.cache_valid = true;
        decision
    }

    /// Get the current coherence score for a partition.
    #[must_use]
    pub fn score(&self, id: PartitionId) -> CoherenceScore {
        for entry in &self.entries {
            if entry.active && entry.id == id {
                return entry.score;
            }
        }
        CoherenceScore::MAX // unknown partition treated as fully coherent
    }

    /// Get the current cut pressure for a partition.
    #[must_use]
    pub fn pressure(&self, id: PartitionId) -> CutPressure {
        for entry in &self.entries {
            if entry.active && entry.id == id {
                return entry.pressure;
            }
        }
        CutPressure::ZERO // unknown partition has no pressure
    }

    /// Get the strongest split or merge recommendation without advancing
    /// the epoch.
    ///
    /// Split decisions combine pressure ratio AND cut quality: see the
    /// policy block at the top of this module ([`CRITICAL_PRESSURE_BP`],
    /// [`MAX_SPLIT_CONDUCTANCE_BP`]).
    #[must_use]
    pub fn recommend(&self) -> CoherenceDecision {
        if let Some((partition, pressure)) = self.best_split_candidate() {
            if self.split_allowed(partition, pressure) {
                return CoherenceDecision::SplitRecommended {
                    partition,
                    pressure,
                };
            }
            // Candidate exceeded the pressure threshold but no usable
            // low-conductance boundary exists: fall through to merge
            // evaluation rather than splitting blindly.
        }

        // Check for merge candidates among all pairs
        let mut best_merge: Option<MergeSignal> = None;
        let active_entries: [Option<PartitionId>; ENGINE_MAX_NODES] = {
            let mut arr = [None; ENGINE_MAX_NODES];
            for (i, entry) in self.entries.iter().enumerate() {
                if entry.active {
                    arr[i] = Some(entry.id);
                }
            }
            arr
        };

        for i in 0..ENGINE_MAX_NODES {
            let a = match active_entries[i] {
                Some(id) => id,
                None => continue,
            };
            for j in (i + 1)..ENGINE_MAX_NODES {
                let b = match active_entries[j] {
                    Some(id) => id,
                    None => continue,
                };
                // Non-adjacent pairs can never merge: with zero connecting
                // weight, `evaluate_merge` always yields mutual_bp == 0,
                // which is below the merge threshold. Skip them via the
                // O(1) adjacency-matrix lookup.
                if self.graph.edge_weight_between(a, b) == 0 {
                    continue;
                }
                let signal = pressure::evaluate_merge(a, b, &self.graph);
                if signal.should_merge {
                    match best_merge {
                        None => best_merge = Some(signal),
                        Some(ref prev)
                            if signal.mutual_coherence > prev.mutual_coherence =>
                        {
                            best_merge = Some(signal);
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(signal) = best_merge {
            return CoherenceDecision::MergeRecommended {
                a: signal.partition_a,
                b: signal.partition_b,
                mutual_coherence: signal.mutual_coherence,
            };
        }

        CoherenceDecision::NoAction
    }

    /// The partition with the highest cut pressure above the split
    /// threshold, if any.
    fn best_split_candidate(&self) -> Option<(PartitionId, CutPressure)> {
        let mut best: Option<(PartitionId, CutPressure)> = None;
        for entry in &self.entries {
            if !entry.active {
                continue;
            }
            if entry.pressure.as_fixed() > SPLIT_THRESHOLD_BP {
                match best {
                    None => best = Some((entry.id, entry.pressure)),
                    Some((_, prev)) if entry.pressure > prev => {
                        best = Some((entry.id, entry.pressure));
                    }
                    _ => {}
                }
            }
        }
        best
    }

    /// Apply the split policy: critical pressure always splits; mid-band
    /// pressure splits only with a fresh, usable, low-conductance plan.
    fn split_allowed(&self, partition: PartitionId, pressure: CutPressure) -> bool {
        if pressure.as_fixed() >= CRITICAL_PRESSURE_BP {
            return true;
        }
        match &self.split_plan {
            Some(plan) => plan.partition == partition && plan.is_low_conductance(),
            None => false,
        }
    }

    /// EPOCH TASK: compute the mincut boundary for `candidate` and
    /// derive its conductance. Runs the (budgeted) exact mincut -- never
    /// call this from the partition-switch hot path.
    fn build_split_plan(&mut self, candidate: PartitionId) -> SplitPlan {
        let cut: BackendMinCutResult =
            self.mincut_backend.find_min_cut(&self.graph, candidate);

        // The side containing the candidate stays with the source.
        let candidate_on_left = cut.left[..(cut.left_count as usize).min(ENGINE_MAX_NODES)]
            .contains(&Some(candidate));

        let mut plan = SplitPlan {
            partition: candidate,
            keep: [None; ENGINE_MAX_NODES],
            keep_count: 0,
            move_out: [None; ENGINE_MAX_NODES],
            move_count: 0,
            cut_weight: cut.cut_weight,
            conductance_bp: 10_000,
            within_budget: cut.within_budget,
            epoch: self.epoch,
        };

        let (keep_src, keep_n, move_src, move_n) = if candidate_on_left {
            (&cut.left, cut.left_count, &cut.right, cut.right_count)
        } else {
            (&cut.right, cut.right_count, &cut.left, cut.left_count)
        };

        let mut vol_keep = 0u64;
        for slot in keep_src[..(keep_n as usize).min(ENGINE_MAX_NODES)].iter() {
            if let Some(pid) = slot {
                plan.keep[plan.keep_count as usize] = Some(*pid);
                plan.keep_count += 1;
                vol_keep = vol_keep.saturating_add(self.graph.total_weight(*pid));
            }
        }
        let mut vol_move = 0u64;
        for slot in move_src[..(move_n as usize).min(ENGINE_MAX_NODES)].iter() {
            if let Some(pid) = slot {
                plan.move_out[plan.move_count as usize] = Some(*pid);
                plan.move_count += 1;
                vol_move = vol_move.saturating_add(self.graph.total_weight(*pid));
            }
        }

        let min_vol = vol_keep.min(vol_move);
        plan.conductance_bp = if plan.keep_count == 0 || plan.move_count == 0 || min_vol == 0 {
            10_000 // no usable two-sided cut
        } else {
            (((plan.cut_weight as u128) * 10_000 / (min_vol as u128)) as u32).min(10_000)
        };
        plan
    }

    /// Seed the hot-path Fennel placer from an epoch plan: keep side 0,
    /// move side 1.
    fn seed_placer_from_plan(&mut self, plan: &SplitPlan) {
        self.placer.clear();
        for slot in plan.keep[..plan.keep_count as usize].iter() {
            if let Some(pid) = slot {
                let _ = self.placer.seed(*pid, 0);
            }
        }
        for slot in plan.move_out[..plan.move_count as usize].iter() {
            if let Some(pid) = slot {
                let _ = self.placer.seed(*pid, 1);
            }
        }
    }

    /// The cached epoch split plan, if one was computed.
    #[must_use]
    pub fn split_plan(&self) -> Option<&SplitPlan> {
        self.split_plan.as_ref()
    }

    /// Get a split plan for `source`, reusing the cached epoch plan when
    /// it is fresh (matching partition, no graph mutations since it was
    /// computed) and recomputing otherwise.
    ///
    /// This is the entry point for the split executor: it guarantees the
    /// returned boundary reflects the current graph.
    pub fn split_plan_for(&mut self, source: PartitionId) -> SplitPlan {
        if self.cache_valid {
            if let Some(plan) = &self.split_plan {
                if plan.partition == source {
                    return plan.clone();
                }
            }
        }
        let plan = self.build_split_plan(source);
        self.seed_placer_from_plan(&plan);
        self.split_plan = Some(plan.clone());
        plan
    }

    /// HOT PATH: place a newly arrived partition on a boundary side in
    /// O(degree), without recomputing the mincut.
    ///
    /// Returns `None` when no epoch plan exists (nothing to place
    /// against). The placement is advisory and is refreshed when the
    /// next epoch recomputation produces a new plan.
    pub fn place_incremental(&mut self, pid: PartitionId) -> Option<SplitSide> {
        self.split_plan.as_ref()?;
        let side = self.placer.place(&self.graph, pid);
        Some(if side == 0 {
            SplitSide::Keep
        } else {
            SplitSide::MoveToChild
        })
    }

    /// Access the hot-path Fennel placer (for inspection/testing).
    #[must_use]
    pub fn fennel_placer(&self) -> &FennelPlacer {
        &self.placer
    }

    /// Apply a split boundary: re-home every `move_out` neighbor's
    /// communication edges from `source` to `child` so the post-split
    /// topology matches the computed mincut.
    ///
    /// Returns the number of neighbors whose edges were re-homed. The
    /// consumed plan is dropped (the next epoch recomputes it).
    pub fn apply_split_boundary(
        &mut self,
        source: PartitionId,
        child: PartitionId,
        plan: &SplitPlan,
    ) -> Result<u16, RvmError> {
        if self.graph.find_node(source).is_none() || self.graph.find_node(child).is_none() {
            return Err(RvmError::PartitionNotFound);
        }
        self.cache_valid = false;

        let mut moved = 0u16;
        for slot in plan.move_out[..(plan.move_count as usize).min(ENGINE_MAX_NODES)].iter() {
            let Some(neighbor) = slot else { continue };
            let neighbor = *neighbor;
            if neighbor == source || neighbor == child {
                continue;
            }
            let out_w = self
                .graph
                .remove_directed_edges(source, neighbor)
                .unwrap_or(0);
            let in_w = self
                .graph
                .remove_directed_edges(neighbor, source)
                .unwrap_or(0);
            if out_w > 0 {
                let _ = self.graph.add_edge(child, neighbor, out_w);
            }
            if in_w > 0 {
                let _ = self.graph.add_edge(neighbor, child, in_w);
            }
            if out_w > 0 || in_w > 0 {
                moved += 1;
            }
        }

        // The boundary has been consumed; drop the stale plan.
        self.split_plan = None;
        Ok(moved)
    }

    /// Access the underlying coherence graph (for inspection/testing).
    #[must_use]
    pub fn graph(&self) -> &CoherenceGraph<ENGINE_MAX_NODES, ENGINE_MAX_EDGES> {
        &self.graph
    }

    /// Access the adaptive engine (for inspection/testing).
    #[must_use]
    pub fn adaptive(&self) -> &AdaptiveCoherenceEngine {
        &self.adaptive
    }

    /// The name of the active mincut backend.
    #[must_use]
    pub fn mincut_backend_name(&self) -> &'static str {
        self.mincut_backend.backend_name()
    }

    /// The name of the active coherence scoring backend.
    #[must_use]
    pub fn coherence_backend_name(&self) -> &'static str {
        self.coherence_backend.backend_name()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PartitionId {
        PartitionId::new(n)
    }

    #[test]
    fn engine_creation_defaults() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.epoch(), 0);
        assert_eq!(engine.partition_count(), 0);
        assert_eq!(engine.mincut_backend_name(), "stoer-wagner-builtin");
        assert_eq!(engine.coherence_backend_name(), "ratio-builtin");
    }

    #[test]
    fn add_and_remove_partitions() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);

        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        assert_eq!(engine.partition_count(), 2);

        engine.remove_partition(pid(1)).unwrap();
        assert_eq!(engine.partition_count(), 1);
    }

    #[test]
    fn duplicate_partition_rejected() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        assert_eq!(
            engine.add_partition(pid(1)),
            Err(RvmError::InvalidPartitionState)
        );
    }

    #[test]
    fn remove_nonexistent_partition_fails() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(
            engine.remove_partition(pid(99)),
            Err(RvmError::PartitionNotFound)
        );
    }

    #[test]
    fn record_communication_creates_edge() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        engine.record_communication(pid(1), pid(2), 500).unwrap();
        assert_eq!(engine.graph().edge_count(), 1);

        // Second call increments weight rather than creating new edge
        engine.record_communication(pid(1), pid(2), 300).unwrap();
        assert_eq!(engine.graph().edge_count(), 1);
    }

    #[test]
    fn record_communication_to_unknown_partition_fails() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        assert_eq!(
            engine.record_communication(pid(1), pid(99), 100),
            Err(RvmError::PartitionNotFound)
        );
    }

    #[test]
    fn tick_advances_epoch() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();

        assert_eq!(engine.epoch(), 0);
        engine.tick(20);
        assert_eq!(engine.epoch(), 1);
        engine.tick(20);
        assert_eq!(engine.epoch(), 2);
    }

    #[test]
    fn score_after_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // Before tick, score is the initial MAX
        assert_eq!(engine.score(pid(1)), CoherenceScore::MAX);

        // After tick at low load, scores are recomputed
        engine.tick(10);

        // pid(1) has external-only edges, so score should be 0
        assert_eq!(engine.score(pid(1)).as_basis_points(), 0);
    }

    #[test]
    fn pressure_after_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        engine.tick(10);

        // pid(1) has fully external traffic => max pressure
        assert_eq!(engine.pressure(pid(1)).as_fixed(), 10_000);
    }

    #[test]
    fn split_recommended_for_high_pressure() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        let decision = engine.tick(10);

        match decision {
            CoherenceDecision::SplitRecommended { partition, pressure } => {
                // Either pid(1) or pid(2) should be recommended for split
                assert!(partition == pid(1) || partition == pid(2));
                assert!(pressure.as_fixed() > SPLIT_THRESHOLD_BP);
            }
            _ => panic!("expected SplitRecommended"),
        }
    }

    #[test]
    fn no_action_for_isolated_partitions() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        // No communication recorded

        let decision = engine.tick(10);
        assert_eq!(decision, CoherenceDecision::NoAction);
    }

    #[test]
    fn recommend_without_tick() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        // No edges, no pressure
        assert_eq!(engine.recommend(), CoherenceDecision::NoAction);
    }

    #[test]
    fn score_of_unknown_partition_returns_max() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.score(pid(99)), CoherenceScore::MAX);
    }

    #[test]
    fn pressure_of_unknown_partition_returns_zero() {
        let engine = DefaultCoherenceEngine::with_defaults(100);
        assert_eq!(engine.pressure(pid(99)), CutPressure::ZERO);
    }

    #[test]
    fn adaptive_skips_under_high_load() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // First tick at high load -- always computes on first epoch
        let _ = engine.tick(90);
        assert_eq!(engine.epoch(), 1);
        // Score should have been computed
        assert_eq!(engine.score(pid(1)).as_basis_points(), 0);

        // Next 3 ticks at high load should skip recomputation
        // (interval = 4 at >80% load). Scores stay the same.
        let _ = engine.tick(90);
        let _ = engine.tick(90);
        let _ = engine.tick(90);
        assert_eq!(engine.epoch(), 4);
    }

    #[test]
    fn skip_tick_returns_cached_decision() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        // First tick at high load always computes (split expected).
        let first = engine.tick(90);
        assert!(matches!(
            first,
            CoherenceDecision::SplitRecommended { .. }
        ));

        // Subsequent skip ticks (interval = 4 at >80% load) must return
        // the same cached decision without recomputation.
        assert_eq!(engine.tick(90), first);
        assert_eq!(engine.tick(90), first);
    }

    #[test]
    fn graph_mutation_invalidates_decision_cache() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        // First tick at high load computes: no edges => NoAction.
        assert_eq!(engine.tick(90), CoherenceDecision::NoAction);

        // Mutate the graph with heavy mutual traffic; the next skip tick
        // must not return the stale cached NoAction.
        engine.record_communication(pid(1), pid(2), 8000).unwrap();
        engine.record_communication(pid(2), pid(1), 8000).unwrap();

        let decision = engine.tick(90); // skip tick, cache invalidated
        assert!(matches!(
            decision,
            CoherenceDecision::MergeRecommended { .. }
        ));
    }

    #[test]
    fn record_communication_huge_weight_does_not_decrease_edge() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();

        // Create the edge, then record a weight >= 2^63 on the update
        // path. A naive `as i64` cast would make the delta negative and
        // shrink the edge weight.
        engine.record_communication(pid(1), pid(2), 100).unwrap();
        engine
            .record_communication(pid(1), pid(2), u64::MAX)
            .unwrap();

        let w = engine.graph().edge_weight_between(pid(1), pid(2));
        // Clamped delta: 100 + i64::MAX.
        assert_eq!(w, 100u64 + i64::MAX as u64);
        assert!(w > 100);
    }

    // -------------------------------------------------------------------
    // Split decision policy tests (pressure ratio AND cut quality)
    // -------------------------------------------------------------------

    /// S has mid-band pressure (8000 < p < 9500) and a weakly attached
    /// sub-cluster {X} whose own cluster mass makes the cut low
    /// conductance. Build: S self-loop 300, S<->A 600 each way,
    /// S<->X 100 each way, X<->Y 2000 each way, plus self-loops on
    /// A/X/Y to keep their pressures below the threshold so S is the
    /// sole candidate.
    fn mid_band_engine_with_cluster() -> DefaultCoherenceEngine {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        for n in [1, 2, 3, 4] {
            engine.add_partition(pid(n)).unwrap();
        }
        let (s, a, x, y) = (pid(1), pid(2), pid(3), pid(4));
        engine.record_communication(s, s, 300).unwrap();
        engine.record_communication(s, a, 600).unwrap();
        engine.record_communication(a, s, 600).unwrap();
        engine.record_communication(s, x, 100).unwrap();
        engine.record_communication(x, s, 100).unwrap();
        engine.record_communication(x, y, 2000).unwrap();
        engine.record_communication(y, x, 2000).unwrap();
        engine.record_communication(a, a, 2000).unwrap();
        engine.record_communication(x, x, 3000).unwrap();
        engine.record_communication(y, y, 3000).unwrap();
        engine
    }

    #[test]
    fn mid_band_pressure_with_low_conductance_cut_splits() {
        let mut engine = mid_band_engine_with_cluster();
        let decision = engine.tick(10);

        // S: total = 2000, internal = 300 -> pressure 8500 (mid-band).
        let p = engine.pressure(pid(1)).as_fixed();
        assert!(p > SPLIT_THRESHOLD_BP && p < CRITICAL_PRESSURE_BP, "p={p}");

        // The mincut peels {X} off {S, A}: cut = 200, conductance well
        // below the gate -> split is allowed at mid-band pressure.
        match decision {
            CoherenceDecision::SplitRecommended { partition, .. } => {
                assert_eq!(partition, pid(1));
            }
            other => panic!("expected SplitRecommended, got {other:?}"),
        }
        let plan = engine.split_plan().expect("epoch plan must be cached");
        assert_eq!(plan.partition, pid(1));
        assert!(plan.is_low_conductance(), "conductance={}", plan.conductance_bp);
        // S<->X is 100 each way = 200, decayed 5% by the tick -> 190.
        assert_eq!(plan.cut_weight, 190);
        // X is on the move side; A stays with S.
        let in_move = |p: PartitionId, plan: &SplitPlan| {
            plan.move_out[..plan.move_count as usize].contains(&Some(p))
        };
        assert!(in_move(pid(3), plan));
        assert!(!in_move(pid(2), plan));
    }

    #[test]
    fn mid_band_pressure_with_poor_cut_does_not_split() {
        // Same shape but X is a leaf (no X<->Y cluster): cutting X off
        // crosses its entire volume -> conductance 10000 -> no split.
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        for n in [1, 2, 3] {
            engine.add_partition(pid(n)).unwrap();
        }
        let (s, a, x) = (pid(1), pid(2), pid(3));
        engine.record_communication(s, s, 300).unwrap();
        engine.record_communication(s, a, 600).unwrap();
        engine.record_communication(a, s, 600).unwrap();
        engine.record_communication(s, x, 100).unwrap();
        engine.record_communication(x, s, 100).unwrap();
        engine.record_communication(a, a, 2000).unwrap();
        engine.record_communication(x, x, 50).unwrap();

        let decision = engine.tick(10);

        // S still has mid-band pressure...
        let p = engine.pressure(pid(1)).as_fixed();
        assert!(p > SPLIT_THRESHOLD_BP && p < CRITICAL_PRESSURE_BP, "p={p}");
        // ...but the only cut is high conductance, so no split fires.
        assert_eq!(decision, CoherenceDecision::NoAction);
        let plan = engine.split_plan().expect("plan computed for the candidate");
        assert!(!plan.is_low_conductance());
    }

    #[test]
    fn critical_pressure_splits_without_cut_quality() {
        // Fully external traffic -> pressure 10000 >= CRITICAL: the
        // safety valve fires even though the cut is poor (leaf cut).
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 1000).unwrap();

        let decision = engine.tick(10);
        assert!(matches!(
            decision,
            CoherenceDecision::SplitRecommended { .. }
        ));
    }

    // -------------------------------------------------------------------
    // Epoch task vs hot path integration
    // -------------------------------------------------------------------

    #[test]
    fn epoch_plan_then_hotpath_placement_without_recompute() {
        let mut engine = mid_band_engine_with_cluster();

        // EPOCH TASK: tick computes the mincut plan.
        let _ = engine.tick(10);
        let plan_epoch = engine.split_plan().expect("plan").epoch;

        // HOT PATH: a new partition arrives, talking to the move-side
        // cluster member X. Incremental Fennel placement must route it
        // to the move side using only O(degree) work...
        let z = pid(9);
        engine.add_partition(z).unwrap();
        engine.record_communication(z, pid(3), 500).unwrap();
        engine.record_communication(pid(3), z, 500).unwrap();

        let side = engine.place_incremental(z).expect("plan exists");
        assert_eq!(side, SplitSide::MoveToChild);

        // ...without recomputing the epoch mincut: the cached plan is
        // untouched (same epoch stamp, same partition).
        let plan = engine.split_plan().expect("plan survives hot path");
        assert_eq!(plan.epoch, plan_epoch);
        assert_eq!(plan.partition, pid(1));

        // A second arrival with affinity to the keep side goes there.
        let w = pid(10);
        engine.add_partition(w).unwrap();
        engine.record_communication(w, pid(2), 500).unwrap();
        engine.record_communication(pid(2), w, 500).unwrap();
        assert_eq!(engine.place_incremental(w), Some(SplitSide::Keep));
    }

    #[test]
    fn place_incremental_without_plan_returns_none() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        assert_eq!(engine.place_incremental(pid(1)), None);
    }

    // -------------------------------------------------------------------
    // Split boundary application
    // -------------------------------------------------------------------

    #[test]
    fn apply_split_boundary_rehomes_move_side_edges() {
        let mut engine = mid_band_engine_with_cluster();
        let (s, a, x) = (pid(1), pid(2), pid(3));

        let plan = engine.split_plan_for(s);
        assert!(plan.is_usable());

        // Create the child and apply the boundary.
        let child = pid(7);
        engine.add_partition(child).unwrap();
        let moved = engine.apply_split_boundary(s, child, &plan).unwrap();
        assert_eq!(moved, 1, "exactly X is re-homed");

        // X's edges now attach to the child, not the source.
        assert_eq!(engine.graph().edge_weight_between(s, x), 0);
        assert_eq!(engine.graph().edge_weight_between(child, x), 200);
        // The keep side stays with the source.
        assert_eq!(engine.graph().edge_weight_between(s, a), 1200);
        assert_eq!(engine.graph().edge_weight_between(child, a), 0);
        // The consumed plan is dropped.
        assert!(engine.split_plan().is_none());
    }

    #[test]
    fn apply_split_boundary_missing_child_fails() {
        let mut engine = mid_band_engine_with_cluster();
        let plan = engine.split_plan_for(pid(1));
        assert_eq!(
            engine.apply_split_boundary(pid(1), pid(99), &plan),
            Err(RvmError::PartitionNotFound)
        );
    }

    #[test]
    fn split_plan_for_reuses_fresh_cached_plan() {
        let mut engine = mid_band_engine_with_cluster();
        let _ = engine.tick(10); // computes + caches the plan
        let cached_epoch = engine.split_plan().unwrap().epoch;

        // No graph mutations since the tick: the cached plan is reused.
        let plan = engine.split_plan_for(pid(1));
        assert_eq!(plan.epoch, cached_epoch);

        // After a mutation, the plan is recomputed against fresh state:
        // the tick decayed S<->X from 100+100 to 95+95; adding 50 to
        // S->X makes the cut 145 + 95 = 240.
        engine.record_communication(pid(1), pid(3), 50).unwrap();
        let plan2 = engine.split_plan_for(pid(1));
        assert_eq!(plan2.cut_weight, 240);
    }

    #[test]
    fn split_plan_for_isolated_partition_is_unusable() {
        let mut engine = DefaultCoherenceEngine::with_defaults(100);
        engine.add_partition(pid(1)).unwrap();
        let plan = engine.split_plan_for(pid(1));
        assert!(!plan.is_usable());
        assert_eq!(plan.move_count, 0);
        assert_eq!(plan.conductance_bp, 10_000);
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_engine_creation() {
        let engine = RuVectorCoherenceEngine::with_ruvector(100);
        assert_eq!(engine.mincut_backend_name(), "ruvector-mincut-stub");
        assert_eq!(engine.coherence_backend_name(), "ruvector-spectral-stub");
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_engine_lifecycle() {
        let mut engine = RuVectorCoherenceEngine::with_ruvector(100);
        engine.add_partition(pid(1)).unwrap();
        engine.add_partition(pid(2)).unwrap();
        engine.record_communication(pid(1), pid(2), 500).unwrap();

        let decision = engine.tick(10);
        // With only external traffic, should recommend split
        match decision {
            CoherenceDecision::SplitRecommended { .. } => {}
            _ => panic!("expected SplitRecommended from ruvector engine"),
        }
    }

    #[cfg(feature = "ruvector")]
    #[test]
    fn ruvector_matches_builtin_results() {
        // Since the ruvector stubs delegate to the builtin, results
        // should be identical.
        let mut default_engine = DefaultCoherenceEngine::with_defaults(100);
        let mut rv_engine = RuVectorCoherenceEngine::with_ruvector(100);

        for engine in [&mut default_engine as &mut dyn EngineOps, &mut rv_engine] {
            engine.add_p(pid(1)).unwrap();
            engine.add_p(pid(2)).unwrap();
            engine.record(pid(1), pid(2), 1000).unwrap();
            engine.do_tick(10);
        }

        assert_eq!(
            default_engine.score(pid(1)),
            rv_engine.score(pid(1))
        );
        assert_eq!(
            default_engine.pressure(pid(1)),
            rv_engine.pressure(pid(1))
        );
    }
}

// Helper trait for the ruvector_matches_builtin_results test
#[cfg(all(test, feature = "ruvector"))]
trait EngineOps {
    fn add_p(&mut self, id: PartitionId) -> Result<(), RvmError>;
    fn record(&mut self, from: PartitionId, to: PartitionId, w: u64) -> Result<(), RvmError>;
    fn do_tick(&mut self, load: u8) -> CoherenceDecision;
}

#[cfg(all(test, feature = "ruvector"))]
impl<MCB: MinCutBackend, CB: CoherenceBackend> EngineOps for CoherenceEngine<MCB, CB> {
    fn add_p(&mut self, id: PartitionId) -> Result<(), RvmError> {
        self.add_partition(id)
    }
    fn record(&mut self, from: PartitionId, to: PartitionId, w: u64) -> Result<(), RvmError> {
        self.record_communication(from, to, w)
    }
    fn do_tick(&mut self, load: u8) -> CoherenceDecision {
        self.tick(load)
    }
}
