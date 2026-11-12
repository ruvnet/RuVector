//! The governed placement planner (PIR WP20, ruvector ADR-323).
//!
//! Given a set of pipeline-stage [`ModelShard`]s and a heterogeneous fleet of
//! [`FleetNode`]s, [`PlacementPlanner::plan`] finds the **minimum
//! latency/cost** assignment of shards → nodes that satisfies every hard
//! governance constraint (residency, trust, hardware capability, and
//! cumulative memory capacity), or **rejects** the request with a specific,
//! structured [`PlacementRejection`] when no feasible assignment exists.
//!
//! ## Method
//!
//! Hard constraints are inviolable, so the planner first computes, per shard,
//! the set of nodes that could ever host it (residency/trust/capability, plus
//! per-shard memory fit on a node's full capacity). If any shard has *no*
//! eligible node, the request is rejected immediately with a per-node reason.
//!
//! Cumulative memory capacity (one node hosting several shards) couples the
//! choices, so the optimum is found by depth-first branch-and-bound over
//! feasible complete assignments: candidate nodes are visited cheapest-first,
//! a node's residual memory is decremented as shards are placed, and branches
//! whose optimistic lower bound cannot beat the best complete assignment found
//! so far are pruned. This yields the exact minimum-cost feasible placement
//! for the small fleets this planner targets (a handful of stages across a
//! handful of nodes), while still proving the governance property: an
//! infeasible request is never silently satisfied on a violating node.

use super::policy::{GovernancePolicy, NodeViolation};
use super::{FleetNode, ModelShard, NodeId, ShardId};
use std::collections::BTreeMap;

/// Maximum shards a single placement request may contain (PIR WP20 defensive
/// input bound, ruvector ADR-323, #867).
///
/// The branch-and-bound [`search`] recurses to depth = shard count, so an
/// unbounded shard count is an unbounded native call stack. Every pipeline this
/// planner governs is a handful of stages (the source paper's largest is a
/// four-node/70B pipeline); 64 sits far above any real pipeline while capping
/// the recursion depth. A request above this cap is rejected up front with
/// [`PlacementRejection::InputTooLarge`] rather than risking stack exhaustion.
pub const MAX_SHARDS: usize = 64;

/// Maximum fleet nodes a single placement request may contain (PIR WP20
/// defensive input bound, ruvector ADR-323, #867).
///
/// The search fans out over each shard's eligible nodes, so node count bounds
/// the per-level search breadth (and, with [`MAX_SHARDS`], the combinatorial
/// worst case). Governed fleets are handfuls of nodes; 256 is far above any
/// real fleet while keeping the search breadth bounded. A request above this
/// cap is rejected up front with [`PlacementRejection::InputTooLarge`].
pub const MAX_NODES: usize = 256;

/// Relative weighting of latency versus cost in the placement objective.
/// Lower objective is better. Both default to `1.0`.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveWeights {
    /// Weight applied to each node's `latency_ms`.
    pub latency: f64,
    /// Weight applied to each node's `cost_units`.
    pub cost: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            latency: 1.0,
            cost: 1.0,
        }
    }
}

/// A single shard-to-node assignment in a finished plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardAssignment {
    /// The placed shard.
    pub shard: ShardId,
    /// Its pipeline stage index.
    pub stage_index: usize,
    /// The node it was placed on.
    pub node: NodeId,
}

/// A complete, governance-respecting placement of every shard.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    /// One assignment per shard, in the order the shards were supplied.
    pub assignments: Vec<ShardAssignment>,
    /// Total latency/cost objective of this placement (lower is better).
    pub total_objective: f64,
    /// Memory consumed per node by this placement, in MB.
    pub memory_used_mb: BTreeMap<NodeId, u64>,
}

impl PlacementPlan {
    /// The node a given shard was placed on, if present in this plan.
    pub fn node_for(&self, shard: &ShardId) -> Option<&NodeId> {
        self.assignments
            .iter()
            .find(|a| &a.shard == shard)
            .map(|a| &a.node)
    }
}

/// Why a placement request could not be satisfied. A rejection is always
/// returned instead of a plan that would violate a hard constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementRejection {
    /// A shard has no individually-eligible node: every node in the fleet
    /// violates at least one hard constraint for it. `violations` records each
    /// node's (first) violation so the reason is specific and actionable.
    NoEligibleNode {
        /// The shard that cannot be placed anywhere.
        shard: ShardId,
        /// Per-node reasons (empty only when the fleet itself is empty).
        violations: Vec<NodeViolation>,
    },
    /// The request exceeds a documented input-size cap ([`MAX_SHARDS`] /
    /// [`MAX_NODES`]) and is rejected *before* the branch-and-bound search runs.
    /// The search recurses to depth = shard count and fans out over eligible
    /// nodes per shard, so a pathologically large request could exhaust the
    /// native stack or blow up combinatorially; bounding the input closes both.
    InputTooLarge {
        /// Number of shards supplied in the request.
        shards: usize,
        /// Number of nodes supplied in the request.
        nodes: usize,
        /// The shard-count cap that was exceeded ([`MAX_SHARDS`]).
        max_shards: usize,
        /// The node-count cap that was exceeded ([`MAX_NODES`]).
        max_nodes: usize,
    },
    /// Every shard is individually placeable, but no *complete* assignment fits
    /// within the fleet's cumulative memory capacity (the shards collectively
    /// exceed what the eligible nodes can jointly hold).
    CapacityExhausted {
        /// A shard the search could not fit given the others' placement.
        shard: ShardId,
        /// That shard's memory requirement, in MB.
        required_mb: u64,
        /// Total memory advertised across the fleet, in MB.
        fleet_available_mb: u64,
    },
}

impl std::fmt::Display for PlacementRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementRejection::NoEligibleNode { shard, violations } => {
                write!(
                    f,
                    "shard '{shard}' has no eligible node ({} candidate node(s) rejected): ",
                    violations.len()
                )?;
                let joined = violations
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                f.write_str(&joined)
            }
            PlacementRejection::InputTooLarge {
                shards,
                nodes,
                max_shards,
                max_nodes,
            } => write!(
                f,
                "placement request too large: {shards} shard(s) (max {max_shards}), \
                 {nodes} node(s) (max {max_nodes}) — rejected before the bounded search to \
                 avoid stack exhaustion / combinatorial blow-up"
            ),
            PlacementRejection::CapacityExhausted {
                shard,
                required_mb,
                fleet_available_mb,
            } => write!(
                f,
                "no feasible placement: shard '{shard}' ({required_mb} MB) does not fit given \
                 cumulative fleet capacity ({fleet_available_mb} MB total across eligible nodes)"
            ),
        }
    }
}

impl std::error::Error for PlacementRejection {}

/// The governed placement planner.
#[derive(Debug, Clone)]
pub struct PlacementPlanner {
    weights: ObjectiveWeights,
}

impl Default for PlacementPlanner {
    fn default() -> Self {
        Self {
            weights: ObjectiveWeights::default(),
        }
    }
}

impl PlacementPlanner {
    /// Construct a planner with explicit objective weights.
    pub fn new(weights: ObjectiveWeights) -> Self {
        Self { weights }
    }

    /// Plan a placement of `shards` onto `nodes`.
    ///
    /// Returns the minimum-objective placement respecting all hard constraints,
    /// or a [`PlacementRejection`] describing precisely why no feasible
    /// placement exists. Shards are placed in the order supplied; the returned
    /// `assignments` preserve that order.
    pub fn plan(
        &self,
        shards: &[ModelShard],
        nodes: &[FleetNode],
    ) -> Result<PlacementPlan, PlacementRejection> {
        // Trivially satisfiable: nothing to place (bounded regardless of fleet
        // size — depth-0 search, no fan-out).
        if shards.is_empty() {
            return Ok(PlacementPlan {
                assignments: Vec::new(),
                total_objective: 0.0,
                memory_used_mb: BTreeMap::new(),
            });
        }

        // Defensive input-size bound (PIR WP20, #867): the branch-and-bound
        // search below recurses to depth = shards.len() and fans out over each
        // shard's eligible nodes, so a pathologically large request could
        // exhaust the native stack or blow up combinatorially. Reject over-cap
        // inputs here, before any recursion, with a specific structured reason.
        if shards.len() > MAX_SHARDS || nodes.len() > MAX_NODES {
            return Err(PlacementRejection::InputTooLarge {
                shards: shards.len(),
                nodes: nodes.len(),
                max_shards: MAX_SHARDS,
                max_nodes: MAX_NODES,
            });
        }

        // Per-node objective cost (shard-independent in this model).
        let node_cost: Vec<f64> = nodes
            .iter()
            .map(|n| self.weights.latency * n.latency_ms + self.weights.cost * n.cost_units)
            .collect();

        // Per-shard eligible-node lists (hard constraints against full node
        // capacity), sorted cheapest-first for a deterministic optimal search.
        let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(shards.len());
        for shard in shards {
            let mut eligible: Vec<usize> = Vec::new();
            for (ni, node) in nodes.iter().enumerate() {
                if GovernancePolicy::is_eligible(shard, node).is_ok() {
                    eligible.push(ni);
                }
            }
            if eligible.is_empty() {
                // Governance: reject with a specific per-node reason rather
                // than placing the shard on any violating node.
                let violations = nodes
                    .iter()
                    .map(|node| NodeViolation {
                        node: node.id.clone(),
                        // Safe: `is_eligible` returned Err for every node here.
                        violation: GovernancePolicy::is_eligible(shard, node)
                            .expect_err("node was not eligible"),
                    })
                    .collect();
                return Err(PlacementRejection::NoEligibleNode {
                    shard: shard.id.clone(),
                    violations,
                });
            }
            eligible.sort_by(|&a, &b| {
                node_cost[a]
                    .partial_cmp(&node_cost[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            candidates.push(eligible);
        }

        // Optimistic per-shard minimum cost = cheapest eligible node's cost.
        let min_cost: Vec<f64> = candidates.iter().map(|c| node_cost[c[0]]).collect();
        // suffix_min[i] = sum of min_cost[i..], a lower bound on the remaining
        // objective once shard `i` onward are placed.
        let mut suffix_min = vec![0.0f64; shards.len() + 1];
        for i in (0..shards.len()).rev() {
            suffix_min[i] = suffix_min[i + 1] + min_cost[i];
        }

        let mut residual: Vec<u64> = nodes.iter().map(|n| n.available_memory_mb).collect();
        let mut assign: Vec<usize> = vec![usize::MAX; shards.len()];
        let mut best_cost = f64::INFINITY;
        let mut best_assign: Option<Vec<usize>> = None;
        let mut deepest = 0usize;

        search(
            0,
            shards,
            &candidates,
            &node_cost,
            &suffix_min,
            &mut residual,
            &mut assign,
            0.0,
            &mut best_cost,
            &mut best_assign,
            &mut deepest,
        );

        let best = match best_assign {
            Some(b) => b,
            None => {
                // Every shard is individually placeable, but cumulative
                // capacity cannot hold them all — reject, do not overcommit.
                let fleet_available_mb: u64 = nodes.iter().map(|n| n.available_memory_mb).sum();
                let stuck = &shards[deepest.min(shards.len() - 1)];
                return Err(PlacementRejection::CapacityExhausted {
                    shard: stuck.id.clone(),
                    required_mb: stuck.required_memory_mb,
                    fleet_available_mb,
                });
            }
        };

        let mut memory_used_mb: BTreeMap<NodeId, u64> = BTreeMap::new();
        let mut assignments = Vec::with_capacity(shards.len());
        let mut total_objective = 0.0;
        for (i, shard) in shards.iter().enumerate() {
            let ni = best[i];
            let node = &nodes[ni];
            assignments.push(ShardAssignment {
                shard: shard.id.clone(),
                stage_index: shard.stage_index,
                node: node.id.clone(),
            });
            *memory_used_mb.entry(node.id.clone()).or_insert(0) += shard.required_memory_mb;
            total_objective += node_cost[ni];
        }

        Ok(PlacementPlan {
            assignments,
            total_objective,
            memory_used_mb,
        })
    }
}

/// Depth-first branch-and-bound over feasible complete assignments. Places
/// shard `i`, respecting cumulative residual memory, and records the minimum-
/// cost complete assignment in `best_assign`.
#[allow(clippy::too_many_arguments)]
fn search(
    i: usize,
    shards: &[ModelShard],
    candidates: &[Vec<usize>],
    node_cost: &[f64],
    suffix_min: &[f64],
    residual: &mut [u64],
    assign: &mut [usize],
    cur_cost: f64,
    best_cost: &mut f64,
    best_assign: &mut Option<Vec<usize>>,
    deepest: &mut usize,
) {
    if i > *deepest {
        *deepest = i;
    }
    if i == shards.len() {
        if cur_cost < *best_cost {
            *best_cost = cur_cost;
            *best_assign = Some(assign.to_vec());
        }
        return;
    }
    // Prune: even the most optimistic completion cannot beat the incumbent.
    if cur_cost + suffix_min[i] >= *best_cost {
        return;
    }

    let need = shards[i].required_memory_mb;
    for &ni in &candidates[i] {
        if residual[ni] < need {
            continue; // cumulative capacity: this node is already too full
        }
        residual[ni] -= need;
        assign[i] = ni;
        let next_cost = cur_cost + node_cost[ni];
        if next_cost + suffix_min[i + 1] < *best_cost {
            search(
                i + 1,
                shards,
                candidates,
                node_cost,
                suffix_min,
                residual,
                assign,
                next_cost,
                best_cost,
                best_assign,
                deepest,
            );
        }
        residual[ni] += need;
        assign[i] = usize::MAX;
    }
}
