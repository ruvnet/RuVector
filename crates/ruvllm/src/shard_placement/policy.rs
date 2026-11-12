//! Governance policy: the hard-constraint eligibility check for placing a
//! single shard on a single node (PIR WP20, ruvector ADR-323).
//!
//! ADR-323 makes the split explicit: **residency, trust, and hardware
//! capability are inviolable hard constraints**; latency/cost only optimize
//! *within* the feasible set. This module owns the hard-constraint check for
//! one `(shard, node)` pair. Cumulative memory capacity (a node hosting
//! several shards) is a hard constraint too, but it is a property of a whole
//! assignment, so the planner enforces it by passing the node's *residual*
//! memory into [`GovernancePolicy::check`].

use super::{FleetNode, ModelShard, NodeId, ResidencyZone, TrustTier};
use std::collections::BTreeSet;

/// A single hard-constraint violation explaining why a node is ineligible to
/// host a shard. Ordered by the check sequence in [`GovernancePolicy::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintViolation {
    /// The shard is pinned to a residency zone the node is not in.
    Residency {
        /// Zone the shard's data is tagged for.
        required: ResidencyZone,
        /// Zone the node actually lives in.
        node_zone: ResidencyZone,
    },
    /// The node's trust tier is below the shard's minimum.
    Trust {
        /// Minimum tier the shard requires.
        required: TrustTier,
        /// Tier the node satisfies.
        node_tier: TrustTier,
    },
    /// The node is missing one or more hardware capabilities the shard needs.
    Capability {
        /// Capabilities the shard requires that the node does not provide.
        missing: BTreeSet<String>,
    },
    /// The (residual) memory available on the node is below the shard's need.
    Capacity {
        /// Memory the shard requires, in MB.
        required_mb: u64,
        /// Memory available on the node at check time, in MB.
        available_mb: u64,
    },
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintViolation::Residency {
                required,
                node_zone,
            } => write!(
                f,
                "residency: shard requires zone '{required}' but node is in '{node_zone}'"
            ),
            ConstraintViolation::Trust {
                required,
                node_tier,
            } => write!(
                f,
                "trust: shard requires tier >= {required} but node is {node_tier}"
            ),
            ConstraintViolation::Capability { missing } => {
                let list = missing.iter().cloned().collect::<Vec<_>>().join(", ");
                write!(f, "capability: node is missing [{list}]")
            }
            ConstraintViolation::Capacity {
                required_mb,
                available_mb,
            } => write!(
                f,
                "capacity: shard needs {required_mb} MB but node has {available_mb} MB available"
            ),
        }
    }
}

/// One node's ineligibility, pairing the node with its (first) violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeViolation {
    /// The ineligible node.
    pub node: NodeId,
    /// Why it is ineligible.
    pub violation: ConstraintViolation,
}

impl std::fmt::Display for NodeViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node '{}': {}", self.node, self.violation)
    }
}

/// The stateless governance policy — a pure hard-constraint checker.
pub struct GovernancePolicy;

impl GovernancePolicy {
    /// Check whether `shard` may be placed on `node` given `available_mb` of
    /// residual memory on that node.
    ///
    /// Returns the **first** violated hard constraint, checked in the order
    /// residency → trust → capability → capacity, or `Ok(())` if the node is
    /// eligible. Residency/trust/capability are node-intrinsic and never
    /// change with load; capacity is checked against the residual memory the
    /// caller supplies so the same routine enforces cumulative capacity during
    /// assignment.
    pub fn check(
        shard: &ModelShard,
        node: &FleetNode,
        available_mb: u64,
    ) -> Result<(), ConstraintViolation> {
        // 1. Residency — inviolable. A shard tagged for a zone must not cross
        //    the boundary onto a node in a different zone.
        if let Some(required) = &shard.residency {
            if &node.residency_zone != required {
                return Err(ConstraintViolation::Residency {
                    required: required.clone(),
                    node_zone: node.residency_zone.clone(),
                });
            }
        }

        // 2. Trust — inviolable. A shard above trust tier T only on nodes >= T.
        if node.trust_tier < shard.min_trust_tier {
            return Err(ConstraintViolation::Trust {
                required: shard.min_trust_tier,
                node_tier: node.trust_tier,
            });
        }

        // 3. Hardware capability — inviolable. The node must provide every
        //    capability the shard requires.
        let missing: BTreeSet<String> = shard
            .required_capabilities
            .difference(&node.capabilities)
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(ConstraintViolation::Capability { missing });
        }

        // 4. Capacity — inviolable. The shard must fit in the residual memory.
        if available_mb < shard.required_memory_mb {
            return Err(ConstraintViolation::Capacity {
                required_mb: shard.required_memory_mb,
                available_mb,
            });
        }

        Ok(())
    }

    /// Convenience check against a node's *full* advertised capacity — the
    /// eligibility test used before any load is assigned (i.e. "could this
    /// shard ever live on this node in isolation?").
    pub fn is_eligible(shard: &ModelShard, node: &FleetNode) -> Result<(), ConstraintViolation> {
        Self::check(shard, node, node.available_memory_mb)
    }
}
