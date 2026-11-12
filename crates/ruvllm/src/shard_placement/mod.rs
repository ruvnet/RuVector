//! Governed Pipeline-Shard Placement (PIR WP20, ruvector ADR-323)
//!
//! This module implements the **governed placement planner** slice of
//! ruvector ADR-323 (which extends ruvector ADR-314, the WP13 KV-cache
//! cross-model migration also living in `crates/ruvllm`). It ports the
//! *placement-governance* half of the pipeline-sharding pattern studied in:
//!
//! - **arXiv:2608.19147**, "Pre-Compiled Pipeline Shards for Distributed LLM
//!   Inference on Intel AI PC Fleets" (submitted 2026-08-19). The paper
//!   reports **two separate results**, which this program is required to keep
//!   distinct (never conflated into one headline number):
//!     1. **1.79× concurrent throughput** — abstract, verbatim: "a two-node
//!        Llama 3.1 8B INT4 pipeline serves two concurrent users at 1.79x the
//!        single-user throughput of the unsplit model on the same hardware."
//!        This figure applies **only** to the two-node / Llama-3.1-8B-INT4
//!        configuration.
//!     2. **Four-node Lunar Lake, 70B, interactive single-user serving** — a
//!        *separate* result (5.72 tok/s single-stream at 72.2% accept rate,
//!        6.43 tok/s aggregate two-stream in the paper's full text). **No
//!        1.79× figure is attached to this configuration.** The relevance to
//!        this slice is purely the *feasibility* shape: a model whose shards
//!        fit collectively across the fleet but not on any single node.
//!
//! ## What this slice is (and is not)
//!
//! Per ADR-323, the RuV-native contribution is **not** distributed inference
//! itself but **governed placement**: deciding which pipeline stage (shard)
//! runs on which fleet node, subject to inviolable governance constraints.
//! This slice ships exactly that planner:
//!
//! - Model a set of [`ModelShard`]s (pipeline stages) and a set of
//!   [`FleetNode`]s with governance attributes (residency zone, trust tier,
//!   available memory, hardware capabilities, plus a latency/cost profile).
//! - A [`planner::PlacementPlanner`] assigns shards → nodes subject to
//!   **hard constraints** — data residency, trust tier, hardware capability,
//!   and (cumulative) memory capacity — optimizing a latency/cost objective
//!   *only among feasible placements*.
//! - An infeasible placement is **rejected with a structured, specific
//!   reason** ([`planner::PlacementRejection`]) — never silently placed on a
//!   constraint-violating node.
//!
//! ### Deliberately deferred (follow-ups, honest scope)
//!
//! Real OpenVINO per-stage shard compilation (including the `beam_idx`-Gather
//! / `IndirectKVCache` fusion detail), actual distributed execution and
//! inter-node transport, speculative decoding on stateful sharded models, and
//! live request-specific KV-state routing are **all out of scope for this
//! slice** — they are the execution runtime ADR-323 governs, tracked as
//! follow-up work. This slice is the placement **planner + governance policy**
//! only.
//!
//! Node trust here is an input attribute ([`TrustTier`]); in production it is
//! sourced from this program's witness chain (ADR-134 / ADR-312), and wiring
//! that verification into the planner is likewise a deferred follow-up.
//!
//! Feature-gated behind `shard-placement` (default off), mirroring WP13's
//! `kv-migration` gate, so it cannot affect the default `ruvllm` build.
//!
//! ## Preprint-reproduction rule
//!
//! The paper's 1.79× and 4-node/70B figures describe the source paper's own
//! hardware, not this program's fleet. They are the *target shape* to
//! reproduce internally, never numbers this program may cite as its own
//! acceptance bar without independently measuring them (ADR-323 §Preprint-
//! reproduction rule; ADR-306 promotion gating).

pub mod planner;
pub mod policy;

#[cfg(test)]
mod tests;

pub use planner::{
    ObjectiveWeights, PlacementPlan, PlacementPlanner, PlacementRejection, ShardAssignment,
};
pub use policy::{ConstraintViolation, GovernancePolicy, NodeViolation};

use std::collections::BTreeSet;

/// A data-residency zone (e.g. `"eu-west"`, `"us"`). A shard whose state is
/// tagged for a zone must be placed on a node in that same zone.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidencyZone(pub String);

impl ResidencyZone {
    /// Construct a residency zone from anything string-like.
    pub fn new(zone: impl Into<String>) -> Self {
        Self(zone.into())
    }

    /// Borrow the zone identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ResidencyZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A trust tier. **Higher is more trusted.** A shard requiring a minimum tier
/// `T` may only be placed on a node whose tier is `>= T`.
///
/// In production the concrete tier of a node is derived from its witness /
/// attestation status (ADR-134 / ADR-312); here it is a plain input attribute
/// so the planner can be tested in isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrustTier(pub u8);

impl std::fmt::Display for TrustTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// Stable identifier for a fleet node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    /// Construct a node id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable identifier for a model shard (pipeline stage).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardId(pub String);

impl ShardId {
    /// Construct a shard id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A node in a heterogeneous serving fleet, with the governance attributes the
/// placement policy reasons over.
#[derive(Debug, Clone)]
pub struct FleetNode {
    /// Stable node identifier.
    pub id: NodeId,
    /// Residency zone this node physically lives in.
    pub residency_zone: ResidencyZone,
    /// Trust tier this node currently satisfies (higher = more trusted).
    pub trust_tier: TrustTier,
    /// Memory available for hosting shards, in MB.
    pub available_memory_mb: u64,
    /// Hardware capability tags this node provides (e.g. `"int4"`, `"avx512"`,
    /// `"npu"`). A shard is only eligible if the node provides *all* of the
    /// shard's required capabilities.
    pub capabilities: BTreeSet<String>,
    /// Serving-latency contribution of this node, in milliseconds — an input
    /// to the placement objective (lower is better).
    pub latency_ms: f64,
    /// Resource/compute cost of using this node — an input to the placement
    /// objective (lower is better).
    pub cost_units: f64,
}

impl FleetNode {
    /// Construct a node with the mandatory governance attributes; latency and
    /// cost default to `0.0` and capabilities to empty. Use the `with_*`
    /// builders to set the objective/capability fields.
    pub fn new(
        id: impl Into<String>,
        residency_zone: ResidencyZone,
        trust_tier: TrustTier,
        available_memory_mb: u64,
    ) -> Self {
        Self {
            id: NodeId::new(id),
            residency_zone,
            trust_tier,
            available_memory_mb,
            capabilities: BTreeSet::new(),
            latency_ms: 0.0,
            cost_units: 0.0,
        }
    }

    /// Set the node's latency/cost objective profile.
    pub fn with_profile(mut self, latency_ms: f64, cost_units: f64) -> Self {
        self.latency_ms = latency_ms;
        self.cost_units = cost_units;
        self
    }

    /// Add hardware capability tags this node provides.
    pub fn with_capabilities<I, S>(mut self, caps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.capabilities.extend(caps.into_iter().map(Into::into));
        self
    }
}

/// A model shard (one pipeline stage) with the governance requirements it
/// imposes on any node that hosts it.
#[derive(Debug, Clone)]
pub struct ModelShard {
    /// Stable shard identifier.
    pub id: ShardId,
    /// Position of this stage in the pipeline (0-based).
    pub stage_index: usize,
    /// Memory this shard requires, in MB.
    pub required_memory_mb: u64,
    /// Minimum trust tier a hosting node must satisfy.
    pub min_trust_tier: TrustTier,
    /// Residency constraint: `Some(zone)` pins the shard to nodes in `zone`;
    /// `None` imposes no residency constraint.
    pub residency: Option<ResidencyZone>,
    /// Hardware capabilities a hosting node must provide (all of them).
    pub required_capabilities: BTreeSet<String>,
}

impl ModelShard {
    /// Construct a shard with no residency constraint, minimum trust tier
    /// `TrustTier(0)`, and no required capabilities. Use the `with_*` builders
    /// to tighten the governance requirements.
    pub fn new(id: impl Into<String>, stage_index: usize, required_memory_mb: u64) -> Self {
        Self {
            id: ShardId::new(id),
            stage_index,
            required_memory_mb,
            min_trust_tier: TrustTier(0),
            residency: None,
            required_capabilities: BTreeSet::new(),
        }
    }

    /// Require a minimum trust tier of any hosting node.
    pub fn with_min_trust(mut self, tier: TrustTier) -> Self {
        self.min_trust_tier = tier;
        self
    }

    /// Pin this shard to a residency zone.
    pub fn with_residency(mut self, zone: ResidencyZone) -> Self {
        self.residency = Some(zone);
        self
    }

    /// Require hardware capabilities of any hosting node.
    pub fn with_required_capabilities<I, S>(mut self, caps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.required_capabilities
            .extend(caps.into_iter().map(Into::into));
        self
    }
}
