//! Proof tier routing based on operation complexity.
//!
//! Routes mutations to the appropriate proof tier based on:
//! - Operation type (vector update, graph mutation, structural change)
//! - Coherence requirements
//! - Performance constraints

use ruvix_types::ProofTier;

/// Context for routing a proof to the appropriate tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingContext {
    /// Type of mutation being performed.
    pub mutation_type: MutationType,
    /// Whether coherence verification is required.
    pub requires_coherence: bool,
    /// Whether this is a high-frequency operation.
    pub high_frequency: bool,
    /// Number of graph nodes affected (0 for vector ops).
    pub affected_nodes: u32,
    /// Estimated latency budget in microseconds.
    pub latency_budget_us: u32,
}

impl Default for RoutingContext {
    fn default() -> Self {
        Self {
            mutation_type: MutationType::VectorUpdate,
            requires_coherence: false,
            high_frequency: false,
            affected_nodes: 0,
            latency_budget_us: 100,
        }
    }
}

/// Type of mutation being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MutationType {
    /// High-frequency vector update (single vector).
    VectorUpdate = 0,
    /// Batch vector update.
    VectorBatch = 1,
    /// Graph edge insertion/deletion.
    GraphEdge = 2,
    /// Graph node insertion/deletion.
    GraphNode = 3,
    /// Structural change affecting partitions.
    StructuralChange = 4,
    /// Full graph rebalancing.
    GraphRebalance = 5,
}

impl MutationType {
    /// Returns the minimum proof tier for this mutation type.
    #[inline]
    #[must_use]
    pub const fn min_tier(&self) -> ProofTier {
        match self {
            Self::VectorUpdate => ProofTier::Reflex,
            Self::VectorBatch => ProofTier::Reflex,
            Self::GraphEdge => ProofTier::Standard,
            Self::GraphNode => ProofTier::Standard,
            Self::StructuralChange => ProofTier::Deep,
            Self::GraphRebalance => ProofTier::Deep,
        }
    }

    /// Returns the expected operation count for this mutation type.
    #[inline]
    #[must_use]
    pub const fn expected_ops(&self) -> u32 {
        match self {
            Self::VectorUpdate => 1,
            Self::VectorBatch => 16,
            Self::GraphEdge => 2,
            Self::GraphNode => 4,
            Self::StructuralChange => 32,
            Self::GraphRebalance => 128,
        }
    }
}

/// Router for determining proof tier based on context.
#[derive(Debug, Clone, Copy)]
pub struct TierRouter {
    /// Threshold for upgrading Reflex to Standard (affected nodes).
    pub reflex_to_standard_threshold: u32,
    /// Threshold for upgrading Standard to Deep (affected nodes).
    pub standard_to_deep_threshold: u32,
    /// Whether to allow tier downgrades for latency.
    pub allow_downgrade: bool,
}

impl Default for TierRouter {
    fn default() -> Self {
        Self {
            reflex_to_standard_threshold: 10,
            standard_to_deep_threshold: 100,
            allow_downgrade: false,
        }
    }
}

impl TierRouter {
    /// Creates a new tier router with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            reflex_to_standard_threshold: 10,
            standard_to_deep_threshold: 100,
            allow_downgrade: false,
        }
    }

    /// Routes the proof to the appropriate tier.
    #[must_use]
    pub fn route(&self, ctx: &RoutingContext) -> ProofTier {
        // Start with minimum tier for mutation type
        let min_tier = ctx.mutation_type.min_tier();

        // Check if coherence requires Deep tier
        if ctx.requires_coherence {
            return ProofTier::Deep;
        }

        // Route based on affected nodes
        let tier_by_nodes = if ctx.affected_nodes >= self.standard_to_deep_threshold {
            ProofTier::Deep
        } else if ctx.affected_nodes >= self.reflex_to_standard_threshold {
            ProofTier::Standard
        } else {
            ProofTier::Reflex
        };

        // Use the higher of min_tier and tier_by_nodes
        let selected = if (tier_by_nodes as u8) > (min_tier as u8) {
            tier_by_nodes
        } else {
            min_tier
        };

        // Check latency budget and potentially downgrade
        if self.allow_downgrade {
            let max_time = selected.max_verification_time_us();
            if ctx.latency_budget_us < max_time && selected != ProofTier::Reflex {
                // Try to downgrade if latency budget is tight
                match selected {
                    ProofTier::Deep => {
                        if ctx.latency_budget_us >= ProofTier::Standard.max_verification_time_us() {
                            return ProofTier::Standard;
                        }
                    }
                    ProofTier::Standard => {
                        if ctx.latency_budget_us >= ProofTier::Reflex.max_verification_time_us() {
                            return ProofTier::Reflex;
                        }
                    }
                    ProofTier::Reflex => {}
                }
            }
        }

        selected
    }
}

/// Routes a mutation to the appropriate proof tier.
///
/// This is a convenience function for simple routing decisions.
#[must_use]
pub fn route_proof_tier(ctx: &RoutingContext) -> ProofTier {
    TierRouter::default().route(ctx)
}

/// Builder for constructing routing contexts.
#[derive(Debug, Default)]
pub struct RoutingContextBuilder {
    ctx: RoutingContext,
}

impl RoutingContextBuilder {
    /// Creates a new routing context builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ctx: RoutingContext {
                mutation_type: MutationType::VectorUpdate,
                requires_coherence: false,
                high_frequency: false,
                affected_nodes: 0,
                latency_budget_us: 100,
            },
        }
    }

    /// Sets the mutation type.
    #[must_use]
    pub const fn mutation_type(mut self, mutation_type: MutationType) -> Self {
        self.ctx.mutation_type = mutation_type;
        self
    }

    /// Sets whether coherence verification is required.
    #[must_use]
    pub const fn requires_coherence(mut self, requires: bool) -> Self {
        self.ctx.requires_coherence = requires;
        self
    }

    /// Sets whether this is a high-frequency operation.
    #[must_use]
    pub const fn high_frequency(mut self, high_freq: bool) -> Self {
        self.ctx.high_frequency = high_freq;
        self
    }

    /// Sets the number of affected nodes.
    #[must_use]
    pub const fn affected_nodes(mut self, nodes: u32) -> Self {
        self.ctx.affected_nodes = nodes;
        self
    }

    /// Sets the latency budget in microseconds.
    #[must_use]
    pub const fn latency_budget_us(mut self, budget: u32) -> Self {
        self.ctx.latency_budget_us = budget;
        self
    }

    /// Builds the routing context.
    #[must_use]
    pub const fn build(self) -> RoutingContext {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_type_min_tier() {
        assert_eq!(MutationType::VectorUpdate.min_tier(), ProofTier::Reflex);
        assert_eq!(MutationType::GraphEdge.min_tier(), ProofTier::Standard);
        assert_eq!(MutationType::StructuralChange.min_tier(), ProofTier::Deep);
    }

    #[test]
    fn test_routing_vector_update() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::VectorUpdate)
            .affected_nodes(1)
            .build();

        assert_eq!(route_proof_tier(&ctx), ProofTier::Reflex);
    }

    #[test]
    fn test_routing_graph_edge() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::GraphEdge)
            .affected_nodes(5)
            .build();

        assert_eq!(route_proof_tier(&ctx), ProofTier::Standard);
    }

    #[test]
    fn test_routing_structural_change() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::StructuralChange)
            .affected_nodes(50)
            .build();

        assert_eq!(route_proof_tier(&ctx), ProofTier::Deep);
    }

    #[test]
    fn test_routing_coherence_required() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::VectorUpdate)
            .requires_coherence(true)
            .build();

        // Coherence always requires Deep tier
        assert_eq!(route_proof_tier(&ctx), ProofTier::Deep);
    }

    #[test]
    fn test_routing_many_affected_nodes() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::VectorUpdate)
            .affected_nodes(150)
            .build();

        // Many nodes upgrades to Deep
        assert_eq!(route_proof_tier(&ctx), ProofTier::Deep);
    }

    #[test]
    fn test_tier_router_downgrade() {
        let router = TierRouter {
            allow_downgrade: true,
            ..Default::default()
        };

        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::GraphEdge)
            .affected_nodes(5)
            .latency_budget_us(1) // Very tight budget
            .build();

        // Should downgrade from Standard to Reflex
        assert_eq!(router.route(&ctx), ProofTier::Reflex);
    }

    #[test]
    fn test_tier_router_no_downgrade() {
        let router = TierRouter {
            allow_downgrade: false,
            ..Default::default()
        };

        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::GraphEdge)
            .affected_nodes(5)
            .latency_budget_us(1)
            .build();

        // Should not downgrade
        assert_eq!(router.route(&ctx), ProofTier::Standard);
    }

    #[test]
    fn test_routing_context_builder() {
        let ctx = RoutingContextBuilder::new()
            .mutation_type(MutationType::GraphRebalance)
            .requires_coherence(true)
            .high_frequency(false)
            .affected_nodes(200)
            .latency_budget_us(5000)
            .build();

        assert_eq!(ctx.mutation_type, MutationType::GraphRebalance);
        assert!(ctx.requires_coherence);
        assert!(!ctx.high_frequency);
        assert_eq!(ctx.affected_nodes, 200);
        assert_eq!(ctx.latency_budget_us, 5000);
    }
}
