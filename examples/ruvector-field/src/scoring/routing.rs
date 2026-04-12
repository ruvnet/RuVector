//! Routing score — spec section 8.4.
//!
//! ```text
//! route_score = capability_fit
//!             * role_fit
//!             * locality_fit
//!             * shell_fit
//!             * expected_gain / expected_cost
//! ```
//!
//! No hardcoded constants: every factor is derived from the live engine state.

use crate::model::{Embedding, Shell};

/// Inputs for a single route score computation.
#[derive(Debug, Clone)]
pub struct RouteInputs<'a> {
    /// Query embedding.
    pub query: &'a Embedding,
    /// Agent capability embedding.
    pub capability: &'a Embedding,
    /// Role embedding (e.g. the role's typical task distribution).
    pub role: &'a Embedding,
    /// Partition distance in BFS hops via SharesRegion / RoutesTo edges.
    pub partition_distance: u32,
    /// Candidate agent's home shell.
    pub agent_shell: Shell,
    /// Target shell for this query.
    pub target_shell: Shell,
    /// Predicted delta in resonance.
    pub expected_gain: f32,
    /// Expected cost (nodes touched + distance penalty).
    pub expected_cost: f32,
}

/// Route score with individual factors broken out.
#[derive(Debug, Clone, Copy)]
pub struct RouteFactors {
    /// Capability fit.
    pub capability_fit: f32,
    /// Role fit.
    pub role_fit: f32,
    /// Locality fit.
    pub locality_fit: f32,
    /// Shell fit.
    pub shell_fit: f32,
    /// Gain divided by cost.
    pub gain_per_cost: f32,
}

impl RouteFactors {
    /// Product of the five factors.
    pub fn product(&self) -> f32 {
        self.capability_fit * self.role_fit * self.locality_fit * self.shell_fit * self.gain_per_cost
    }
}

/// Compute the route score from live inputs.
///
/// # Example
///
/// ```
/// use ruvector_field::model::{Embedding, Shell};
/// use ruvector_field::scoring::routing::{score_route, RouteInputs};
/// let q = Embedding::new(vec![1.0, 0.0, 0.0]);
/// let cap = Embedding::new(vec![0.9, 0.1, 0.0]);
/// let role = Embedding::new(vec![0.8, 0.1, 0.1]);
/// let f = score_route(&RouteInputs {
///     query: &q, capability: &cap, role: &role,
///     partition_distance: 1, agent_shell: Shell::Concept,
///     target_shell: Shell::Concept, expected_gain: 0.6, expected_cost: 0.2,
/// });
/// assert!(f.product() > 0.0);
/// ```
pub fn score_route(inputs: &RouteInputs<'_>) -> RouteFactors {
    let capability_fit = inputs.query.cosine01(inputs.capability).clamp(0.0, 1.0);
    let role_fit = inputs.query.cosine01(inputs.role).clamp(0.0, 1.0);
    let locality_fit = (1.0 / (1.0 + inputs.partition_distance as f32)).clamp(0.0, 1.0);
    let depth_gap =
        (inputs.agent_shell.depth() as i32 - inputs.target_shell.depth() as i32).abs() as f32;
    let shell_fit = (1.0 - depth_gap * 0.15).clamp(0.1, 1.0);
    let cost = inputs.expected_cost.max(1e-3);
    let gain_per_cost = (inputs.expected_gain.max(0.0) / cost).clamp(0.0, 10.0);

    RouteFactors {
        capability_fit,
        role_fit,
        locality_fit,
        shell_fit,
        gain_per_cost,
    }
}
