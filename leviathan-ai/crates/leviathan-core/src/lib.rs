//! # Leviathan AI Core
//!
//! Enterprise AI orchestration with full DAG auditability.
//! Built for bank-grade compliance (FFIEC, BCBS 239, SR 11-7, GDPR).
//!
//! ## Features
//!
//! - **φ-Lattice Processor**: Pure integer Zeckendorf arithmetic with perfect perplexity
//! - **DAG Audit Trail**: Full data lineage tracking with cryptographic verification
//! - **Swarm Orchestrator**: Pure Rust parallel execution with no MCP dependencies
//! - **Self-Replicating Agents**: Agents that can spawn modified copies of themselves
//! - **Regulatory Compliance**: Production-ready FFIEC, BCBS 239, SR 11-7, GDPR

#![cfg_attr(not(feature = "std"), no_std)]

use serde::{Deserialize, Serialize};

// Re-export all sub-crates
#[cfg(feature = "lattice")]
pub use leviathan_lattice as lattice;

#[cfg(feature = "dag")]
pub use leviathan_dag as dag;

#[cfg(feature = "swarm")]
pub use leviathan_swarm as swarm;

#[cfg(feature = "agent")]
pub use leviathan_agent as agent;

#[cfg(feature = "compliance")]
pub use leviathan_compliance as compliance;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::*;

    #[cfg(feature = "lattice")]
    pub use crate::lattice::*;

    #[cfg(feature = "dag")]
    pub use crate::dag::*;

    #[cfg(feature = "swarm")]
    pub use crate::swarm::*;

    #[cfg(feature = "agent")]
    pub use crate::agent::prelude::*;

    #[cfg(feature = "compliance")]
    pub use crate::compliance::*;
}

/// System-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub organization: String,
    pub audit_enabled: bool,
    pub max_agents: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            name: "Leviathan AI".into(),
            organization: "Leviathan AI Inc.".into(),
            audit_enabled: true,
            max_agents: 100,
        }
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Leviathan AI";
pub const COMPANY: &str = "Leviathan AI Inc.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SystemConfig::default();
        assert!(config.audit_enabled);
        assert_eq!(config.max_agents, 100);
    }
}
