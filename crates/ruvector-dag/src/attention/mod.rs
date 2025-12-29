//! DAG Attention Mechanisms
//!
//! This module provides graph-topology-aware attention mechanisms for DAG-based
//! query optimization. Unlike traditional neural attention, these mechanisms
//! leverage the structural properties of the DAG (topology, paths, cuts) to
//! compute attention scores.

// Team 2 (Agent #2) - Base attention mechanisms
mod traits;
mod topological;
mod causal_cone;
mod critical_path;
mod mincut_gated;

// Team 2 (Agent #3) - Advanced attention mechanisms
mod trait_def;
mod hierarchical_lorentz;
mod parallel_branch;
mod temporal_btsp;
mod selector;
mod cache;

// Export base mechanisms
pub use traits::{DagAttention, AttentionScores, AttentionConfig, AttentionError};
pub use topological::{TopologicalAttention, TopologicalConfig};
pub use causal_cone::{CausalConeAttention, CausalConeConfig};
pub use critical_path::{CriticalPathAttention, CriticalPathConfig};
pub use mincut_gated::{MinCutGatedAttention, MinCutConfig, FlowCapacity};

// Export advanced mechanisms
pub use trait_def::{DagAttentionMechanism, AttentionScores as AttentionScoresV2, AttentionError as AttentionErrorV2};
pub use hierarchical_lorentz::{HierarchicalLorentzAttention, HierarchicalLorentzConfig};
pub use parallel_branch::{ParallelBranchAttention, ParallelBranchConfig};
pub use temporal_btsp::{TemporalBTSPAttention, TemporalBTSPConfig};
pub use selector::{AttentionSelector, SelectorConfig, MechanismStats};
pub use cache::{AttentionCache, CacheConfig, CacheStats};
