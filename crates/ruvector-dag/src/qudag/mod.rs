//! QuDAG Integration - Quantum-Resistant Distributed Pattern Learning

pub mod crypto;
pub mod tokens;
mod client;
mod network;
mod proposal;
mod consensus;
mod sync;

pub use client::QuDagClient;
pub use network::{NetworkConfig, NetworkStatus};
pub use proposal::{PatternProposal, ProposalStatus};
pub use consensus::{ConsensusResult, Vote};
pub use sync::PatternSync;
pub use tokens::{StakingManager, RewardCalculator, GovernanceSystem};
pub use tokens::{StakeInfo, StakingError, RewardClaim, RewardSource};
pub use tokens::{Proposal as GovProposal, ProposalType, ProposalStatus as GovProposalStatus, VoteChoice, GovernanceError};
