//! `rvagent-core` — Core types for the rvAgent framework.
//!
//! This crate provides the foundational types used across the rvAgent system:
//!
//! - [`config`] — Agent configuration (`RvAgentConfig`, `SecurityPolicy`, `ResourceBudget`)
//! - [`error`] — Error types (`RvAgentError`)
//! - [`graph`] — Agent execution graph / state machine (`AgentGraph`)
//! - [`messages`] — Message types (`Message`, `ToolCall`)
//! - [`models`] — Model resolution and `ChatModel` trait
//! - [`prompt`] — System prompt constants and builder
//! - [`state`] — Typed agent state with Arc-based O(1) cloning
//! - [`arena`] — Bump arena allocator for hot-path scratch allocations (ADR-103 A8)
//! - [`metrics`] — Lock-free performance metrics collection (ADR-103 A9)
//! - [`parallel`] — Parallel async execution utilities (ADR-103 A2)
//! - [`string_pool`] — Thread-safe string interning for repeated strings

pub mod arena;
pub mod config;
pub mod error;
pub mod graph;
pub mod messages;
pub mod metrics;
pub mod models;
pub mod parallel;
pub mod prompt;
pub mod rvf_bridge;
pub mod state;
pub mod string_pool;

// Re-export key types at crate root for convenience.
pub use config::{BackendConfig, ResourceBudget, RvAgentConfig, SecurityPolicy};
pub use error::{Result, RvAgentError};
pub use graph::{AgentGraph, AgentNode, GraphConfig, ToolExecutor};
pub use messages::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};
pub use models::{ChatModel, ModelConfig, Provider};
pub use prompt::{SystemPromptBuilder, BASE_AGENT_PROMPT};
pub use rvf_bridge::{
    GovernanceMode, MountTable, PolicyCheck, RvfBridgeConfig, RvfComponentId, RvfManifest,
    RvfManifestEntry, RvfManifestEntryType, RvfMountHandle, RvfToolCallEntry, RvfVerifyStatus,
    RvfWitnessHeader, TaskOutcome, WitTypeId,
};
pub use state::{AgentState, FileData, SkillMetadata, TodoItem, TodoStatus};
