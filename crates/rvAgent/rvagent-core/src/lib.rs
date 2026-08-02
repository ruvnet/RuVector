//! `rvagent-core` — Core types for the rvAgent framework.
//!
//! This crate provides the foundational types used across the rvAgent system:
//!
//! - [`config`] — Agent configuration (`RvAgentConfig`, `SecurityPolicy`, `ResourceBudget`)
//! - [`budget`] — Resource budget enforcement (ADR-103 B4)
//! - [`error`] — Error types (`RvAgentError`)
//! - [`graph`] — Agent execution graph / state machine (`AgentGraph`)
//! - [`messages`] — Message types (`Message`, `ToolCall`)
//! - [`models`] — Model resolution and `ChatModel` trait
//! - [`prompt`] — System prompt constants and builder
//! - [`state`] — Typed agent state with Arc-based O(1) cloning
//! - [`cow_state`] — Copy-on-write state backend for efficient forking (ADR-103 B2)
//! - [`arena`] — Bump arena allocator for hot-path scratch allocations (ADR-103 A8)
//! - [`metrics`] — Lock-free performance metrics collection (ADR-103 A9)
//! - [`parallel`] — Parallel async execution utilities (ADR-103 A2)
//! - [`string_pool`] — Thread-safe string interning for repeated strings
//! - [`agi_container`] — AGI Container building using RVF segments (ADR-103 B1)

pub mod agi_container;
pub mod arena;
pub mod budget;
pub mod bootstrap;
pub mod config;
pub mod cow_state;
pub mod error;
pub mod graph;
pub mod invariants;
pub mod masking;
pub mod messages;
pub mod metrics;
pub mod models;
pub mod parallel;
pub mod policy;
pub mod prompt;
pub mod rvf_bridge;
pub mod session_crypto;
pub mod state;
pub mod string_pool;

// Re-export key types at crate root for convenience.
pub use agi_container::{
    agi_tags, AgentNode as AgiAgentNode, AgentPrompt, AgiContainerBuilder, ContainerError,
    OrchestratorConfig, ParsedContainer, SegmentType, SkillDefinition, ToolDefinition,
};
pub use budget::{BudgetEnforcer, BudgetError, BudgetUtilization};
pub use config::{BackendConfig, ResourceBudget, RunnableConfig, RvAgentConfig, SecurityPolicy};
pub use cow_state::CowStateBackend;
pub use error::{Result, RvAgentError};
pub use graph::{AgentGraph, AgentNode, GraphConfig, ToolExecutor};
pub use messages::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};
pub use models::{ChatModel, ModelConfig, Provider, StreamChunk, StreamUsage, StreamingChatModel};
pub use prompt::{SystemPromptBuilder, BASE_AGENT_PROMPT};
pub use rvf_bridge::{
    GovernanceMode, MountTable, PolicyCheck, RvfBridgeConfig, RvfComponentId, RvfManifest,
    RvfManifestEntry, RvfManifestEntryType, RvfMountHandle, RvfToolCallEntry, RvfVerifyStatus,
    RvfWitnessHeader, TaskOutcome, WitTypeId,
};
pub use session_crypto::{
    derive_key, generate_key, generate_session_filename, CryptoError, EncryptionKey, SessionCrypto,
};
pub use state::{AgentState, FileData, SkillMetadata, TodoItem, TodoStatus};
