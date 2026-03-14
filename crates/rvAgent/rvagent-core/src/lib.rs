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

pub mod config;
pub mod error;
pub mod graph;
pub mod messages;
pub mod models;
pub mod prompt;
pub mod state;

// Re-export key types at crate root for convenience.
pub use config::{BackendConfig, ResourceBudget, RvAgentConfig, SecurityPolicy};
pub use error::{Result, RvAgentError};
pub use graph::{AgentGraph, AgentNode, GraphConfig, ToolExecutor};
pub use messages::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};
pub use models::{ChatModel, ModelConfig, Provider};
pub use prompt::{SystemPromptBuilder, BASE_AGENT_PROMPT};
pub use state::{AgentState, FileData, SkillMetadata, TodoItem, TodoStatus};
