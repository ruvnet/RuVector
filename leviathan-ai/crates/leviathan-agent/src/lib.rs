//! Leviathan Agent - Self-Replicating AI Agent System
//!
//! This crate provides a framework for creating, managing, and replicating AI agents
//! with specific roles, capabilities, and tools. Agents can spawn new agents,
//! evolve their specifications, and maintain lineage tracking.
//!
//! # Core Concepts
//!
//! - **AgentSpec**: The specification/blueprint of an agent
//! - **AgentBuilder**: Fluent API for constructing agents
//! - **AgentReplicator**: System for spawning new agents from specs
//! - **AgentExecutor**: Executes agent tasks with tool invocation
//!
//! # Example
//!
//! ```rust,no_run
//! use leviathan_agent::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a Junior AI Engineer agent
//!     let spec = junior_ai_engineer_spec();
//!
//!     // Build and execute
//!     let executor = AgentExecutor::new(spec);
//!     executor.execute_task("Implement a RAG system using LangChain").await?;
//!
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod executor;
pub mod replication;
pub mod spec;
pub mod templates;

// Re-exports for convenience
pub mod prelude {
    pub use crate::builder::AgentBuilder;
    pub use crate::executor::{AgentExecutor, ExecutionResult, TaskContext};
    pub use crate::replication::{AgentReplicator, LineageTree, MutationOperator};
    pub use crate::spec::{
        AgentRole, AgentSpec, Capability, KnowledgeItem, OutputParser, ToolSpec,
    };
    pub use crate::templates::junior_ai_engineer::junior_ai_engineer_spec;
    pub use anyhow::Result;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_create_agent_spec() {
        let spec = AgentSpec {
            id: uuid::Uuid::new_v4(),
            name: "Test Agent".into(),
            role: AgentRole::Tester,
            capabilities: vec![],
            tools: vec![],
            instructions: "Test instructions".into(),
            knowledge_base: vec![],
            parent_spec_hash: None,
        };

        assert_eq!(spec.name, "Test Agent");
        assert!(matches!(spec.role, AgentRole::Tester));
    }

    #[test]
    fn test_builder_pattern() {
        let spec = AgentBuilder::new("Test Builder Agent")
            .role(AgentRole::Researcher)
            .instruction("Research and analyze")
            .build()
            .unwrap();

        assert_eq!(spec.name, "Test Builder Agent");
    }
}
