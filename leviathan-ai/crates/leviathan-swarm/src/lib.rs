//! # Leviathan Swarm
//!
//! Pure Rust swarm orchestrator with parallel execution, topology support, and performance metrics.
//!
//! ## Features
//!
//! - Multiple topology patterns (mesh, hierarchical, star, ring)
//! - DAG-based task orchestration with dependency resolution
//! - Parallel and sequential execution strategies
//! - Comprehensive metrics and audit trails
//! - No external orchestration dependencies

pub mod agent;
pub mod metrics;
pub mod orchestrator;
pub mod task;
pub mod topology;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

pub use agent::{AgentResult, AgentSpec, AgentState};
pub use metrics::Metrics;
pub use orchestrator::Orchestrator;
pub use task::{Task, TaskQueue, TaskResult};
pub use topology::{HierarchicalTopology, MeshTopology, RingTopology, StarTopology, Topology};

/// Swarm orchestration errors
#[derive(Debug, Error)]
pub enum SwarmError {
    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Task error: {0}")]
    Task(String),

    #[error("Topology error: {0}")]
    Topology(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Dependency cycle detected")]
    DependencyCycle,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, SwarmError>;

/// Execution strategy for the swarm
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Execute tasks in parallel where possible
    Parallel,
    /// Execute tasks sequentially in order
    Sequential,
    /// Execute tasks based on dependency graph (DAG)
    DAG,
}

/// Topology type for agent communication
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TopologyType {
    /// All-to-all communication
    Mesh,
    /// Coordinator with workers
    Hierarchical,
    /// Central hub pattern
    Star,
    /// Pipeline processing
    Ring,
}

/// Swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Unique swarm identifier
    pub id: Uuid,
    /// Swarm name
    pub name: String,
    /// Communication topology
    pub topology: TopologyType,
    /// Maximum number of concurrent agents
    pub max_agents: usize,
    /// Execution strategy
    pub strategy: ExecutionStrategy,
    /// Enable detailed audit logging
    pub audit_enabled: bool,
    /// Enable performance metrics
    pub metrics_enabled: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "default-swarm".to_string(),
            topology: TopologyType::Mesh,
            max_agents: num_cpus::get(),
            strategy: ExecutionStrategy::DAG,
            audit_enabled: true,
            metrics_enabled: true,
        }
    }
}

/// Builder for swarm configuration
pub struct SwarmBuilder {
    config: SwarmConfig,
}

impl SwarmBuilder {
    /// Create a new swarm builder
    pub fn new() -> Self {
        Self {
            config: SwarmConfig::default(),
        }
    }

    /// Set swarm name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set topology type
    pub fn topology(mut self, topology: TopologyType) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set maximum concurrent agents
    pub fn max_agents(mut self, max_agents: usize) -> Self {
        self.config.max_agents = max_agents;
        self
    }

    /// Set execution strategy
    pub fn strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable audit logging
    pub fn audit_enabled(mut self, enabled: bool) -> Self {
        self.config.audit_enabled = enabled;
        self
    }

    /// Enable metrics collection
    pub fn metrics_enabled(mut self, enabled: bool) -> Self {
        self.config.metrics_enabled = enabled;
        self
    }

    /// Build the swarm
    pub fn build(self) -> Swarm {
        Swarm::new(self.config)
    }
}

impl Default for SwarmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main swarm orchestrator
pub struct Swarm {
    /// Swarm configuration
    config: SwarmConfig,
    /// Registered agents
    agents: Arc<dashmap::DashMap<Uuid, AgentSpec>>,
    /// Task orchestrator
    orchestrator: Orchestrator,
    /// Communication topology
    topology: Arc<dyn Topology>,
    /// Performance metrics
    metrics: Arc<Metrics>,
}

impl Swarm {
    /// Create a new swarm with the given configuration
    pub fn new(config: SwarmConfig) -> Self {
        let topology: Arc<dyn Topology> = match config.topology {
            TopologyType::Mesh => Arc::new(MeshTopology::new()),
            TopologyType::Hierarchical => Arc::new(HierarchicalTopology::new()),
            TopologyType::Star => Arc::new(StarTopology::new()),
            TopologyType::Ring => Arc::new(RingTopology::new()),
        };

        let metrics = Arc::new(Metrics::new(config.metrics_enabled));

        Self {
            orchestrator: Orchestrator::new(config.clone(), Arc::clone(&metrics)),
            config,
            agents: Arc::new(dashmap::DashMap::new()),
            topology,
            metrics,
        }
    }

    /// Register an agent with the swarm
    pub fn register_agent(&self, agent: AgentSpec) -> Result<Uuid> {
        if self.agents.len() >= self.config.max_agents {
            return Err(SwarmError::Agent(format!(
                "Maximum agent limit ({}) reached",
                self.config.max_agents
            )));
        }

        let agent_id = agent.id;
        self.agents.insert(agent_id, agent);
        self.topology.add_agent(agent_id)?;

        tracing::info!("Registered agent: {}", agent_id);
        Ok(agent_id)
    }

    /// Remove an agent from the swarm
    pub fn unregister_agent(&self, agent_id: Uuid) -> Result<()> {
        self.agents.remove(&agent_id);
        self.topology.remove_agent(agent_id)?;

        tracing::info!("Unregistered agent: {}", agent_id);
        Ok(())
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: Uuid) -> Option<AgentSpec> {
        self.agents.get(&agent_id).map(|entry| entry.clone())
    }

    /// List all registered agents
    pub fn list_agents(&self) -> Vec<AgentSpec> {
        self.agents.iter().map(|entry| entry.clone()).collect()
    }

    /// Execute tasks using the configured strategy
    pub async fn execute(&self, tasks: Vec<Task>) -> Result<Vec<TaskResult>> {
        match self.config.strategy {
            ExecutionStrategy::Parallel => self.orchestrator.execute_parallel(tasks).await,
            ExecutionStrategy::Sequential => self.orchestrator.execute_sequential(tasks).await,
            ExecutionStrategy::DAG => self.orchestrator.execute_dag(tasks).await,
        }
    }

    /// Get swarm metrics
    pub fn get_metrics(&self) -> serde_json::Value {
        self.metrics.export()
    }

    /// Get swarm configuration
    pub fn config(&self) -> &SwarmConfig {
        &self.config
    }

    /// Get swarm ID
    pub fn id(&self) -> Uuid {
        self.config.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_builder() {
        let swarm = SwarmBuilder::new()
            .name("test-swarm")
            .topology(TopologyType::Mesh)
            .max_agents(10)
            .strategy(ExecutionStrategy::DAG)
            .build();

        assert_eq!(swarm.config().name, "test-swarm");
        assert_eq!(swarm.config().topology, TopologyType::Mesh);
        assert_eq!(swarm.config().max_agents, 10);
        assert_eq!(swarm.config().strategy, ExecutionStrategy::DAG);
    }

    #[test]
    fn test_agent_registration() {
        let swarm = SwarmBuilder::new().max_agents(2).build();

        let agent1 = AgentSpec::new("agent1", vec!["capability1".to_string()]);
        let agent2 = AgentSpec::new("agent2", vec!["capability2".to_string()]);

        assert!(swarm.register_agent(agent1.clone()).is_ok());
        assert!(swarm.register_agent(agent2.clone()).is_ok());

        // Should fail - max agents reached
        let agent3 = AgentSpec::new("agent3", vec!["capability3".to_string()]);
        assert!(swarm.register_agent(agent3).is_err());
    }
}

// Re-export num_cpus for internal use
use num_cpus;
