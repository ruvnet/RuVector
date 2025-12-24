//! Communication topology implementations

use crate::{Result, SwarmError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use uuid::Uuid;

/// Trait for swarm communication topology
pub trait Topology: Send + Sync {
    /// Add an agent to the topology
    fn add_agent(&self, agent_id: Uuid) -> Result<()>;

    /// Remove an agent from the topology
    fn remove_agent(&self, agent_id: Uuid) -> Result<()>;

    /// Get neighbors for an agent (agents it can communicate with)
    fn get_neighbors(&self, agent_id: Uuid) -> Result<Vec<Uuid>>;

    /// Get all agents in the topology
    fn get_all_agents(&self) -> Vec<Uuid>;

    /// Get topology metadata
    fn metadata(&self) -> TopologyMetadata;
}

/// Topology metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetadata {
    pub topology_type: String,
    pub agent_count: usize,
    pub edge_count: usize,
    pub is_connected: bool,
}

/// Mesh topology - all-to-all communication
pub struct MeshTopology {
    agents: RwLock<HashSet<Uuid>>,
}

impl MeshTopology {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(HashSet::new()),
        }
    }
}

impl Default for MeshTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl Topology for MeshTopology {
    fn add_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        agents.insert(agent_id);
        Ok(())
    }

    fn remove_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        agents.remove(&agent_id);
        Ok(())
    }

    fn get_neighbors(&self, agent_id: Uuid) -> Result<Vec<Uuid>> {
        let agents = self.agents.read().unwrap();
        if !agents.contains(&agent_id) {
            return Err(SwarmError::Topology(format!(
                "Agent {} not found in topology",
                agent_id
            )));
        }

        // In mesh, every agent can communicate with every other agent
        Ok(agents
            .iter()
            .filter(|&&id| id != agent_id)
            .copied()
            .collect())
    }

    fn get_all_agents(&self) -> Vec<Uuid> {
        self.agents.read().unwrap().iter().copied().collect()
    }

    fn metadata(&self) -> TopologyMetadata {
        let agents = self.agents.read().unwrap();
        let agent_count = agents.len();
        let edge_count = if agent_count > 1 {
            agent_count * (agent_count - 1) / 2
        } else {
            0
        };

        TopologyMetadata {
            topology_type: "mesh".to_string(),
            agent_count,
            edge_count,
            is_connected: agent_count > 0,
        }
    }
}

/// Hierarchical topology - coordinator with workers
pub struct HierarchicalTopology {
    coordinator: RwLock<Option<Uuid>>,
    workers: RwLock<HashSet<Uuid>>,
}

impl HierarchicalTopology {
    pub fn new() -> Self {
        Self {
            coordinator: RwLock::new(None),
            workers: RwLock::new(HashSet::new()),
        }
    }

    pub fn set_coordinator(&self, agent_id: Uuid) -> Result<()> {
        let mut coordinator = self.coordinator.write().unwrap();
        *coordinator = Some(agent_id);
        Ok(())
    }
}

impl Default for HierarchicalTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl Topology for HierarchicalTopology {
    fn add_agent(&self, agent_id: Uuid) -> Result<()> {
        let coordinator = self.coordinator.read().unwrap();
        if coordinator.is_none() {
            // First agent becomes coordinator
            drop(coordinator);
            self.set_coordinator(agent_id)?;
        } else {
            // Subsequent agents become workers
            let mut workers = self.workers.write().unwrap();
            workers.insert(agent_id);
        }
        Ok(())
    }

    fn remove_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut coordinator = self.coordinator.write().unwrap();
        if coordinator.as_ref() == Some(&agent_id) {
            *coordinator = None;
        } else {
            let mut workers = self.workers.write().unwrap();
            workers.remove(&agent_id);
        }
        Ok(())
    }

    fn get_neighbors(&self, agent_id: Uuid) -> Result<Vec<Uuid>> {
        let coordinator = self.coordinator.read().unwrap();
        let workers = self.workers.read().unwrap();

        if coordinator.as_ref() == Some(&agent_id) {
            // Coordinator can communicate with all workers
            Ok(workers.iter().copied().collect())
        } else if workers.contains(&agent_id) {
            // Workers can only communicate with coordinator
            Ok(coordinator.iter().copied().collect())
        } else {
            Err(SwarmError::Topology(format!(
                "Agent {} not found in topology",
                agent_id
            )))
        }
    }

    fn get_all_agents(&self) -> Vec<Uuid> {
        let coordinator = self.coordinator.read().unwrap();
        let workers = self.workers.read().unwrap();

        let mut agents = Vec::new();
        if let Some(coord) = *coordinator {
            agents.push(coord);
        }
        agents.extend(workers.iter().copied());
        agents
    }

    fn metadata(&self) -> TopologyMetadata {
        let coordinator = self.coordinator.read().unwrap();
        let workers = self.workers.read().unwrap();
        let agent_count = coordinator.iter().count() + workers.len();
        let edge_count = workers.len(); // Each worker connects to coordinator

        TopologyMetadata {
            topology_type: "hierarchical".to_string(),
            agent_count,
            edge_count,
            is_connected: coordinator.is_some() && !workers.is_empty(),
        }
    }
}

/// Star topology - central hub
pub struct StarTopology {
    hub: RwLock<Option<Uuid>>,
    nodes: RwLock<HashSet<Uuid>>,
}

impl StarTopology {
    pub fn new() -> Self {
        Self {
            hub: RwLock::new(None),
            nodes: RwLock::new(HashSet::new()),
        }
    }

    pub fn set_hub(&self, agent_id: Uuid) -> Result<()> {
        let mut hub = self.hub.write().unwrap();
        *hub = Some(agent_id);
        Ok(())
    }
}

impl Default for StarTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl Topology for StarTopology {
    fn add_agent(&self, agent_id: Uuid) -> Result<()> {
        let hub = self.hub.read().unwrap();
        if hub.is_none() {
            // First agent becomes hub
            drop(hub);
            self.set_hub(agent_id)?;
        } else {
            // Subsequent agents become nodes
            let mut nodes = self.nodes.write().unwrap();
            nodes.insert(agent_id);
        }
        Ok(())
    }

    fn remove_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut hub = self.hub.write().unwrap();
        if hub.as_ref() == Some(&agent_id) {
            *hub = None;
        } else {
            let mut nodes = self.nodes.write().unwrap();
            nodes.remove(&agent_id);
        }
        Ok(())
    }

    fn get_neighbors(&self, agent_id: Uuid) -> Result<Vec<Uuid>> {
        let hub = self.hub.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        if hub.as_ref() == Some(&agent_id) {
            // Hub can communicate with all nodes
            Ok(nodes.iter().copied().collect())
        } else if nodes.contains(&agent_id) {
            // Nodes can only communicate with hub
            Ok(hub.iter().copied().collect())
        } else {
            Err(SwarmError::Topology(format!(
                "Agent {} not found in topology",
                agent_id
            )))
        }
    }

    fn get_all_agents(&self) -> Vec<Uuid> {
        let hub = self.hub.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        let mut agents = Vec::new();
        if let Some(h) = *hub {
            agents.push(h);
        }
        agents.extend(nodes.iter().copied());
        agents
    }

    fn metadata(&self) -> TopologyMetadata {
        let hub = self.hub.read().unwrap();
        let nodes = self.nodes.read().unwrap();
        let agent_count = hub.iter().count() + nodes.len();
        let edge_count = nodes.len();

        TopologyMetadata {
            topology_type: "star".to_string(),
            agent_count,
            edge_count,
            is_connected: hub.is_some() && !nodes.is_empty(),
        }
    }
}

/// Ring topology - pipeline processing
pub struct RingTopology {
    agents: RwLock<Vec<Uuid>>,
}

impl RingTopology {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(Vec::new()),
        }
    }
}

impl Default for RingTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl Topology for RingTopology {
    fn add_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        agents.push(agent_id);
        Ok(())
    }

    fn remove_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        agents.retain(|&id| id != agent_id);
        Ok(())
    }

    fn get_neighbors(&self, agent_id: Uuid) -> Result<Vec<Uuid>> {
        let agents = self.agents.read().unwrap();
        let pos = agents
            .iter()
            .position(|&id| id == agent_id)
            .ok_or_else(|| {
                SwarmError::Topology(format!("Agent {} not found in topology", agent_id))
            })?;

        let len = agents.len();
        if len < 2 {
            return Ok(Vec::new());
        }

        let prev = agents[(pos + len - 1) % len];
        let next = agents[(pos + 1) % len];

        Ok(vec![prev, next])
    }

    fn get_all_agents(&self) -> Vec<Uuid> {
        self.agents.read().unwrap().clone()
    }

    fn metadata(&self) -> TopologyMetadata {
        let agents = self.agents.read().unwrap();
        let agent_count = agents.len();
        let edge_count = if agent_count > 1 { agent_count } else { 0 };

        TopologyMetadata {
            topology_type: "ring".to_string(),
            agent_count,
            edge_count,
            is_connected: agent_count > 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_topology() {
        let topology = MeshTopology::new();

        let agent1 = Uuid::new_v4();
        let agent2 = Uuid::new_v4();
        let agent3 = Uuid::new_v4();

        topology.add_agent(agent1).unwrap();
        topology.add_agent(agent2).unwrap();
        topology.add_agent(agent3).unwrap();

        // In mesh, each agent should see all others
        let neighbors = topology.get_neighbors(agent1).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&agent2));
        assert!(neighbors.contains(&agent3));

        let metadata = topology.metadata();
        assert_eq!(metadata.agent_count, 3);
        assert_eq!(metadata.edge_count, 3); // 3 * 2 / 2 = 3 edges
    }

    #[test]
    fn test_hierarchical_topology() {
        let topology = HierarchicalTopology::new();

        let coordinator = Uuid::new_v4();
        let worker1 = Uuid::new_v4();
        let worker2 = Uuid::new_v4();

        topology.add_agent(coordinator).unwrap();
        topology.add_agent(worker1).unwrap();
        topology.add_agent(worker2).unwrap();

        // Coordinator should see all workers
        let coord_neighbors = topology.get_neighbors(coordinator).unwrap();
        assert_eq!(coord_neighbors.len(), 2);

        // Workers should only see coordinator
        let worker_neighbors = topology.get_neighbors(worker1).unwrap();
        assert_eq!(worker_neighbors.len(), 1);
        assert_eq!(worker_neighbors[0], coordinator);
    }

    #[test]
    fn test_ring_topology() {
        let topology = RingTopology::new();

        let agent1 = Uuid::new_v4();
        let agent2 = Uuid::new_v4();
        let agent3 = Uuid::new_v4();

        topology.add_agent(agent1).unwrap();
        topology.add_agent(agent2).unwrap();
        topology.add_agent(agent3).unwrap();

        // Each agent should see exactly 2 neighbors
        let neighbors = topology.get_neighbors(agent2).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&agent1));
        assert!(neighbors.contains(&agent3));
    }
}
