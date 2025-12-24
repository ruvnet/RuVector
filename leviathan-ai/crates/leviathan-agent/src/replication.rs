//! Agent Replication Module
//!
//! Provides systems for self-replication, mutation, and lineage tracking of agents.

use crate::spec::{AgentRole, AgentSpec, Capability, KnowledgeItem, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// System for spawning new agents from existing specifications
#[derive(Debug)]
pub struct AgentReplicator {
    /// History of all replications
    lineage: LineageTree,
}

impl AgentReplicator {
    /// Create a new replicator
    pub fn new() -> Self {
        Self {
            lineage: LineageTree::new(),
        }
    }

    /// Spawn an exact copy of an agent
    pub fn replicate(&mut self, parent: &AgentSpec) -> AgentSpec {
        let parent_hash = parent.hash();

        let mut child = parent.clone();
        child.id = Uuid::new_v4();
        child.parent_spec_hash = Some(parent_hash);

        self.lineage.add_child(parent_hash, child.hash());

        child
    }

    /// Spawn a mutated version of an agent
    pub fn replicate_with_mutation(
        &mut self,
        parent: &AgentSpec,
        mutations: Vec<MutationOperator>,
    ) -> anyhow::Result<AgentSpec> {
        let mut child = self.replicate(parent);

        for mutation in mutations {
            mutation.apply(&mut child)?;
        }

        // Re-validate after mutations
        child.validate()?;

        // Update lineage with new hash after mutations
        let parent_hash = parent.hash();
        self.lineage.add_child(parent_hash, child.hash());

        Ok(child)
    }

    /// Get the lineage tree
    pub fn lineage(&self) -> &LineageTree {
        &self.lineage
    }

    /// Spawn multiple agents from a template
    pub fn spawn_swarm(
        &mut self,
        template: &AgentSpec,
        count: usize,
        mutate: bool,
    ) -> Vec<AgentSpec> {
        let mut agents = Vec::with_capacity(count);

        for i in 0..count {
            let agent = if mutate {
                // Apply slight variations
                let mutations = vec![
                    MutationOperator::ModifyInstructions(format!(
                        "{}\n\nAgent #{} in swarm of {}",
                        template.instructions, i + 1, count
                    )),
                ];
                self.replicate_with_mutation(template, mutations)
                    .unwrap_or_else(|_| self.replicate(template))
            } else {
                self.replicate(template)
            };

            agents.push(agent);
        }

        agents
    }
}

impl Default for AgentReplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks the lineage of agent replications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageTree {
    /// Map of parent hash to children hashes
    children: HashMap<[u8; 32], Vec<[u8; 32]>>,

    /// Map of child hash to parent hash
    parents: HashMap<[u8; 32], [u8; 32]>,

    /// Root agents (no parent)
    roots: Vec<[u8; 32]>,
}

impl LineageTree {
    /// Create a new lineage tree
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            parents: HashMap::new(),
            roots: Vec::new(),
        }
    }

    /// Add a parent-child relationship
    pub fn add_child(&mut self, parent_hash: [u8; 32], child_hash: [u8; 32]) {
        self.children
            .entry(parent_hash)
            .or_insert_with(Vec::new)
            .push(child_hash);

        self.parents.insert(child_hash, parent_hash);
    }

    /// Register a root agent (no parent)
    pub fn add_root(&mut self, agent_hash: [u8; 32]) {
        if !self.roots.contains(&agent_hash) {
            self.roots.push(agent_hash);
        }
    }

    /// Get all children of an agent
    pub fn children(&self, parent_hash: &[u8; 32]) -> Option<&Vec<[u8; 32]>> {
        self.children.get(parent_hash)
    }

    /// Get the parent of an agent
    pub fn parent(&self, child_hash: &[u8; 32]) -> Option<&[u8; 32]> {
        self.parents.get(child_hash)
    }

    /// Get all root agents
    pub fn roots(&self) -> &Vec<[u8; 32]> {
        &self.roots
    }

    /// Get the full ancestry path from root to this agent
    pub fn ancestry(&self, agent_hash: &[u8; 32]) -> Vec<[u8; 32]> {
        let mut path = vec![*agent_hash];
        let mut current = *agent_hash;

        while let Some(parent) = self.parent(&current) {
            path.push(*parent);
            current = *parent;
        }

        path.reverse();
        path
    }

    /// Calculate generation depth (distance from root)
    pub fn generation(&self, agent_hash: &[u8; 32]) -> usize {
        self.ancestry(agent_hash).len() - 1
    }
}

impl Default for LineageTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Operations that can mutate an agent specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationOperator {
    /// Add a new capability
    AddCapability(Capability),

    /// Remove a capability by name
    RemoveCapability(String),

    /// Add a new tool
    AddTool(ToolSpec),

    /// Remove a tool by name
    RemoveTool(String),

    /// Modify the instructions
    ModifyInstructions(String),

    /// Append to the instructions
    AppendInstructions(String),

    /// Add a knowledge item
    AddKnowledge(KnowledgeItem),

    /// Change the role
    ChangeRole(AgentRole),

    /// Custom mutation with a description
    Custom { description: String, data: String },
}

impl MutationOperator {
    /// Apply this mutation to an agent spec
    pub fn apply(&self, spec: &mut AgentSpec) -> anyhow::Result<()> {
        match self {
            MutationOperator::AddCapability(cap) => {
                if spec.capabilities.iter().any(|c| c.name == cap.name) {
                    anyhow::bail!("Capability '{}' already exists", cap.name);
                }
                spec.capabilities.push(cap.clone());
            }

            MutationOperator::RemoveCapability(name) => {
                spec.capabilities.retain(|c| c.name != *name);
            }

            MutationOperator::AddTool(tool) => {
                if spec.tools.iter().any(|t| t.name == tool.name) {
                    anyhow::bail!("Tool '{}' already exists", tool.name);
                }
                spec.tools.push(tool.clone());
            }

            MutationOperator::RemoveTool(name) => {
                spec.tools.retain(|t| t.name != *name);
            }

            MutationOperator::ModifyInstructions(instructions) => {
                spec.instructions = instructions.clone();
            }

            MutationOperator::AppendInstructions(additional) => {
                spec.instructions.push('\n');
                spec.instructions.push_str(additional);
            }

            MutationOperator::AddKnowledge(item) => {
                if !spec.knowledge_base.contains(item) {
                    spec.knowledge_base.push(item.clone());
                }
            }

            MutationOperator::ChangeRole(role) => {
                spec.role = role.clone();
            }

            MutationOperator::Custom { .. } => {
                // Custom mutations would need to be handled by external logic
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::AgentBuilder;

    #[test]
    fn test_replication() {
        let mut replicator = AgentReplicator::new();

        let parent = AgentBuilder::new("Parent")
            .role(AgentRole::Researcher)
            .instruction("Original instructions")
            .build()
            .unwrap();

        let child = replicator.replicate(&parent);

        assert_ne!(parent.id, child.id);
        assert_eq!(parent.name, child.name);
        assert_eq!(child.parent_spec_hash, Some(parent.hash()));
    }

    #[test]
    fn test_mutation() {
        let mut replicator = AgentReplicator::new();

        let parent = AgentBuilder::new("Parent")
            .role(AgentRole::Researcher)
            .instruction("Original")
            .build()
            .unwrap();

        let mutations = vec![
            MutationOperator::AppendInstructions("Additional instructions".into()),
            MutationOperator::ChangeRole(AgentRole::MLEngineer),
        ];

        let child = replicator.replicate_with_mutation(&parent, mutations).unwrap();

        assert!(child.instructions.contains("Additional instructions"));
        assert!(matches!(child.role, AgentRole::MLEngineer));
    }

    #[test]
    fn test_lineage_tracking() {
        let tree = LineageTree::new();
        let root_hash = [0u8; 32];
        let child_hash = [1u8; 32];

        let mut tree = tree;
        tree.add_root(root_hash);
        tree.add_child(root_hash, child_hash);

        assert_eq!(tree.parent(&child_hash), Some(&root_hash));
        assert!(tree.children(&root_hash).unwrap().contains(&child_hash));
    }

    #[test]
    fn test_spawn_swarm() {
        let mut replicator = AgentReplicator::new();

        let template = AgentBuilder::new("Template")
            .role(AgentRole::Tester)
            .instruction("Test things")
            .build()
            .unwrap();

        let swarm = replicator.spawn_swarm(&template, 5, false);

        assert_eq!(swarm.len(), 5);
        assert!(swarm.iter().all(|a| a.name == "Template"));
    }
}
