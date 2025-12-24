//! Agent Builder Module
//!
//! Provides a fluent API for constructing agent specifications.

use crate::spec::{AgentRole, AgentSpec, Capability, KnowledgeItem, ToolSpec};
use std::collections::HashMap;
use uuid::Uuid;

/// Fluent builder for creating agent specifications
#[derive(Debug, Default)]
pub struct AgentBuilder {
    name: String,
    role: Option<AgentRole>,
    capabilities: Vec<Capability>,
    tools: Vec<ToolSpec>,
    instructions: String,
    knowledge_base: Vec<KnowledgeItem>,
    parent_spec_hash: Option<[u8; 32]>,
}

impl AgentBuilder {
    /// Create a new agent builder with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the agent's role
    pub fn role(mut self, role: AgentRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Add a capability to the agent
    pub fn capability(mut self, capability: Capability) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Add multiple capabilities at once
    pub fn capabilities(mut self, capabilities: Vec<Capability>) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    /// Add a tool to the agent
    pub fn tool(mut self, tool: ToolSpec) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools at once
    pub fn tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the agent's instructions
    pub fn instruction(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Add a knowledge item
    pub fn knowledge(mut self, item: KnowledgeItem) -> Self {
        self.knowledge_base.push(item);
        self
    }

    /// Add multiple knowledge items
    pub fn knowledge_items(mut self, items: Vec<KnowledgeItem>) -> Self {
        self.knowledge_base.extend(items);
        self
    }

    /// Set the parent spec hash for lineage tracking
    pub fn parent(mut self, parent_hash: [u8; 32]) -> Self {
        self.parent_spec_hash = Some(parent_hash);
        self
    }

    /// Build the agent specification
    pub fn build(self) -> anyhow::Result<AgentSpec> {
        if self.name.is_empty() {
            anyhow::bail!("Agent name cannot be empty");
        }

        let role = self.role.ok_or_else(|| anyhow::anyhow!("Agent role must be specified"))?;

        let spec = AgentSpec {
            id: Uuid::new_v4(),
            name: self.name,
            role,
            capabilities: self.capabilities,
            tools: self.tools,
            instructions: self.instructions,
            knowledge_base: self.knowledge_base,
            parent_spec_hash: self.parent_spec_hash,
        };

        // Validate the spec
        spec.validate()?;

        Ok(spec)
    }
}

/// Helper function to create a simple capability
pub fn capability(
    name: impl Into<String>,
    description: impl Into<String>,
    required_tools: Vec<String>,
) -> Capability {
    Capability {
        name: name.into(),
        description: description.into(),
        required_tools,
    }
}

/// Helper function to create a simple tool
pub fn tool(
    name: impl Into<String>,
    command: impl Into<String>,
    args_template: impl Into<String>,
) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        command: command.into(),
        args_template: args_template.into(),
        output_parser: crate::spec::OutputParser::Raw,
        working_dir: None,
        env_vars: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::OutputParser;

    #[test]
    fn test_builder_basic() {
        let spec = AgentBuilder::new("Test Agent")
            .role(AgentRole::Researcher)
            .instruction("Do research")
            .build()
            .unwrap();

        assert_eq!(spec.name, "Test Agent");
        assert!(matches!(spec.role, AgentRole::Researcher));
    }

    #[test]
    fn test_builder_with_capabilities() {
        let spec = AgentBuilder::new("Data Engineer")
            .role(AgentRole::DataEngineer)
            .capability(capability(
                "python_dev",
                "Python development",
                vec!["python".into()],
            ))
            .tool(tool("python", "python3", "{{script}}"))
            .instruction("Build data pipelines")
            .build()
            .unwrap();

        assert_eq!(spec.capabilities.len(), 1);
        assert_eq!(spec.tools.len(), 1);
    }

    #[test]
    fn test_builder_validation_fails() {
        let result = AgentBuilder::new("Invalid Agent")
            .role(AgentRole::MLEngineer)
            .capability(capability(
                "missing_tool",
                "Needs a tool",
                vec!["nonexistent".into()],
            ))
            .instruction("Will fail")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_role() {
        let result = AgentBuilder::new("No Role Agent")
            .instruction("Missing role")
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("role must be specified"));
    }
}
