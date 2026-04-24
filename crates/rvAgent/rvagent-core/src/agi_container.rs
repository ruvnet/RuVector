//! AGI Container building using RVF segments and tags.
//!
//! This module implements the B1 (Concrete AGI Container) specification from ADR-103.
//! It provides a builder for constructing RVF-compliant AGI containers that package:
//! - Tool registries
//! - Agent prompts
//! - Skill libraries
//! - Orchestrator configurations
//!
//! The format follows the RVF specification with magic bytes, segment headers, and SHA3-256 checksums.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::fmt;

/// AGI Container segment types (from RVF spec)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentType {
    Header = 0x01,
    Metadata = 0x02,
    Code = 0x03,
    Data = 0x04,
    Weights = 0x05,
    Config = 0x06,
    Manifest = 0x07,
    Signature = 0x08,
    Checkpoint = 0x09,
    Witness = 0x0A,
    Profile = 0x0B,
}

impl SegmentType {
    /// Parse segment type from byte
    pub fn from_u8(value: u8) -> Result<Self, ContainerError> {
        match value {
            0x01 => Ok(SegmentType::Header),
            0x02 => Ok(SegmentType::Metadata),
            0x03 => Ok(SegmentType::Code),
            0x04 => Ok(SegmentType::Data),
            0x05 => Ok(SegmentType::Weights),
            0x06 => Ok(SegmentType::Config),
            0x07 => Ok(SegmentType::Manifest),
            0x08 => Ok(SegmentType::Signature),
            0x09 => Ok(SegmentType::Checkpoint),
            0x0A => Ok(SegmentType::Witness),
            0x0B => Ok(SegmentType::Profile),
            _ => Err(ContainerError::InvalidSegment(format!(
                "Unknown segment type: 0x{:02x}",
                value
            ))),
        }
    }
}

/// AGI-specific tags
pub mod agi_tags {
    pub const TOOL_REGISTRY: u16 = 0x0105;
    pub const AGENT_PROMPTS: u16 = 0x0106;
    pub const SKILL_LIBRARY: u16 = 0x0109;
    pub const ORCHESTRATOR: u16 = 0x0108;
    pub const MIDDLEWARE_CONFIG: u16 = 0x010A;
}

/// Tool definition for registry
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub returns: Option<String>,
}

/// Agent prompt definition
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentPrompt {
    pub name: String,
    pub system_prompt: String,
    pub version: String,
}

/// Skill definition for library
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub trigger: String,
    pub content: String,
}

/// Orchestrator topology
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrchestratorConfig {
    pub topology: String,
    pub agents: Vec<AgentNode>,
    pub connections: Vec<(String, String)>,
}

/// Agent node in orchestrator topology
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentNode {
    pub id: String,
    pub agent_type: String,
    pub prompt_ref: String,
}

/// Internal segment representation
struct Segment {
    segment_type: SegmentType,
    tag: u16,
    data: Vec<u8>,
}

/// AGI Container builder
///
/// # Example
///
/// ```
/// use rvagent_core::agi_container::{AgiContainerBuilder, ToolDefinition};
/// use serde_json::json;
///
/// let tool = ToolDefinition {
///     name: "web_search".to_string(),
///     description: "Search the web".to_string(),
///     parameters: json!({"query": "string"}),
///     returns: Some("SearchResults".to_string()),
/// };
///
/// let container = AgiContainerBuilder::new()
///     .with_tools(&[tool])
///     .build();
///
/// assert_eq!(&container[0..4], b"RVF\x01");
/// ```
pub struct AgiContainerBuilder {
    segments: Vec<Segment>,
}

impl AgiContainerBuilder {
    /// Create a new AGI container builder
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add tool registry
    pub fn with_tools(mut self, tools: &[ToolDefinition]) -> Self {
        let data = serde_json::to_vec(tools).expect("serialize tools");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::TOOL_REGISTRY,
            data,
        });
        self
    }

    /// Add agent prompts
    pub fn with_prompts(mut self, prompts: &[AgentPrompt]) -> Self {
        let data = serde_json::to_vec(prompts).expect("serialize prompts");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::AGENT_PROMPTS,
            data,
        });
        self
    }

    /// Add skill library
    pub fn with_skills(mut self, skills: &[SkillDefinition]) -> Self {
        let data = serde_json::to_vec(skills).expect("serialize skills");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::SKILL_LIBRARY,
            data,
        });
        self
    }

    /// Add orchestrator config
    pub fn with_orchestrator(mut self, config: &OrchestratorConfig) -> Self {
        let data = serde_json::to_vec(config).expect("serialize orchestrator");
        self.segments.push(Segment {
            segment_type: SegmentType::Profile,
            tag: agi_tags::ORCHESTRATOR,
            data,
        });
        self
    }

    /// Build the container as bytes
    ///
    /// Format:
    /// - Magic bytes: "RVF\x01" (4 bytes)
    /// - Segment count: u32 LE (4 bytes)
    /// - For each segment:
    ///   - Type: u8 (1 byte)
    ///   - Tag: u16 LE (2 bytes)
    ///   - Length: u32 LE (4 bytes)
    ///   - Data: [u8; length]
    /// - Checksum: SHA3-256 hash (32 bytes)
    pub fn build(self) -> Vec<u8> {
        let mut output = Vec::new();

        // Magic bytes "RVF\x01"
        output.extend_from_slice(b"RVF\x01");

        // Number of segments
        output.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());

        // Each segment: type(1) + tag(2) + len(4) + data
        for seg in self.segments {
            output.push(seg.segment_type as u8);
            output.extend_from_slice(&seg.tag.to_le_bytes());
            output.extend_from_slice(&(seg.data.len() as u32).to_le_bytes());
            output.extend_from_slice(&seg.data);
        }

        // Append SHA3-256 checksum
        let mut hasher = Sha3_256::new();
        hasher.update(&output);
        output.extend_from_slice(&hasher.finalize());

        output
    }

    /// Parse container from bytes
    pub fn parse(data: &[u8]) -> Result<ParsedContainer, ContainerError> {
        if data.len() < 40 {
            // min: 4 (magic) + 4 (count) + 32 (checksum)
            return Err(ContainerError::InvalidFormat(
                "Container too small".to_string(),
            ));
        }

        // Verify magic bytes
        if &data[0..4] != b"RVF\x01" {
            return Err(ContainerError::InvalidMagic);
        }

        // Verify checksum
        let checksum_start = data.len() - 32;
        let stored_checksum = &data[checksum_start..];
        let mut hasher = Sha3_256::new();
        hasher.update(&data[..checksum_start]);
        let computed_checksum = hasher.finalize();

        if stored_checksum != computed_checksum.as_slice() {
            return Err(ContainerError::ChecksumMismatch);
        }

        // Parse segment count
        let segment_count =
            u32::from_le_bytes(data[4..8].try_into().expect("4 bytes for u32")) as usize;

        let mut tools = Vec::new();
        let mut prompts = Vec::new();
        let mut skills = Vec::new();
        let mut orchestrator = None;

        // Parse segments
        let mut offset = 8;
        for _ in 0..segment_count {
            if offset + 7 > checksum_start {
                return Err(ContainerError::InvalidFormat(
                    "Truncated segment header".to_string(),
                ));
            }

            let _segment_type = SegmentType::from_u8(data[offset])?;
            offset += 1;

            let tag = u16::from_le_bytes(data[offset..offset + 2].try_into().expect("2 bytes"));
            offset += 2;

            let len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().expect("4 bytes")) as usize;
            offset += 4;

            if offset + len > checksum_start {
                return Err(ContainerError::InvalidFormat(
                    "Truncated segment data".to_string(),
                ));
            }

            let segment_data = &data[offset..offset + len];
            offset += len;

            // Parse based on tag
            match tag {
                agi_tags::TOOL_REGISTRY => {
                    tools = serde_json::from_slice(segment_data).map_err(|e| {
                        ContainerError::ParseError(format!("Failed to parse tools: {}", e))
                    })?;
                }
                agi_tags::AGENT_PROMPTS => {
                    prompts = serde_json::from_slice(segment_data).map_err(|e| {
                        ContainerError::ParseError(format!("Failed to parse prompts: {}", e))
                    })?;
                }
                agi_tags::SKILL_LIBRARY => {
                    skills = serde_json::from_slice(segment_data).map_err(|e| {
                        ContainerError::ParseError(format!("Failed to parse skills: {}", e))
                    })?;
                }
                agi_tags::ORCHESTRATOR => {
                    orchestrator = Some(serde_json::from_slice(segment_data).map_err(|e| {
                        ContainerError::ParseError(format!("Failed to parse orchestrator: {}", e))
                    })?);
                }
                _ => {
                    // Unknown tag - skip
                }
            }
        }

        Ok(ParsedContainer {
            tools,
            prompts,
            skills,
            orchestrator,
        })
    }
}

impl Default for AgiContainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed AGI container
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedContainer {
    pub tools: Vec<ToolDefinition>,
    pub prompts: Vec<AgentPrompt>,
    pub skills: Vec<SkillDefinition>,
    pub orchestrator: Option<OrchestratorConfig>,
}

/// Container parsing and building errors
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerError {
    InvalidMagic,
    ChecksumMismatch,
    InvalidSegment(String),
    InvalidFormat(String),
    ParseError(String),
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerError::InvalidMagic => write!(f, "Invalid magic bytes (expected 'RVF\\x01')"),
            ContainerError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            ContainerError::InvalidSegment(msg) => write!(f, "Invalid segment: {}", msg),
            ContainerError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ContainerError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ContainerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_empty_container() {
        let container = AgiContainerBuilder::new().build();

        // Should have magic + count + checksum
        assert_eq!(&container[0..4], b"RVF\x01");
        assert_eq!(container.len(), 4 + 4 + 32); // magic + count + checksum

        let count = u32::from_le_bytes(container[4..8].try_into().unwrap());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_build_with_tools() {
        let tool = ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({"query": "string"}),
            returns: Some("SearchResults".to_string()),
        };

        let container = AgiContainerBuilder::new()
            .with_tools(&[tool.clone()])
            .build();

        // Verify magic
        assert_eq!(&container[0..4], b"RVF\x01");

        // Verify segment count
        let count = u32::from_le_bytes(container[4..8].try_into().unwrap());
        assert_eq!(count, 1);

        // Parse and verify
        let parsed = AgiContainerBuilder::parse(&container).unwrap();
        assert_eq!(parsed.tools.len(), 1);
        assert_eq!(parsed.tools[0], tool);
    }

    #[test]
    fn test_build_with_prompts() {
        let prompt = AgentPrompt {
            name: "researcher".to_string(),
            system_prompt: "You are a research assistant".to_string(),
            version: "1.0.0".to_string(),
        };

        let container = AgiContainerBuilder::new()
            .with_prompts(&[prompt.clone()])
            .build();

        let parsed = AgiContainerBuilder::parse(&container).unwrap();
        assert_eq!(parsed.prompts.len(), 1);
        assert_eq!(parsed.prompts[0], prompt);
    }

    #[test]
    fn test_build_with_skills() {
        let skill = SkillDefinition {
            name: "code-review".to_string(),
            description: "Review code for quality".to_string(),
            trigger: "/review".to_string(),
            content: "Check for best practices".to_string(),
        };

        let container = AgiContainerBuilder::new()
            .with_skills(&[skill.clone()])
            .build();

        let parsed = AgiContainerBuilder::parse(&container).unwrap();
        assert_eq!(parsed.skills.len(), 1);
        assert_eq!(parsed.skills[0], skill);
    }

    #[test]
    fn test_build_with_orchestrator() {
        let config = OrchestratorConfig {
            topology: "hierarchical".to_string(),
            agents: vec![
                AgentNode {
                    id: "agent-1".to_string(),
                    agent_type: "researcher".to_string(),
                    prompt_ref: "researcher-v1".to_string(),
                },
                AgentNode {
                    id: "agent-2".to_string(),
                    agent_type: "coder".to_string(),
                    prompt_ref: "coder-v1".to_string(),
                },
            ],
            connections: vec![("agent-1".to_string(), "agent-2".to_string())],
        };

        let container = AgiContainerBuilder::new()
            .with_orchestrator(&config)
            .build();

        let parsed = AgiContainerBuilder::parse(&container).unwrap();
        assert!(parsed.orchestrator.is_some());
        assert_eq!(parsed.orchestrator.unwrap(), config);
    }

    #[test]
    fn test_build_complete_container() {
        let tool = ToolDefinition {
            name: "execute_code".to_string(),
            description: "Execute code snippet".to_string(),
            parameters: json!({"code": "string", "language": "string"}),
            returns: Some("ExecutionResult".to_string()),
        };

        let prompt = AgentPrompt {
            name: "coder".to_string(),
            system_prompt: "You are an expert programmer".to_string(),
            version: "2.0.0".to_string(),
        };

        let skill = SkillDefinition {
            name: "refactor".to_string(),
            description: "Refactor code".to_string(),
            trigger: "/refactor".to_string(),
            content: "Improve code structure".to_string(),
        };

        let orchestrator = OrchestratorConfig {
            topology: "mesh".to_string(),
            agents: vec![AgentNode {
                id: "coder-1".to_string(),
                agent_type: "coder".to_string(),
                prompt_ref: "coder".to_string(),
            }],
            connections: vec![],
        };

        let container = AgiContainerBuilder::new()
            .with_tools(&[tool.clone()])
            .with_prompts(&[prompt.clone()])
            .with_skills(&[skill.clone()])
            .with_orchestrator(&orchestrator)
            .build();

        let parsed = AgiContainerBuilder::parse(&container).unwrap();

        assert_eq!(parsed.tools.len(), 1);
        assert_eq!(parsed.tools[0], tool);

        assert_eq!(parsed.prompts.len(), 1);
        assert_eq!(parsed.prompts[0], prompt);

        assert_eq!(parsed.skills.len(), 1);
        assert_eq!(parsed.skills[0], skill);

        assert!(parsed.orchestrator.is_some());
        assert_eq!(parsed.orchestrator.unwrap(), orchestrator);
    }

    #[test]
    fn test_parse_invalid_magic() {
        // Need at least 40 bytes to pass size check before magic check
        let mut data = vec![0u8; 40];
        data[0..4].copy_from_slice(b"XXXX"); // Invalid magic
        let result = AgiContainerBuilder::parse(&data);
        assert!(matches!(result, Err(ContainerError::InvalidMagic)));
    }

    #[test]
    fn test_parse_invalid_checksum() {
        let mut container = AgiContainerBuilder::new().build();
        // Corrupt the checksum
        let len = container.len();
        container[len - 1] ^= 0xFF;

        let result = AgiContainerBuilder::parse(&container);
        assert!(matches!(result, Err(ContainerError::ChecksumMismatch)));
    }

    #[test]
    fn test_parse_truncated_container() {
        let data = b"RVF\x01\x01".to_vec();
        let result = AgiContainerBuilder::parse(&data);
        assert!(matches!(result, Err(ContainerError::InvalidFormat(_))));
    }

    #[test]
    fn test_segment_type_parsing() {
        assert_eq!(SegmentType::from_u8(0x01).unwrap(), SegmentType::Header);
        assert_eq!(SegmentType::from_u8(0x04).unwrap(), SegmentType::Data);
        assert_eq!(SegmentType::from_u8(0x0B).unwrap(), SegmentType::Profile);
        assert!(SegmentType::from_u8(0xFF).is_err());
    }

    #[test]
    fn test_checksum_integrity() {
        let tool = ToolDefinition {
            name: "test".to_string(),
            description: "Test tool".to_string(),
            parameters: json!({}),
            returns: None,
        };

        let container = AgiContainerBuilder::new().with_tools(&[tool]).build();

        // Parse should succeed with correct checksum
        assert!(AgiContainerBuilder::parse(&container).is_ok());

        // Modify data before checksum
        let mut corrupted = container.clone();
        corrupted[10] ^= 0x01;

        // Parse should fail with incorrect checksum
        assert!(matches!(
            AgiContainerBuilder::parse(&corrupted),
            Err(ContainerError::ChecksumMismatch)
        ));
    }

    #[test]
    fn test_multiple_tools_roundtrip() {
        let tools = vec![
            ToolDefinition {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                parameters: json!({"param1": "type1"}),
                returns: Some("Result1".to_string()),
            },
            ToolDefinition {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                parameters: json!({"param2": "type2"}),
                returns: Some("Result2".to_string()),
            },
        ];

        let container = AgiContainerBuilder::new().with_tools(&tools).build();
        let parsed = AgiContainerBuilder::parse(&container).unwrap();

        assert_eq!(parsed.tools.len(), 2);
        assert_eq!(parsed.tools, tools);
    }

    #[test]
    fn test_empty_orchestrator_connections() {
        let config = OrchestratorConfig {
            topology: "star".to_string(),
            agents: vec![],
            connections: vec![],
        };

        let container = AgiContainerBuilder::new()
            .with_orchestrator(&config)
            .build();

        let parsed = AgiContainerBuilder::parse(&container).unwrap();
        let parsed_config = parsed.orchestrator.unwrap();
        assert_eq!(parsed_config.topology, "star");
        assert!(parsed_config.agents.is_empty());
        assert!(parsed_config.connections.is_empty());
    }
}
