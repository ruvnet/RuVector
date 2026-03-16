//! RVF Container support for rvagent-wasm.
//!
//! Provides WASM bindings for building and parsing RVF (RuVector Format)
//! cognitive containers that package tools, prompts, skills, and orchestrator
//! configurations for complex multi-agent systems.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use wasm_bindgen::prelude::*;

use crate::bridge::to_js_value;

// ---------------------------------------------------------------------------
// RVF Segment Types
// ---------------------------------------------------------------------------

/// RVF segment types (from spec)
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
    fn from_u8(value: u8) -> Result<Self, RvfError> {
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
            _ => Err(RvfError::InvalidSegment(format!(
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
    pub const MCP_TOOLS: u16 = 0x010B;
    pub const CAPABILITY_SET: u16 = 0x010C;
}

// ---------------------------------------------------------------------------
// RVF Data Types
// ---------------------------------------------------------------------------

/// Tool definition for registry
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
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

/// MCP tool definition for container
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct McpToolEntry {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub group: Option<String>,
}

/// Capability definition (Ruvix-compatible)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CapabilityDef {
    pub name: String,
    pub rights: Vec<String>,
    pub scope: String,
    pub delegation_depth: u8,
}

// ---------------------------------------------------------------------------
// Internal Segment
// ---------------------------------------------------------------------------

struct Segment {
    segment_type: SegmentType,
    tag: u16,
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Security Constants
// ---------------------------------------------------------------------------

/// Maximum container size (10 MB)
pub const MAX_CONTAINER_SIZE: usize = 10 * 1024 * 1024;

/// Maximum segment size (5 MB)
pub const MAX_SEGMENT_SIZE: usize = 5 * 1024 * 1024;

/// Maximum number of segments
pub const MAX_SEGMENTS: usize = 1000;

/// Maximum delegation depth (prevents deep delegation chains)
pub const MAX_DELEGATION_DEPTH: u8 = 10;

// ---------------------------------------------------------------------------
// RVF Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RvfError {
    InvalidMagic,
    ChecksumMismatch,
    InvalidSegment(String),
    InvalidFormat(String),
    ParseError(String),
    SizeExceeded(String),
    SecurityViolation(String),
}

impl std::fmt::Display for RvfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RvfError::InvalidMagic => write!(f, "Invalid magic bytes (expected 'RVF\\x01')"),
            RvfError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            RvfError::InvalidSegment(msg) => write!(f, "Invalid segment: {}", msg),
            RvfError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            RvfError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            RvfError::SizeExceeded(msg) => write!(f, "Size exceeded: {}", msg),
            RvfError::SecurityViolation(msg) => write!(f, "Security violation: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed Container
// ---------------------------------------------------------------------------

/// Parsed RVF container
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParsedContainer {
    pub tools: Vec<ToolDefinition>,
    pub prompts: Vec<AgentPrompt>,
    pub skills: Vec<SkillDefinition>,
    pub orchestrator: Option<OrchestratorConfig>,
    pub mcp_tools: Vec<McpToolEntry>,
    pub capabilities: Vec<CapabilityDef>,
}

// ---------------------------------------------------------------------------
// WasmRvfBuilder — WASM-exported RVF container builder
// ---------------------------------------------------------------------------

/// RVF Container Builder for WASM.
///
/// Build RVF cognitive containers that package tools, prompts, skills,
/// orchestrator configs, MCP tools, and Ruvix capabilities.
///
/// # Example (JavaScript)
/// ```js
/// const builder = new WasmRvfBuilder();
/// builder.addTool({ name: "search", description: "Web search", parameters: {} });
/// builder.addPrompt({ name: "coder", system_prompt: "You are a coder", version: "1.0" });
/// const container = builder.build();
/// // container is Uint8Array with RVF magic bytes
/// ```
#[wasm_bindgen]
pub struct WasmRvfBuilder {
    segments: Vec<Segment>,
}

#[wasm_bindgen]
impl WasmRvfBuilder {
    /// Create a new RVF container builder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add a tool definition.
    #[wasm_bindgen(js_name = addTool)]
    pub fn add_tool(&mut self, tool_json: &str) -> Result<(), JsValue> {
        let tool: ToolDefinition = serde_json::from_str(tool_json)
            .map_err(|e| JsValue::from_str(&format!("invalid tool JSON: {}", e)))?;
        self.add_tool_internal(tool);
        Ok(())
    }

    /// Add multiple tools from JSON array.
    #[wasm_bindgen(js_name = addTools)]
    pub fn add_tools(&mut self, tools_json: &str) -> Result<(), JsValue> {
        let tools: Vec<ToolDefinition> = serde_json::from_str(tools_json)
            .map_err(|e| JsValue::from_str(&format!("invalid tools JSON: {}", e)))?;
        let data = serde_json::to_vec(&tools).expect("serialize tools");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::TOOL_REGISTRY,
            data,
        });
        Ok(())
    }

    /// Add an agent prompt.
    #[wasm_bindgen(js_name = addPrompt)]
    pub fn add_prompt(&mut self, prompt_json: &str) -> Result<(), JsValue> {
        let prompt: AgentPrompt = serde_json::from_str(prompt_json)
            .map_err(|e| JsValue::from_str(&format!("invalid prompt JSON: {}", e)))?;
        self.add_prompt_internal(prompt);
        Ok(())
    }

    /// Add multiple prompts from JSON array.
    #[wasm_bindgen(js_name = addPrompts)]
    pub fn add_prompts(&mut self, prompts_json: &str) -> Result<(), JsValue> {
        let prompts: Vec<AgentPrompt> = serde_json::from_str(prompts_json)
            .map_err(|e| JsValue::from_str(&format!("invalid prompts JSON: {}", e)))?;
        let data = serde_json::to_vec(&prompts).expect("serialize prompts");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::AGENT_PROMPTS,
            data,
        });
        Ok(())
    }

    /// Add a skill definition.
    #[wasm_bindgen(js_name = addSkill)]
    pub fn add_skill(&mut self, skill_json: &str) -> Result<(), JsValue> {
        let skill: SkillDefinition = serde_json::from_str(skill_json)
            .map_err(|e| JsValue::from_str(&format!("invalid skill JSON: {}", e)))?;
        self.add_skill_internal(skill);
        Ok(())
    }

    /// Add multiple skills from JSON array.
    #[wasm_bindgen(js_name = addSkills)]
    pub fn add_skills(&mut self, skills_json: &str) -> Result<(), JsValue> {
        let skills: Vec<SkillDefinition> = serde_json::from_str(skills_json)
            .map_err(|e| JsValue::from_str(&format!("invalid skills JSON: {}", e)))?;
        let data = serde_json::to_vec(&skills).expect("serialize skills");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::SKILL_LIBRARY,
            data,
        });
        Ok(())
    }

    /// Set orchestrator configuration.
    #[wasm_bindgen(js_name = setOrchestrator)]
    pub fn set_orchestrator(&mut self, config_json: &str) -> Result<(), JsValue> {
        let config: OrchestratorConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid orchestrator JSON: {}", e)))?;
        let data = serde_json::to_vec(&config).expect("serialize orchestrator");
        self.segments.push(Segment {
            segment_type: SegmentType::Profile,
            tag: agi_tags::ORCHESTRATOR,
            data,
        });
        Ok(())
    }

    /// Add MCP tool entries.
    #[wasm_bindgen(js_name = addMcpTools)]
    pub fn add_mcp_tools(&mut self, tools_json: &str) -> Result<(), JsValue> {
        let tools: Vec<McpToolEntry> = serde_json::from_str(tools_json)
            .map_err(|e| JsValue::from_str(&format!("invalid MCP tools JSON: {}", e)))?;
        let data = serde_json::to_vec(&tools).expect("serialize MCP tools");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::MCP_TOOLS,
            data,
        });
        Ok(())
    }

    /// Add Ruvix capability definitions.
    #[wasm_bindgen(js_name = addCapabilities)]
    pub fn add_capabilities(&mut self, caps_json: &str) -> Result<(), JsValue> {
        let caps: Vec<CapabilityDef> = serde_json::from_str(caps_json)
            .map_err(|e| JsValue::from_str(&format!("invalid capabilities JSON: {}", e)))?;
        let data = serde_json::to_vec(&caps).expect("serialize capabilities");
        self.segments.push(Segment {
            segment_type: SegmentType::Config,
            tag: agi_tags::CAPABILITY_SET,
            data,
        });
        Ok(())
    }

    /// Build the RVF container as bytes.
    ///
    /// Returns a Uint8Array containing the RVF binary:
    /// - Magic bytes: "RVF\x01" (4 bytes)
    /// - Segment count: u32 LE (4 bytes)
    /// - Segments: type(1) + tag(2) + len(4) + data
    /// - Checksum: SHA3-256 (32 bytes)
    pub fn build(&self) -> Result<js_sys::Uint8Array, JsValue> {
        let bytes = self.build_internal();
        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// Parse an RVF container from bytes.
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(data: &[u8]) -> Result<JsValue, JsValue> {
        let parsed = Self::parse_internal(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_js_value(&parsed)
    }

    /// Validate an RVF container (check magic and checksum).
    pub fn validate(data: &[u8]) -> Result<bool, JsValue> {
        match Self::validate_internal(data) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get the RVF magic bytes for detection.
    #[wasm_bindgen(js_name = getMagic)]
    pub fn get_magic() -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&b"RVF\x01"[..])
    }
}

impl Default for WasmRvfBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmRvfBuilder {
    fn add_tool_internal(&mut self, tool: ToolDefinition) {
        // For single tool, we wrap in array for consistency
        let data = serde_json::to_vec(&vec![tool]).expect("serialize tool");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::TOOL_REGISTRY,
            data,
        });
    }

    fn add_prompt_internal(&mut self, prompt: AgentPrompt) {
        let data = serde_json::to_vec(&vec![prompt]).expect("serialize prompt");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::AGENT_PROMPTS,
            data,
        });
    }

    fn add_skill_internal(&mut self, skill: SkillDefinition) {
        let data = serde_json::to_vec(&vec![skill]).expect("serialize skill");
        self.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::SKILL_LIBRARY,
            data,
        });
    }

    /// Build the RVF container as raw bytes (Rust-only).
    pub(crate) fn build_internal(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Magic bytes "RVF\x01"
        output.extend_from_slice(b"RVF\x01");

        // Number of segments
        output.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());

        // Each segment: type(1) + tag(2) + len(4) + data
        for seg in &self.segments {
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

    fn validate_internal(data: &[u8]) -> Result<(), RvfError> {
        // Security: Check container size limit
        if data.len() > MAX_CONTAINER_SIZE {
            return Err(RvfError::SizeExceeded(format!(
                "Container size {} exceeds maximum {}",
                data.len(),
                MAX_CONTAINER_SIZE
            )));
        }

        if data.len() < 40 {
            return Err(RvfError::InvalidFormat("Container too small".to_string()));
        }

        // Verify magic bytes
        if &data[0..4] != b"RVF\x01" {
            return Err(RvfError::InvalidMagic);
        }

        // Security: Check segment count
        let segment_count =
            u32::from_le_bytes(data[4..8].try_into().expect("4 bytes for u32")) as usize;
        if segment_count > MAX_SEGMENTS {
            return Err(RvfError::SizeExceeded(format!(
                "Segment count {} exceeds maximum {}",
                segment_count,
                MAX_SEGMENTS
            )));
        }

        // Verify checksum
        let checksum_start = data.len() - 32;
        let stored_checksum = &data[checksum_start..];
        let mut hasher = Sha3_256::new();
        hasher.update(&data[..checksum_start]);
        let computed_checksum = hasher.finalize();

        if stored_checksum != computed_checksum.as_slice() {
            return Err(RvfError::ChecksumMismatch);
        }

        Ok(())
    }

    fn parse_internal(data: &[u8]) -> Result<ParsedContainer, RvfError> {
        Self::validate_internal(data)?;

        // Parse segment count
        let segment_count =
            u32::from_le_bytes(data[4..8].try_into().expect("4 bytes for u32")) as usize;

        let mut tools = Vec::new();
        let mut prompts = Vec::new();
        let mut skills = Vec::new();
        let mut orchestrator = None;
        let mut mcp_tools = Vec::new();
        let mut capabilities = Vec::new();

        let checksum_start = data.len() - 32;
        let mut offset = 8;

        for _ in 0..segment_count {
            if offset + 7 > checksum_start {
                return Err(RvfError::InvalidFormat("Truncated segment header".to_string()));
            }

            let _segment_type = SegmentType::from_u8(data[offset])?;
            offset += 1;

            let tag = u16::from_le_bytes(data[offset..offset + 2].try_into().expect("2 bytes"));
            offset += 2;

            let len = u32::from_le_bytes(data[offset..offset + 4].try_into().expect("4 bytes")) as usize;
            offset += 4;

            // Security: Check individual segment size
            if len > MAX_SEGMENT_SIZE {
                return Err(RvfError::SizeExceeded(format!(
                    "Segment size {} exceeds maximum {}",
                    len,
                    MAX_SEGMENT_SIZE
                )));
            }

            if offset + len > checksum_start {
                return Err(RvfError::InvalidFormat("Truncated segment data".to_string()));
            }

            let segment_data = &data[offset..offset + len];
            offset += len;

            // Parse based on tag
            match tag {
                agi_tags::TOOL_REGISTRY => {
                    let parsed: Vec<ToolDefinition> = serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse tools: {}", e)))?;
                    tools.extend(parsed);
                }
                agi_tags::AGENT_PROMPTS => {
                    let parsed: Vec<AgentPrompt> = serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse prompts: {}", e)))?;
                    prompts.extend(parsed);
                }
                agi_tags::SKILL_LIBRARY => {
                    let parsed: Vec<SkillDefinition> = serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse skills: {}", e)))?;
                    skills.extend(parsed);
                }
                agi_tags::ORCHESTRATOR => {
                    orchestrator = Some(serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse orchestrator: {}", e)))?);
                }
                agi_tags::MCP_TOOLS => {
                    let parsed: Vec<McpToolEntry> = serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse MCP tools: {}", e)))?;
                    mcp_tools.extend(parsed);
                }
                agi_tags::CAPABILITY_SET => {
                    let parsed: Vec<CapabilityDef> = serde_json::from_slice(segment_data)
                        .map_err(|e| RvfError::ParseError(format!("Failed to parse capabilities: {}", e)))?;
                    // Security: Validate delegation depth
                    for cap in &parsed {
                        if cap.delegation_depth > MAX_DELEGATION_DEPTH {
                            return Err(RvfError::SecurityViolation(format!(
                                "Capability '{}' has delegation_depth {} exceeding maximum {}",
                                cap.name, cap.delegation_depth, MAX_DELEGATION_DEPTH
                            )));
                        }
                    }
                    capabilities.extend(parsed);
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
            mcp_tools,
            capabilities,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty_container() {
        let builder = WasmRvfBuilder::new();
        let container = builder.build_internal();

        assert_eq!(&container[0..4], b"RVF\x01");
        assert_eq!(container.len(), 4 + 4 + 32);

        let count = u32::from_le_bytes(container[4..8].try_into().unwrap());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_build_with_tool() {
        let mut builder = WasmRvfBuilder::new();
        builder.add_tool_internal(ToolDefinition {
            name: "test".to_string(),
            description: "Test tool".to_string(),
            parameters: serde_json::json!({}),
            returns: None,
        });

        let container = builder.build_internal();
        assert_eq!(&container[0..4], b"RVF\x01");

        let parsed = WasmRvfBuilder::parse_internal(&container).unwrap();
        assert_eq!(parsed.tools.len(), 1);
        assert_eq!(parsed.tools[0].name, "test");
    }

    #[test]
    fn test_build_with_mcp_tools() {
        let mut builder = WasmRvfBuilder::new();
        let tools = vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read file".to_string(),
                input_schema: serde_json::json!({"path": "string"}),
                group: Some("file".to_string()),
            },
        ];
        let data = serde_json::to_vec(&tools).unwrap();
        builder.segments.push(Segment {
            segment_type: SegmentType::Data,
            tag: agi_tags::MCP_TOOLS,
            data,
        });

        let container = builder.build_internal();
        let parsed = WasmRvfBuilder::parse_internal(&container).unwrap();
        assert_eq!(parsed.mcp_tools.len(), 1);
        assert_eq!(parsed.mcp_tools[0].name, "read_file");
    }

    #[test]
    fn test_build_with_capabilities() {
        let mut builder = WasmRvfBuilder::new();
        let caps = vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 2,
            },
        ];
        let data = serde_json::to_vec(&caps).unwrap();
        builder.segments.push(Segment {
            segment_type: SegmentType::Config,
            tag: agi_tags::CAPABILITY_SET,
            data,
        });

        let container = builder.build_internal();
        let parsed = WasmRvfBuilder::parse_internal(&container).unwrap();
        assert_eq!(parsed.capabilities.len(), 1);
        assert_eq!(parsed.capabilities[0].name, "file_read");
    }

    #[test]
    fn test_validate_invalid_magic() {
        let data = vec![0u8; 40];
        let result = WasmRvfBuilder::validate_internal(&data);
        assert!(matches!(result, Err(RvfError::InvalidMagic)));
    }

    #[test]
    fn test_validate_invalid_checksum() {
        let mut container = WasmRvfBuilder::new().build_internal();
        let len = container.len();
        container[len - 1] ^= 0xFF;

        let result = WasmRvfBuilder::validate_internal(&container);
        assert!(matches!(result, Err(RvfError::ChecksumMismatch)));
    }

    #[test]
    fn test_size_limits() {
        // Test that container size is validated
        let huge_data = vec![0u8; MAX_CONTAINER_SIZE + 1];
        let result = WasmRvfBuilder::validate_internal(&huge_data);
        assert!(matches!(result, Err(RvfError::SizeExceeded(_))));
    }

    #[test]
    fn test_delegation_depth_limit() {
        let mut builder = WasmRvfBuilder::new();

        // Create capability with excessive delegation depth
        let caps = vec![CapabilityDef {
            name: "excessive_cap".to_string(),
            rights: vec!["read".to_string()],
            scope: "sandbox".to_string(),
            delegation_depth: MAX_DELEGATION_DEPTH + 1,
        }];
        let data = serde_json::to_vec(&caps).unwrap();
        builder.segments.push(Segment {
            segment_type: SegmentType::Config,
            tag: agi_tags::CAPABILITY_SET,
            data,
        });

        let container = builder.build_internal();
        let result = WasmRvfBuilder::parse_internal(&container);
        assert!(matches!(result, Err(RvfError::SecurityViolation(_))));
    }

    #[test]
    fn test_full_roundtrip() {
        let mut builder = WasmRvfBuilder::new();

        builder.add_tool_internal(ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({"query": "string"}),
            returns: Some("results".to_string()),
        });

        builder.add_prompt_internal(AgentPrompt {
            name: "researcher".to_string(),
            system_prompt: "You are a researcher".to_string(),
            version: "1.0.0".to_string(),
        });

        builder.add_skill_internal(SkillDefinition {
            name: "summarize".to_string(),
            description: "Summarize content".to_string(),
            trigger: "/summarize".to_string(),
            content: "Provide a concise summary".to_string(),
        });

        let container = builder.build_internal();
        let parsed = WasmRvfBuilder::parse_internal(&container).unwrap();

        assert_eq!(parsed.tools.len(), 1);
        assert_eq!(parsed.prompts.len(), 1);
        assert_eq!(parsed.skills.len(), 1);
        assert!(parsed.orchestrator.is_none());
    }
}
