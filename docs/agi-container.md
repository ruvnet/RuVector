# AGI Container (B1 Implementation)

## Overview

The AGI Container is a concrete implementation of the B1 specification from ADR-103. It provides a standardized format for packaging agent components using the RVF (RuVector Format) specification.

## Format Specification

### Container Structure

```
┌─────────────────────────────────────┐
│ Magic Bytes: "RVF\x01" (4 bytes)   │
├─────────────────────────────────────┤
│ Segment Count: u32 LE (4 bytes)    │
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ Segment 1:                      │ │
│ │   Type: u8 (1 byte)             │ │
│ │   Tag: u16 LE (2 bytes)         │ │
│ │   Length: u32 LE (4 bytes)      │ │
│ │   Data: [u8; length]            │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Segment 2:                      │ │
│ │   ...                           │ │
│ └─────────────────────────────────┘ │
│ ...                                 │
├─────────────────────────────────────┤
│ Checksum: SHA3-256 (32 bytes)      │
└─────────────────────────────────────┘
```

### Segment Types

| Type | Value | Description |
|------|-------|-------------|
| Header | 0x01 | Container header metadata |
| Metadata | 0x02 | General metadata |
| Code | 0x03 | Executable code |
| Data | 0x04 | Data segments |
| Weights | 0x05 | Model weights |
| Config | 0x06 | Configuration |
| Manifest | 0x07 | Manifest entries |
| Signature | 0x08 | Cryptographic signatures |
| Checkpoint | 0x09 | State checkpoints |
| Witness | 0x0A | Witness data |
| Profile | 0x0B | Profile data |

### AGI Tags

| Tag | Value | Description |
|-----|-------|-------------|
| TOOL_REGISTRY | 0x0105 | Tool definitions |
| AGENT_PROMPTS | 0x0106 | Agent system prompts |
| ORCHESTRATOR | 0x0108 | Orchestrator configuration |
| SKILL_LIBRARY | 0x0109 | Skill definitions |
| MIDDLEWARE_CONFIG | 0x010A | Middleware configuration |

## Usage

### Building a Container

```rust
use rvagent_core::agi_container::{
    AgiContainerBuilder, ToolDefinition, AgentPrompt,
    SkillDefinition, OrchestratorConfig, AgentNode
};
use serde_json::json;

// Define tools
let tools = vec![
    ToolDefinition {
        name: "web_search".to_string(),
        description: "Search the web".to_string(),
        parameters: json!({"query": "string"}),
        returns: Some("SearchResults".to_string()),
    }
];

// Define prompts
let prompts = vec![
    AgentPrompt {
        name: "researcher".to_string(),
        system_prompt: "You are a research assistant.".to_string(),
        version: "1.0.0".to_string(),
    }
];

// Define skills
let skills = vec![
    SkillDefinition {
        name: "code-review".to_string(),
        description: "Review code quality".to_string(),
        trigger: "/review".to_string(),
        content: "Check for best practices".to_string(),
    }
];

// Define orchestrator
let orchestrator = OrchestratorConfig {
    topology: "hierarchical".to_string(),
    agents: vec![
        AgentNode {
            id: "researcher-1".to_string(),
            agent_type: "researcher".to_string(),
            prompt_ref: "researcher".to_string(),
        }
    ],
    connections: vec![],
};

// Build container
let container = AgiContainerBuilder::new()
    .with_tools(&tools)
    .with_prompts(&prompts)
    .with_skills(&skills)
    .with_orchestrator(&orchestrator)
    .build();

// Container is now a Vec<u8> ready for storage or transmission
```

### Parsing a Container

```rust
use rvagent_core::agi_container::AgiContainerBuilder;

let container_bytes = /* ... */;

// Parse the container
let parsed = AgiContainerBuilder::parse(&container_bytes)?;

// Access components
println!("Tools: {}", parsed.tools.len());
println!("Prompts: {}", parsed.prompts.len());
println!("Skills: {}", parsed.skills.len());

if let Some(orch) = parsed.orchestrator {
    println!("Orchestrator topology: {}", orch.topology);
}
```

## Data Structures

### ToolDefinition

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub returns: Option<String>,
}
```

### AgentPrompt

```rust
pub struct AgentPrompt {
    pub name: String,
    pub system_prompt: String,
    pub version: String,
}
```

### SkillDefinition

```rust
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub trigger: String,
    pub content: String,
}
```

### OrchestratorConfig

```rust
pub struct OrchestratorConfig {
    pub topology: String,
    pub agents: Vec<AgentNode>,
    pub connections: Vec<(String, String)>,
}

pub struct AgentNode {
    pub id: String,
    pub agent_type: String,
    pub prompt_ref: String,
}
```

## Security

### Checksum Verification

All containers include a SHA3-256 checksum of the container data (excluding the checksum itself). This ensures:

- Data integrity during storage and transmission
- Detection of corruption or tampering
- Cryptographic verification of container authenticity

The parser automatically verifies the checksum and returns `ContainerError::ChecksumMismatch` if verification fails.

### Error Handling

```rust
pub enum ContainerError {
    InvalidMagic,           // Wrong magic bytes
    ChecksumMismatch,       // Checksum verification failed
    InvalidSegment(String), // Malformed segment
    InvalidFormat(String),  // Container format error
    ParseError(String),     // JSON parsing error
}
```

## Examples

### Complete Example

See [`examples/agi_container_demo.rs`](../crates/rvAgent/rvagent-core/examples/agi_container_demo.rs) for a complete working example.

Run with:
```bash
cargo run --example agi_container_demo
```

### Minimal Example

```rust
use rvagent_core::agi_container::{AgiContainerBuilder, ToolDefinition};
use serde_json::json;

let tool = ToolDefinition {
    name: "test".to_string(),
    description: "Test tool".to_string(),
    parameters: json!({}),
    returns: None,
};

let container = AgiContainerBuilder::new()
    .with_tools(&[tool])
    .build();

assert_eq!(&container[0..4], b"RVF\x01");
```

## Performance

- Container building: O(n) where n is total data size
- Container parsing: O(n) with single pass
- Checksum computation: SHA3-256 (cryptographically secure)
- Memory overhead: Minimal (single allocation for output buffer)

## Compatibility

- Compatible with RVF specification v1
- Supports all segment types defined in RVF
- Extensible via custom tags
- Forward-compatible with future RVF versions

## Integration

The AGI Container integrates with:

- **rvf-bridge**: RVF segment handling and verification
- **session_crypto**: Encryption for sensitive containers
- **state**: Agent state serialization
- **graph**: Agent topology definitions

## Future Enhancements

Planned improvements:

1. **Compression**: Optional compression for large containers
2. **Signatures**: Cryptographic signing with Ed25519
3. **Encryption**: Built-in AES-GCM encryption
4. **Streaming**: Streaming parser for large containers
5. **Validation**: Schema validation for segments
6. **Versioning**: Semantic versioning for containers

## References

- ADR-103: rvAgent Architecture
- RVF Specification v1
- SHA3-256: NIST FIPS 202
