# ADR-098: Memory, Skills & Summarization Middleware

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crate**   | `ruvector-deep-middleware`                       |

## Context

Three middleware layers handle persistent context and conversation management:

1. **MemoryMiddleware** — Loads AGENTS.md files into system prompt with learning guidelines
2. **SkillsMiddleware** — Progressive disclosure of SKILL.md files with YAML frontmatter
3. **SummarizationMiddleware** — Auto-compact conversations when token budget exceeded

## Decision

### 1. MemoryMiddleware

```rust
/// Python: MemoryMiddleware(AgentMiddleware[MemoryState, ContextT, ResponseT])
pub struct MemoryMiddleware {
    backend: BackendRef,
    sources: Vec<String>,
}

impl MemoryMiddleware {
    pub fn new(backend: BackendRef, sources: Vec<String>) -> Self {
        Self { backend, sources }
    }
}

impl Middleware for MemoryMiddleware {
    fn state_keys(&self) -> Vec<&str> {
        vec!["memory_contents"]
    }

    fn before_agent(
        &self,
        state: &AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) -> Option<AgentState> {
        // Skip if already loaded
        if state.contains_key("memory_contents") {
            return None;
        }

        let backend = self.resolve_backend(state, runtime, config);
        let mut contents: HashMap<String, String> = HashMap::new();

        // Batch download all sources
        let responses = backend.download_files(&self.sources);
        for (path, response) in self.sources.iter().zip(responses.iter()) {
            match (&response.error, &response.content) {
                (Some(FileOperationError::FileNotFound), _) => continue,
                (Some(err), _) => panic!("Failed to download {}: {:?}", path, err),
                (None, Some(content)) => {
                    contents.insert(
                        path.clone(),
                        String::from_utf8(content.clone()).unwrap(),
                    );
                }
                _ => {}
            }
        }

        let mut update = AgentState::new();
        update.insert("memory_contents".into(), serde_json::to_value(&contents).unwrap());
        Some(update)
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        let contents: HashMap<String, String> = request.state
            .get("memory_contents")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let agent_memory = self.format_agent_memory(&contents);
        let new_system = append_to_system_message(&request.system_message, &agent_memory);
        handler(request.override_system(new_system))
    }
}
```

#### Memory System Prompt (Exact Fidelity)

```rust
/// Python: MEMORY_SYSTEM_PROMPT — 156-line prompt template
/// Preserved verbatim including all examples and guidelines.
pub const MEMORY_SYSTEM_PROMPT: &str = r#"<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. ...
    [Full prompt preserved — see Python source memory.py lines 97-156]
</memory_guidelines>
"#;
```

### 2. SkillsMiddleware

```rust
/// Skill metadata parsed from YAML frontmatter.
/// Python: SkillMetadata(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    pub path: String,
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub metadata: HashMap<String, String>,
    pub allowed_tools: Vec<String>,
}

/// Validation constants per Agent Skills specification.
pub const MAX_SKILL_NAME_LENGTH: usize = 64;
pub const MAX_SKILL_DESCRIPTION_LENGTH: usize = 1024;
pub const MAX_SKILL_COMPATIBILITY_LENGTH: usize = 500;
pub const MAX_SKILL_FILE_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Python: SkillsMiddleware(AgentMiddleware[SkillsState, ContextT, ResponseT])
pub struct SkillsMiddleware {
    backend: BackendRef,
    sources: Vec<String>,
}

impl Middleware for SkillsMiddleware {
    fn state_keys(&self) -> Vec<&str> {
        vec!["skills_metadata"]
    }

    fn before_agent(
        &self,
        state: &AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) -> Option<AgentState> {
        if state.contains_key("skills_metadata") {
            return None;
        }

        let backend = self.resolve_backend(state, runtime, config);
        let mut all_skills: HashMap<String, SkillMetadata> = HashMap::new();

        // Load from each source, later sources override earlier (last wins)
        for source_path in &self.sources {
            let skills = list_skills(&*backend, source_path);
            for skill in skills {
                all_skills.insert(skill.name.clone(), skill);
            }
        }

        let skills: Vec<SkillMetadata> = all_skills.into_values().collect();
        let mut update = AgentState::new();
        update.insert("skills_metadata".into(), serde_json::to_value(&skills).unwrap());
        Some(update)
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        let skills: Vec<SkillMetadata> = request.state
            .get("skills_metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let locations = self.format_skills_locations();
        let skills_list = self.format_skills_list(&skills);
        let section = SKILLS_SYSTEM_PROMPT
            .replace("{skills_locations}", &locations)
            .replace("{skills_list}", &skills_list);

        let new_system = append_to_system_message(&request.system_message, &section);
        handler(request.override_system(new_system))
    }
}
```

#### Skill Name Validation

```rust
/// Python: _validate_skill_name(name, directory_name)
/// Constraints per Agent Skills specification:
/// - 1-64 chars, Unicode lowercase alphanumeric + hyphens
/// - No leading/trailing/consecutive hyphens
/// - Must match directory name
pub fn validate_skill_name(name: &str, directory_name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("name is required".into());
    }
    if name.len() > MAX_SKILL_NAME_LENGTH {
        return Err("name exceeds 64 characters".into());
    }
    if name.starts_with('-') || name.ends_with('-') || name.contains("--") {
        return Err("name must be lowercase alphanumeric with single hyphens only".into());
    }
    for c in name.chars() {
        if c == '-' { continue; }
        if (c.is_alphabetic() && c.is_lowercase()) || c.is_ascii_digit() { continue; }
        return Err("name must be lowercase alphanumeric with single hyphens only".into());
    }
    if name != directory_name {
        return Err(format!("name '{}' must match directory name '{}'", name, directory_name));
    }
    Ok(())
}
```

#### YAML Frontmatter Parsing

```rust
/// Python: _parse_skill_metadata(content, skill_path, directory_name)
/// Uses serde_yaml for YAML parsing (Python uses yaml.safe_load)
pub fn parse_skill_metadata(
    content: &str,
    skill_path: &str,
    directory_name: &str,
) -> Option<SkillMetadata> {
    if content.len() > MAX_SKILL_FILE_SIZE {
        warn!("Skipping {}: content too large ({} bytes)", skill_path, content.len());
        return None;
    }

    // Match YAML frontmatter between --- delimiters
    let re = regex::Regex::new(r"^---\s*\n(.*?)\n---\s*\n").unwrap();
    let captures = re.captures(content)?;
    let frontmatter_str = captures.get(1)?.as_str();

    let frontmatter: serde_yaml::Value = serde_yaml::from_str(frontmatter_str).ok()?;
    let map = frontmatter.as_mapping()?;

    let name = map.get("name")?.as_str()?.trim().to_string();
    let description = map.get("description")?.as_str()?.trim().to_string();

    // Validate (warn but continue for backwards compatibility)
    if let Err(err) = validate_skill_name(&name, directory_name) {
        warn!("Skill '{}' in {} does not follow spec: {}", name, skill_path, err);
    }

    // Parse allowed-tools (space-delimited string, strip commas for Claude Code compat)
    let allowed_tools = map.get("allowed-tools")
        .and_then(|v| v.as_str())
        .map(|s| s.split_whitespace()
            .map(|t| t.trim_matches(',').to_string())
            .filter(|t| !t.is_empty())
            .collect())
        .unwrap_or_default();

    Some(SkillMetadata {
        path: skill_path.to_string(),
        name,
        description: truncate(&description, MAX_SKILL_DESCRIPTION_LENGTH),
        license: map.get("license").and_then(|v| v.as_str()).map(|s| s.trim().to_string()),
        compatibility: map.get("compatibility").and_then(|v| v.as_str())
            .map(|s| truncate(s.trim(), MAX_SKILL_COMPATIBILITY_LENGTH)),
        metadata: parse_metadata_field(map.get("metadata"), skill_path),
        allowed_tools,
    })
}
```

### 3. SummarizationMiddleware

```rust
/// Python: SummarizationMiddleware — auto-compact when token budget exceeded
pub struct SummarizationMiddleware {
    model: Box<dyn ChatModel>,
    backend: BackendRef,
    trigger: TriggerConfig,
    keep: KeepConfig,
}

/// Trigger configuration for auto-compaction.
/// Python: trigger=("fraction", 0.85) or ("tokens", 100000)
pub enum TriggerConfig {
    Fraction(f64),  // Fraction of context window
    Tokens(u64),    // Absolute token count
}

/// How much context to keep after compaction.
/// Python: keep=("fraction", 0.10) or ("tokens", 10000)
pub enum KeepConfig {
    Fraction(f64),
    Tokens(u64),
}

impl Middleware for SummarizationMiddleware {
    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        let token_count = estimate_tokens(&request.messages);
        let threshold = self.calculate_threshold(&request);

        if token_count > threshold {
            // Compact: summarize older messages, keep recent ones
            let keep_count = self.calculate_keep_count(&request);
            let (to_summarize, to_keep) = request.messages.split_at(
                request.messages.len().saturating_sub(keep_count)
            );

            let summary = self.summarize(to_summarize);

            // Store full history in backend at /conversation_history/{thread_id}.md
            self.offload_history(&request, to_summarize);

            let compacted_request = request.with_messages(
                vec![summary_message(summary), to_keep.to_vec()].concat()
            );
            handler(compacted_request)
        } else {
            handler(request)
        }
    }
}

/// Python: SummarizationToolMiddleware — compact_conversation tool
pub struct SummarizationToolMiddleware {
    summarization: Arc<SummarizationMiddleware>,
}

impl Middleware for SummarizationToolMiddleware {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(CompactConversationTool {
            summarization: self.summarization.clone(),
        })]
    }
}
```

### 4. PatchToolCallsMiddleware

```rust
/// Python: PatchToolCallsMiddleware — fixes dangling tool calls
pub struct PatchToolCallsMiddleware;

impl Middleware for PatchToolCallsMiddleware {
    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentState> {
        let messages = state.get("messages")?.as_array()?;
        if messages.is_empty() { return None; }

        let mut patched = Vec::new();
        for (i, msg) in messages.iter().enumerate() {
            patched.push(msg.clone());

            // Check if this is an AI message with tool_calls
            if msg["type"] == "ai" {
                if let Some(tool_calls) = msg["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let tc_id = tc["id"].as_str().unwrap_or("");
                        // Check if corresponding ToolMessage exists in remaining messages
                        let has_response = messages[i..].iter().any(|m| {
                            m["type"] == "tool" && m["tool_call_id"] == tc_id
                        });
                        if !has_response {
                            // Add cancellation ToolMessage
                            patched.push(serde_json::json!({
                                "type": "tool",
                                "content": format!(
                                    "Tool call {} with id {} was cancelled - \
                                     another message came in before it could be completed.",
                                    tc["name"].as_str().unwrap_or(""),
                                    tc_id
                                ),
                                "name": tc["name"],
                                "tool_call_id": tc_id,
                            }));
                        }
                    }
                }
            }
        }

        let mut update = AgentState::new();
        update.insert("messages".into(), serde_json::json!({"$overwrite": patched}));
        Some(update)
    }
}
```

## Consequences

- All three content middleware layers preserve exact prompt templates and behavior
- YAML frontmatter parsing uses `serde_yaml` (equivalent to Python's `yaml.safe_load`)
- Skill validation follows Agent Skills specification character-for-character
- Summarization uses same trigger/keep fraction logic with identical offload format
- PatchToolCallsMiddleware patches dangling tool calls identically
