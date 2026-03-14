//! SkillsMiddleware — loads SKILL.md files with YAML frontmatter.
//! ASCII-only skill names (ADR-103 C10), YAML frontmatter max 4KB,
//! skill file max 1MB (ADR-103 C4).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    AgentState, AgentStateUpdate, Middleware, ModelHandler, ModelRequest, ModelResponse,
    RunnableConfig, Runtime,
};

/// Maximum skill name length.
pub const MAX_SKILL_NAME_LENGTH: usize = 64;

/// Maximum skill description length.
pub const MAX_SKILL_DESCRIPTION_LENGTH: usize = 1024;

/// Maximum skill compatibility field length.
pub const MAX_SKILL_COMPATIBILITY_LENGTH: usize = 500;

/// Maximum YAML frontmatter size (ADR-103 C4: 4KB).
pub const MAX_FRONTMATTER_SIZE: usize = 4 * 1024;

/// Maximum skill file size (ADR-103 C4: 1MB, down from 10MB).
pub const MAX_SKILL_FILE_SIZE: usize = 1024 * 1024;

/// Skill metadata parsed from YAML frontmatter.
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

/// System prompt template for skills.
pub const SKILLS_SYSTEM_PROMPT: &str = r#"<skills>
<skills_locations>
{skills_locations}
</skills_locations>

<available_skills>
{skills_list}
</available_skills>

When a user's request matches one of the available skills, read the full skill file
for detailed instructions before proceeding.
</skills>"#;

/// Validate a skill name per the Agent Skills specification (ADR-103 C10).
///
/// Constraints:
/// - 1-64 characters
/// - ASCII lowercase alphanumeric + hyphens only (c.is_ascii_lowercase() per C10)
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
        if c == '-' {
            continue;
        }
        // ADR-103 C10: ASCII-only to prevent Unicode confusable attacks
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            continue;
        }
        return Err("name must be lowercase alphanumeric with single hyphens only".into());
    }
    if name != directory_name {
        return Err(format!(
            "name '{}' must match directory name '{}'",
            name, directory_name
        ));
    }
    Ok(())
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Parse skill metadata from YAML frontmatter in a SKILL.md file.
///
/// Returns `None` if the file is too large, has no frontmatter, or frontmatter is invalid.
pub fn parse_skill_metadata(
    content: &str,
    skill_path: &str,
    directory_name: &str,
) -> Option<SkillMetadata> {
    // File size check (ADR-103 C4: max 1MB)
    if content.len() > MAX_SKILL_FILE_SIZE {
        tracing::warn!(
            "Skipping {}: content too large ({} bytes)",
            skill_path,
            content.len()
        );
        return None;
    }

    // Find YAML frontmatter between --- delimiters
    if !content.starts_with("---") {
        return None;
    }

    let after_first = &content[3..];
    let end_idx = after_first.find("\n---")?;
    let frontmatter_str = &after_first[..end_idx].trim_start_matches('\n');

    // Frontmatter size check (ADR-103 C4: max 4KB)
    if frontmatter_str.len() > MAX_FRONTMATTER_SIZE {
        tracing::warn!(
            "Skipping {}: YAML frontmatter too large ({} bytes)",
            skill_path,
            frontmatter_str.len()
        );
        return None;
    }

    let frontmatter: serde_yaml::Value = serde_yaml::from_str(frontmatter_str).ok()?;
    let map = frontmatter.as_mapping()?;

    let name = map
        .get(&serde_yaml::Value::String("name".into()))?
        .as_str()?
        .trim()
        .to_string();
    let description = map
        .get(&serde_yaml::Value::String("description".into()))?
        .as_str()?
        .trim()
        .to_string();

    // Validate skill name (warn but continue for backwards compatibility)
    if let Err(err) = validate_skill_name(&name, directory_name) {
        tracing::warn!(
            "Skill '{}' in {} does not follow spec: {}",
            name,
            skill_path,
            err
        );
    }

    // Parse allowed-tools (space-delimited string, strip commas)
    let allowed_tools = map
        .get(&serde_yaml::Value::String("allowed-tools".into()))
        .and_then(|v| v.as_str())
        .map(|s| {
            s.split_whitespace()
                .map(|t| t.trim_matches(',').to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let license = map
        .get(&serde_yaml::Value::String("license".into()))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string());

    let compatibility = map
        .get(&serde_yaml::Value::String("compatibility".into()))
        .and_then(|v| v.as_str())
        .map(|s| truncate(s.trim(), MAX_SKILL_COMPATIBILITY_LENGTH));

    // Parse metadata field (key-value pairs)
    let metadata = map
        .get(&serde_yaml::Value::String("metadata".into()))
        .and_then(|v| v.as_mapping())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    Some((k.as_str()?.to_string(), v.as_str()?.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    Some(SkillMetadata {
        path: skill_path.to_string(),
        name,
        description: truncate(&description, MAX_SKILL_DESCRIPTION_LENGTH),
        license,
        compatibility,
        metadata,
        allowed_tools,
    })
}

/// Middleware that loads SKILL.md files and injects their descriptions into the system prompt.
pub struct SkillsMiddleware {
    /// Paths to skill source directories.
    sources: Vec<String>,
    /// Pre-loaded skills (for testing).
    preloaded: Option<Vec<SkillMetadata>>,
}

impl SkillsMiddleware {
    pub fn new(sources: Vec<String>) -> Self {
        Self {
            sources,
            preloaded: None,
        }
    }

    /// Set pre-loaded skills (useful for testing).
    pub fn with_preloaded(mut self, skills: Vec<SkillMetadata>) -> Self {
        self.preloaded = Some(skills);
        self
    }

    fn format_skills_locations(&self) -> String {
        self.sources
            .iter()
            .map(|s| format!("- {}", s))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_skills_list(skills: &[SkillMetadata]) -> String {
        skills
            .iter()
            .map(|s| format!("- **{}**: {} (path: {})", s.name, s.description, s.path))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Middleware for SkillsMiddleware {
    fn name(&self) -> &str {
        "skills"
    }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if state.extensions.contains_key("skills_metadata") {
            return None;
        }

        let skills = if let Some(preloaded) = &self.preloaded {
            preloaded.clone()
        } else {
            // In production, would scan skill directories via backend here.
            Vec::new()
        };

        let mut update = AgentStateUpdate::default();
        update.extensions.insert(
            "skills_metadata".into(),
            serde_json::to_value(&skills).unwrap_or_default(),
        );
        Some(update)
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let skills: Vec<SkillMetadata> = request
            .extensions
            .get("skills_metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        if skills.is_empty() {
            return handler.call(request);
        }

        let locations = self.format_skills_locations();
        let skills_list = Self::format_skills_list(&skills);
        let section = SKILLS_SYSTEM_PROMPT
            .replace("{skills_locations}", &locations)
            .replace("{skills_list}", &skills_list);

        let new_system =
            crate::append_to_system_message(&request.system_message, &section);
        handler.call(request.with_system(new_system))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_skill_name_valid() {
        assert!(validate_skill_name("my-skill", "my-skill").is_ok());
        assert!(validate_skill_name("skill123", "skill123").is_ok());
        assert!(validate_skill_name("a", "a").is_ok());
    }

    #[test]
    fn test_validate_skill_name_empty() {
        assert!(validate_skill_name("", "").is_err());
    }

    #[test]
    fn test_validate_skill_name_too_long() {
        let long = "a".repeat(65);
        assert!(validate_skill_name(&long, &long).is_err());
    }

    #[test]
    fn test_validate_skill_name_leading_hyphen() {
        assert!(validate_skill_name("-skill", "-skill").is_err());
    }

    #[test]
    fn test_validate_skill_name_trailing_hyphen() {
        assert!(validate_skill_name("skill-", "skill-").is_err());
    }

    #[test]
    fn test_validate_skill_name_consecutive_hyphens() {
        assert!(validate_skill_name("my--skill", "my--skill").is_err());
    }

    #[test]
    fn test_validate_skill_name_uppercase_rejected() {
        // ADR-103 C10: ASCII-only, no uppercase
        assert!(validate_skill_name("MySkill", "MySkill").is_err());
    }

    #[test]
    fn test_validate_skill_name_unicode_rejected() {
        // ADR-103 C10: ASCII-only — Unicode lowercase letters rejected
        assert!(validate_skill_name("skíll", "skíll").is_err());
        assert!(validate_skill_name("скилл", "скилл").is_err()); // Cyrillic
    }

    #[test]
    fn test_validate_skill_name_directory_mismatch() {
        assert!(validate_skill_name("skill-a", "skill-b").is_err());
    }

    #[test]
    fn test_parse_skill_metadata_valid() {
        let content = r#"---
name: my-skill
description: A test skill
license: MIT
allowed-tools: read_file write_file
---
# My Skill
Instructions here.
"#;
        let meta = parse_skill_metadata(content, ".skills/my-skill/SKILL.md", "my-skill");
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.name, "my-skill");
        assert_eq!(meta.description, "A test skill");
        assert_eq!(meta.license, Some("MIT".into()));
        assert_eq!(meta.allowed_tools, vec!["read_file", "write_file"]);
    }

    #[test]
    fn test_parse_skill_metadata_no_frontmatter() {
        let content = "# Just a markdown file\nNo frontmatter.";
        assert!(parse_skill_metadata(content, "path", "dir").is_none());
    }

    #[test]
    fn test_parse_skill_metadata_too_large() {
        let content = format!("---\nname: x\n---\n{}", "x".repeat(MAX_SKILL_FILE_SIZE + 1));
        assert!(parse_skill_metadata(&content, "path", "dir").is_none());
    }

    #[test]
    fn test_parse_skill_metadata_frontmatter_too_large() {
        let large_desc = "x".repeat(MAX_FRONTMATTER_SIZE + 1);
        let content = format!("---\nname: x\ndescription: {}\n---\ncontent", large_desc);
        assert!(parse_skill_metadata(&content, "path", "dir").is_none());
    }

    #[test]
    fn test_parse_skill_metadata_with_commas_in_tools() {
        let content = r#"---
name: test
description: Test
allowed-tools: read_file, write_file, execute
---
content
"#;
        let meta = parse_skill_metadata(content, "path", "test");
        assert!(meta.is_some());
        let tools = meta.unwrap().allowed_tools;
        assert_eq!(tools, vec!["read_file", "write_file", "execute"]);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a long string", 10), "a long...");
    }

    #[test]
    fn test_middleware_name() {
        let mw = SkillsMiddleware::new(vec![".skills".into()]);
        assert_eq!(mw.name(), "skills");
    }

    #[test]
    fn test_before_agent_skip_if_loaded() {
        let mw = SkillsMiddleware::new(vec![]);
        let mut state = AgentState::default();
        state
            .extensions
            .insert("skills_metadata".into(), serde_json::json!([]));
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_format_skills_list() {
        let skills = vec![SkillMetadata {
            path: ".skills/test/SKILL.md".into(),
            name: "test".into(),
            description: "A test skill".into(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
        }];
        let list = SkillsMiddleware::format_skills_list(&skills);
        assert!(list.contains("test"));
        assert!(list.contains("A test skill"));
    }

    #[test]
    fn test_validate_skill_name_digits_only() {
        assert!(validate_skill_name("123", "123").is_ok());
    }

    #[test]
    fn test_validate_skill_name_max_length() {
        let name = "a".repeat(64);
        assert!(validate_skill_name(&name, &name).is_ok());
    }

    #[test]
    fn test_parse_skill_metadata_with_metadata_field() {
        let content = r#"---
name: my-skill
description: Test
metadata:
  version: "1.0"
  author: test
---
content
"#;
        let meta = parse_skill_metadata(content, "path", "my-skill").unwrap();
        assert_eq!(meta.metadata.get("version"), Some(&"1.0".to_string()));
        assert_eq!(meta.metadata.get("author"), Some(&"test".to_string()));
    }
}
