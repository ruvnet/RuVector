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

/// Validate a skill name per ADR-098 / ADR-103 C10.
/// ASCII lowercase alphanumeric + hyphens only.
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
        // ADR-103 C10: ASCII only (not c.is_alphabetic())
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

// Retain only the scalar strings and string-keyed maps used by skill metadata.
// Other YAML values are consumed without coercing numeric keys into strings.
enum FrontmatterValue {
    String(String),
    Mapping(HashMap<String, FrontmatterValue>),
    Other,
}

impl FrontmatterValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_mapping(&self) -> Option<&HashMap<String, Self>> {
        match self {
            Self::Mapping(value) => Some(value),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for FrontmatterValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = FrontmatterValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("YAML skill frontmatter")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::String(value.to_owned()))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::String(value))
            }

            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::Other)
            }

            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::Other)
            }

            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::Other)
            }

            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::Other)
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(FrontmatterValue::Other)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                while sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(FrontmatterValue::Other)
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut strings = HashMap::new();
                while let Some((key, value)) =
                    map.next_entry::<FrontmatterValue, FrontmatterValue>()?
                {
                    if let FrontmatterValue::String(key) = key {
                        strings.insert(key, value);
                    }
                }
                Ok(FrontmatterValue::Mapping(strings))
            }
        }
        deserializer.deserialize_any(Visitor)
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
    let frontmatter_str = after_first[..end_idx].trim_start_matches('\n');

    // Frontmatter size check (ADR-103 C4: max 4KB)
    if frontmatter_str.len() > MAX_FRONTMATTER_SIZE {
        tracing::warn!(
            "Skipping {}: YAML frontmatter too large ({} bytes)",
            skill_path,
            frontmatter_str.len()
        );
        return None;
    }

    // Pure-Rust YAML parsing; filesystem includes and property expansion stay disabled.
    let mut budget = serde_saphyr::Budget::default();
    budget.max_depth = 32;
    budget.max_documents = 1;
    budget.max_aliases = 64;
    budget.max_anchors = 64;
    budget.max_nodes = 4096;
    budget.max_events = 8192;
    budget.max_total_scalar_bytes = 64 * 1024;
    let mut options = serde_saphyr::Options::default();
    options.budget = Some(budget);
    // Match serde_yaml::Value: merge keys are data unless explicitly applied.
    options.merge_keys = serde_saphyr::MergeKeyPolicy::AsOrdinary;
    options.strict_booleans = true;
    let frontmatter: FrontmatterValue =
        serde_saphyr::from_str_with_options(frontmatter_str, options).ok()?;
    let map = frontmatter.as_mapping()?;

    let name = map.get("name")?.as_str()?.trim().to_string();
    let description = map.get("description")?.as_str()?.trim().to_string();

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
        .get("allowed-tools")
        .and_then(|v| v.as_str())
        .map(|s| {
            s.split_whitespace()
                .map(|t| t.trim_matches(',').to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let license = map
        .get("license")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string());

    let compatibility = map
        .get("compatibility")
        .and_then(|v| v.as_str())
        .map(|s| truncate(s.trim(), MAX_SKILL_COMPATIBILITY_LENGTH));

    let metadata = map
        .get("metadata")
        .and_then(|v| v.as_mapping())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.to_string(), v.as_str()?.to_string())))
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

    async fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if crate::json_extension(state, "skills_metadata").is_some() {
            return None;
        }

        let skills = if let Some(preloaded) = &self.preloaded {
            preloaded.clone()
        } else {
            Vec::new()
        };

        let mut update = AgentStateUpdate::default();
        update.extensions.insert(
            "skills_metadata".into(),
            serde_json::to_value(&skills).unwrap_or_default(),
        );
        Some(update)
    }

    async fn wrap_model_call(
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
            return handler.call(request).await;
        }

        let locations = self.format_skills_locations();
        let skills_list = Self::format_skills_list(&skills);
        let section = SKILLS_SYSTEM_PROMPT
            .replace("{skills_locations}", &locations)
            .replace("{skills_list}", &skills_list);

        let new_system = crate::append_to_system_message(&request.system_message, &section);
        handler.call(request.with_system(new_system)).await
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
        assert!(validate_skill_name("MySkill", "MySkill").is_err());
    }

    #[test]
    fn test_validate_skill_name_unicode_rejected() {
        // ADR-103 C10: ASCII-only
        assert!(validate_skill_name("skilll", "skilll").is_ok()); // plain ascii ok
    }

    #[test]
    fn test_validate_skill_name_directory_mismatch() {
        assert!(validate_skill_name("skill-a", "skill-b").is_err());
    }

    #[test]
    fn test_parse_skill_metadata_valid() {
        let content = "---\nname: my-skill\ndescription: A test skill\nlicense: MIT\nallowed-tools: read_file write_file\n---\n# My Skill\nInstructions here.\n";
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
        let content = "---\nname: test\ndescription: Test\nallowed-tools: read_file, write_file, execute\n---\ncontent\n";
        let meta = parse_skill_metadata(content, "path", "test");
        assert!(meta.is_some());
        let tools = meta.unwrap().allowed_tools;
        assert_eq!(tools, vec!["read_file", "write_file", "execute"]);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        let result = truncate("a long string here", 10);
        assert!(result.len() <= 10);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_middleware_name() {
        let mw = SkillsMiddleware::new(vec![".skills".into()]);
        assert_eq!(mw.name(), "skills");
    }

    #[tokio::test]
    async fn test_before_agent_skip_if_loaded() {
        let mw = SkillsMiddleware::new(vec![]);
        let mut state = AgentState::default();
        state.set_extension("skills_metadata", serde_json::json!([]));
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).await.is_none());
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
        let content = "---\nname: my-skill\ndescription: Test\nmetadata:\n  version: \"1.0\"\n  author: test\n---\ncontent\n";
        let meta = parse_skill_metadata(content, "path", "my-skill").unwrap();
        assert_eq!(meta.metadata.get("version"), Some(&"1.0".to_string()));
        assert_eq!(meta.metadata.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_yaml_metadata_preserves_string_only_filtering() {
        let content = "---\nname: skill\ndescription: Test\nmetadata:\n  4: ignored\n  good: value\n  count: 4\n  flag: true\n  items: [one, two]\nunknown: { [a, b]: ignored }\n---\n";
        let meta = parse_skill_metadata(content, "path", "skill").unwrap();
        assert_eq!(
            meta.metadata,
            HashMap::from([("good".into(), "value".into())])
        );
    }

    #[test]
    fn test_yaml_alias_and_block_scalar() {
        let content = "---\nname: skill\ndescription: &description >-\n  useful\n  skill\nmetadata:\n  copy: *description\n---\n";
        let meta = parse_skill_metadata(content, "path", "skill").unwrap();
        assert_eq!(meta.description, "useful skill");
        assert_eq!(meta.metadata["copy"], "useful skill");
    }

    #[test]
    fn test_yaml_rejects_duplicate_and_malformed_frontmatter() {
        for yaml in [
            "name: skill\nname: other\ndescription: Test",
            "name: [broken",
            "name: 5\ndescription: Test",
        ] {
            assert!(
                parse_skill_metadata(&format!("---\n{yaml}\n---\n"), "path", "skill").is_none()
            );
        }
    }

    #[test]
    fn test_yaml_rejects_excessive_nesting_and_aliases() {
        let nested = format!(
            "---\nname: skill\ndescription: Test\nignored: {}0{}\n---\n",
            "[".repeat(40),
            "]".repeat(40)
        );
        assert!(parse_skill_metadata(&nested, "path", "skill").is_none());
        let aliases = format!(
            "---\nname: skill\ndescription: &a Test\nignored: [{}]\n---\n",
            vec!["*a"; 65].join(",")
        );
        assert!(parse_skill_metadata(&aliases, "path", "skill").is_none());
    }

    #[test]
    fn test_yaml_merge_keys_remain_ordinary_data() {
        let content = "---\nbase: &base {name: skill, description: Test}\n<<: *base\n---\n";
        assert!(parse_skill_metadata(content, "path", "skill").is_none());
        let content = "---\nname: skill\ndescription: Test\nmetadata: {<<: {hidden: ignored}, visible: yes}\n---\n";
        let meta = parse_skill_metadata(content, "path", "skill").unwrap();
        assert_eq!(
            meta.metadata,
            HashMap::from([("visible".into(), "yes".into())])
        );
    }
}
