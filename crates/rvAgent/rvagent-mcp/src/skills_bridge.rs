//! Cross-platform skills bridge -- converts between rvAgent skills
//! and external platform formats (Claude Code, OpenAI Codex CLI).

use rvagent_middleware::skills::SkillMetadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Claude Code skill format
// ---------------------------------------------------------------------------

/// Skill format for Claude Code compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeSkill {
    pub name: String,
    pub description: String,
    pub path: String,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    #[serde(default)]
    pub triggers: Vec<String>,
}

// ---------------------------------------------------------------------------
// Codex skill format
// ---------------------------------------------------------------------------

/// Skill format for OpenAI Codex CLI compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexSkill {
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

// ---------------------------------------------------------------------------
// SkillBridge
// ---------------------------------------------------------------------------

/// Bridge between rvAgent skills and external platforms.
pub struct SkillBridge;

impl SkillBridge {
    /// Convert rvAgent skill to Claude Code skill format.
    pub fn to_claude_code(skill: &SkillMetadata) -> ClaudeCodeSkill {
        ClaudeCodeSkill {
            name: skill.name.clone(),
            description: skill.description.clone(),
            path: skill.path.clone(),
            allowed_tools: skill.allowed_tools.clone(),
            triggers: vec![format!("/{}", skill.name)],
        }
    }

    /// Convert rvAgent skill to Codex skill format.
    pub fn to_codex(skill: &SkillMetadata) -> CodexSkill {
        CodexSkill {
            name: skill.name.clone(),
            prompt: skill.description.clone(),
            tools: skill.allowed_tools.clone(),
            model: None,
        }
    }

    /// Convert Claude Code skill to rvAgent format.
    pub fn from_claude_code(skill: &ClaudeCodeSkill) -> SkillMetadata {
        SkillMetadata {
            path: skill.path.clone(),
            name: skill.name.clone(),
            description: skill.description.clone(),
            license: None,
            compatibility: Some("claude-code".into()),
            metadata: HashMap::new(),
            allowed_tools: skill.allowed_tools.clone(),
        }
    }

    /// Convert Codex skill to rvAgent format.
    pub fn from_codex(skill: &CodexSkill) -> SkillMetadata {
        SkillMetadata {
            path: String::new(),
            name: skill.name.clone(),
            description: skill.prompt.clone(),
            license: None,
            compatibility: Some("codex".into()),
            metadata: HashMap::new(),
            allowed_tools: skill.tools.clone(),
        }
    }

    /// Batch convert rvAgent skills to Claude Code format.
    pub fn to_claude_code_batch(skills: &[SkillMetadata]) -> Vec<ClaudeCodeSkill> {
        skills.iter().map(Self::to_claude_code).collect()
    }

    /// Batch convert rvAgent skills to Codex format.
    pub fn to_codex_batch(skills: &[SkillMetadata]) -> Vec<CodexSkill> {
        skills.iter().map(Self::to_codex).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_skill() -> SkillMetadata {
        SkillMetadata {
            path: ".skills/test-skill/SKILL.md".into(),
            name: "test-skill".into(),
            description: "A test skill for unit tests".into(),
            license: Some("MIT".into()),
            compatibility: Some("claude-code".into()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("version".into(), "1.0".into());
                m
            },
            allowed_tools: vec!["read_file".into(), "write_file".into()],
        }
    }

    #[test]
    fn test_to_claude_code() {
        let skill = sample_skill();
        let cc = SkillBridge::to_claude_code(&skill);
        assert_eq!(cc.name, "test-skill");
        assert_eq!(cc.description, "A test skill for unit tests");
        assert_eq!(cc.path, ".skills/test-skill/SKILL.md");
        assert_eq!(cc.allowed_tools, vec!["read_file", "write_file"]);
        assert_eq!(cc.triggers, vec!["/test-skill"]);
    }

    #[test]
    fn test_to_codex() {
        let skill = sample_skill();
        let codex = SkillBridge::to_codex(&skill);
        assert_eq!(codex.name, "test-skill");
        assert_eq!(codex.prompt, "A test skill for unit tests");
        assert_eq!(codex.tools, vec!["read_file", "write_file"]);
        assert!(codex.model.is_none());
    }

    #[test]
    fn test_from_claude_code() {
        let cc = ClaudeCodeSkill {
            name: "my-skill".into(),
            description: "My skill".into(),
            path: "/skills/my-skill/SKILL.md".into(),
            allowed_tools: vec!["ls".into()],
            triggers: vec!["/my-skill".into()],
        };
        let meta = SkillBridge::from_claude_code(&cc);
        assert_eq!(meta.name, "my-skill");
        assert_eq!(meta.description, "My skill");
        assert_eq!(meta.path, "/skills/my-skill/SKILL.md");
        assert_eq!(meta.compatibility.as_deref(), Some("claude-code"));
        assert_eq!(meta.allowed_tools, vec!["ls"]);
    }

    #[test]
    fn test_from_codex() {
        let codex = CodexSkill {
            name: "codex-skill".into(),
            prompt: "Do something".into(),
            tools: vec!["execute".into()],
            model: Some("gpt-4".into()),
        };
        let meta = SkillBridge::from_codex(&codex);
        assert_eq!(meta.name, "codex-skill");
        assert_eq!(meta.description, "Do something");
        assert_eq!(meta.compatibility.as_deref(), Some("codex"));
        assert_eq!(meta.allowed_tools, vec!["execute"]);
        assert!(meta.path.is_empty());
    }

    #[test]
    fn test_roundtrip_claude_code() {
        let original = sample_skill();
        let cc = SkillBridge::to_claude_code(&original);
        let back = SkillBridge::from_claude_code(&cc);
        assert_eq!(back.name, original.name);
        assert_eq!(back.description, original.description);
        assert_eq!(back.path, original.path);
        assert_eq!(back.allowed_tools, original.allowed_tools);
    }

    #[test]
    fn test_roundtrip_codex() {
        let original = sample_skill();
        let codex = SkillBridge::to_codex(&original);
        let back = SkillBridge::from_codex(&codex);
        assert_eq!(back.name, original.name);
        assert_eq!(back.description, original.description);
        assert_eq!(back.allowed_tools, original.allowed_tools);
    }

    #[test]
    fn test_claude_code_skill_serde() {
        let cc = SkillBridge::to_claude_code(&sample_skill());
        let json = serde_json::to_string(&cc).unwrap();
        let back: ClaudeCodeSkill = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, cc.name);
        assert_eq!(back.triggers, cc.triggers);
    }

    #[test]
    fn test_codex_skill_serde() {
        let codex = SkillBridge::to_codex(&sample_skill());
        let json = serde_json::to_string(&codex).unwrap();
        let back: CodexSkill = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, codex.name);
        assert!(back.model.is_none());
    }

    #[test]
    fn test_batch_claude_code() {
        let skills = vec![sample_skill(), sample_skill()];
        let batch = SkillBridge::to_claude_code_batch(&skills);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_batch_codex() {
        let skills = vec![sample_skill()];
        let batch = SkillBridge::to_codex_batch(&skills);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_empty_skill_conversion() {
        let skill = SkillMetadata {
            path: String::new(),
            name: String::new(),
            description: String::new(),
            license: None,
            compatibility: None,
            metadata: HashMap::new(),
            allowed_tools: vec![],
        };
        let cc = SkillBridge::to_claude_code(&skill);
        assert!(cc.name.is_empty());
        assert_eq!(cc.triggers, vec!["/"]);

        let codex = SkillBridge::to_codex(&skill);
        assert!(codex.prompt.is_empty());
    }
}
