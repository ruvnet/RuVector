//! Skills middleware stub.
use async_trait::async_trait;
use crate::Middleware;

/// Maximum skill name length.
pub const MAX_SKILL_NAME_LENGTH: usize = 64;
/// Maximum skill file size in bytes.
pub const MAX_SKILL_FILE_SIZE: usize = 10 * 1024 * 1024;

/// Validate a skill name per ADR-098 / ADR-103 C10.
/// ASCII lowercase alphanumeric + hyphens only.
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
        // ADR-103 C10: ASCII only (not c.is_alphabetic())
        if c.is_ascii_lowercase() || c.is_ascii_digit() { continue; }
        return Err("name must be lowercase alphanumeric with single hyphens only".into());
    }
    if name != directory_name {
        return Err(format!("name '{}' must match directory name '{}'", name, directory_name));
    }
    Ok(())
}

pub struct SkillsMiddleware {
    sources: Vec<String>,
}
impl SkillsMiddleware {
    pub fn new(sources: Vec<String>) -> Self { Self { sources } }
}
#[async_trait]
impl Middleware for SkillsMiddleware {
    fn name(&self) -> &str { "skills" }
}
