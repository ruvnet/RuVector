//! MemoryMiddleware — loads AGENTS.md content and appends to system prompt.
//! Implements trust verification (ADR-103 C4): hash check, content size limit (1MB),
//! SecurityPolicy field for untrusted file loading.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

use crate::{
    AgentState, AgentStateUpdate, Middleware, ModelHandler, ModelRequest, ModelResponse,
    RunnableConfig, Runtime,
};

/// Maximum content size for memory files (1MB per ADR-103 C4).
pub const MAX_MEMORY_FILE_SIZE: usize = 1024 * 1024;

/// Security policy controlling how untrusted files are loaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityPolicy {
    /// Only load files matching the manifest hash.
    TrustedOnly,
    /// Load all files but warn on hash mismatch.
    WarnUntrusted,
    /// Load all files without verification (development only).
    Permissive,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self::WarnUntrusted
    }
}

/// Entry in the trusted manifest: path -> expected SHA3-256 hash.
#[derive(Debug, Clone, Default)]
pub struct TrustManifest {
    pub entries: HashMap<String, String>,
}

impl TrustManifest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trusted entry with its expected hash.
    pub fn add(&mut self, path: impl Into<String>, hash: impl Into<String>) {
        self.entries.insert(path.into(), hash.into());
    }

    /// Verify content against the manifest entry for the given path.
    pub fn verify(&self, path: &str, content: &[u8]) -> TrustVerification {
        match self.entries.get(path) {
            None => TrustVerification::NotInManifest,
            Some(expected_hash) => {
                let actual_hash = compute_sha3_256(content);
                if actual_hash == *expected_hash {
                    TrustVerification::Trusted
                } else {
                    TrustVerification::HashMismatch {
                        expected: expected_hash.clone(),
                        actual: actual_hash,
                    }
                }
            }
        }
    }
}

/// Result of trust verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustVerification {
    Trusted,
    NotInManifest,
    HashMismatch { expected: String, actual: String },
}

/// Compute SHA3-256 hash of content, returning hex string.
///
/// Uses a pre-allocated buffer to avoid 32 intermediate String allocations.
pub fn compute_sha3_256(content: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(content);
    let result = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in result.iter() {
        use std::fmt::Write;
        write!(hex, "{:02x}", b).unwrap();
    }
    hex
}

/// System prompt template for memory context.
pub const MEMORY_SYSTEM_PROMPT: &str = r#"<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
The above <agent_memory> was loaded in from files in your filesystem.
These files contain important context, guidelines, and learned patterns.
You should follow any instructions or patterns described in the memory files.
If the memory contains coding conventions, style guides, or architectural decisions,
apply them consistently in your work.
</memory_guidelines>"#;

/// Middleware that loads AGENTS.md content and appends it to the system prompt.
pub struct MemoryMiddleware {
    /// Paths to memory source files (e.g., ["AGENTS.md"]).
    sources: Vec<String>,
    /// Security policy for file loading.
    pub security_policy: SecurityPolicy,
    /// Trust manifest for hash verification.
    pub manifest: TrustManifest,
    /// Pre-loaded memory contents (for testing or cached scenarios).
    preloaded: Option<HashMap<String, String>>,
}

impl MemoryMiddleware {
    pub fn new(sources: Vec<String>) -> Self {
        Self {
            sources,
            security_policy: SecurityPolicy::default(),
            manifest: TrustManifest::new(),
            preloaded: None,
        }
    }

    pub fn with_security_policy(mut self, policy: SecurityPolicy) -> Self {
        self.security_policy = policy;
        self
    }

    pub fn with_manifest(mut self, manifest: TrustManifest) -> Self {
        self.manifest = manifest;
        self
    }

    /// Set pre-loaded memory contents (useful for testing).
    pub fn with_preloaded(mut self, contents: HashMap<String, String>) -> Self {
        self.preloaded = Some(contents);
        self
    }

    /// Get the configured memory source paths.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Validate and filter memory content based on security policy.
    fn validate_content(&self, path: &str, content: &str) -> Option<String> {
        // Size limit check (ADR-103 C4: max 1MB)
        if content.len() > MAX_MEMORY_FILE_SIZE {
            tracing::warn!(
                "Memory file {} exceeds size limit ({} > {} bytes), skipping",
                path,
                content.len(),
                MAX_MEMORY_FILE_SIZE
            );
            return None;
        }

        // Trust verification
        let verification = self.manifest.verify(path, content.as_bytes());
        match (&self.security_policy, &verification) {
            (SecurityPolicy::TrustedOnly, TrustVerification::Trusted) => Some(content.to_string()),
            (SecurityPolicy::TrustedOnly, _) => {
                tracing::warn!(
                    "Memory file {} failed trust verification ({:?}), skipping (policy: TrustedOnly)",
                    path, verification
                );
                None
            }
            (SecurityPolicy::WarnUntrusted, TrustVerification::HashMismatch { .. }) => {
                tracing::warn!(
                    "Memory file {} has hash mismatch ({:?}), loading with warning",
                    path,
                    verification
                );
                Some(content.to_string())
            }
            (_, _) => Some(content.to_string()),
        }
    }

    /// Format loaded memory contents into the system prompt section.
    fn format_agent_memory(contents: &HashMap<String, String>) -> String {
        let mut memory_text = String::new();
        for (path, content) in contents {
            memory_text.push_str(&format!(
                "<memory_file path=\"{}\">\n{}\n</memory_file>\n",
                path, content
            ));
        }
        MEMORY_SYSTEM_PROMPT.replace("{agent_memory}", &memory_text)
    }
}

#[async_trait]
impl Middleware for MemoryMiddleware {
    fn name(&self) -> &str {
        "memory"
    }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if state.extensions.contains_key("memory_contents") {
            return None;
        }

        let contents = if let Some(preloaded) = &self.preloaded {
            preloaded
                .iter()
                .filter_map(|(path, content)| {
                    self.validate_content(path, content)
                        .map(|c| (path.clone(), c))
                })
                .collect()
        } else {
            HashMap::new()
        };

        let mut update = AgentStateUpdate::default();
        update.extensions.insert(
            "memory_contents".into(),
            serde_json::to_value(&contents).unwrap_or_default(),
        );
        Some(update)
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        let contents: HashMap<String, String> = request
            .extensions
            .get("memory_contents")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        if contents.is_empty() {
            return handler.call(request);
        }

        let memory_section = Self::format_agent_memory(&contents);
        let new_system = crate::append_to_system_message(&request.system_message, &memory_section);
        handler.call(request.with_system(new_system))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PassthroughHandler;
    impl ModelHandler for PassthroughHandler {
        fn call(&self, request: ModelRequest) -> ModelResponse {
            ModelResponse::text(request.system_message.unwrap_or_default())
        }
    }

    #[test]
    fn test_middleware_name() {
        let mw = MemoryMiddleware::new(vec!["AGENTS.md".into()]);
        assert_eq!(mw.name(), "memory");
    }

    #[test]
    fn test_compute_sha3_256() {
        let hash = compute_sha3_256(b"hello");
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, compute_sha3_256(b"hello"));
        assert_ne!(hash, compute_sha3_256(b"world"));
    }

    #[test]
    fn test_trust_manifest_verify() {
        let mut manifest = TrustManifest::new();
        let hash = compute_sha3_256(b"trusted content");
        manifest.add("AGENTS.md", hash);

        assert_eq!(
            manifest.verify("AGENTS.md", b"trusted content"),
            TrustVerification::Trusted
        );

        match manifest.verify("AGENTS.md", b"tampered content") {
            TrustVerification::HashMismatch { .. } => {}
            other => panic!("Expected HashMismatch, got {:?}", other),
        }

        assert_eq!(
            manifest.verify("other.md", b"anything"),
            TrustVerification::NotInManifest
        );
    }

    #[test]
    fn test_content_size_limit() {
        let mw = MemoryMiddleware::new(vec![]);
        let small = "x".repeat(100);
        assert!(mw.validate_content("test.md", &small).is_some());

        let too_large = "x".repeat(MAX_MEMORY_FILE_SIZE + 1);
        assert!(mw.validate_content("test.md", &too_large).is_none());
    }

    #[test]
    fn test_security_policy_trusted_only() {
        let mut manifest = TrustManifest::new();
        manifest.add("AGENTS.md", compute_sha3_256(b"content"));

        let mw = MemoryMiddleware::new(vec![])
            .with_security_policy(SecurityPolicy::TrustedOnly)
            .with_manifest(manifest);

        assert!(mw.validate_content("AGENTS.md", "content").is_some());
        assert!(mw.validate_content("AGENTS.md", "tampered").is_none());
        assert!(mw.validate_content("other.md", "anything").is_none());
    }

    #[test]
    fn test_security_policy_warn_untrusted() {
        let mut manifest = TrustManifest::new();
        manifest.add("AGENTS.md", compute_sha3_256(b"content"));

        let mw = MemoryMiddleware::new(vec![])
            .with_security_policy(SecurityPolicy::WarnUntrusted)
            .with_manifest(manifest);

        assert!(mw.validate_content("AGENTS.md", "tampered").is_some());
    }

    #[test]
    fn test_security_policy_permissive() {
        let mw = MemoryMiddleware::new(vec![]).with_security_policy(SecurityPolicy::Permissive);

        assert!(mw.validate_content("any.md", "anything").is_some());
    }

    #[test]
    fn test_before_agent_skip_if_loaded() {
        let mw = MemoryMiddleware::new(vec!["AGENTS.md".into()]);
        let mut state = AgentState::default();
        state
            .extensions
            .insert("memory_contents".into(), serde_json::json!({}));
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_before_agent_loads() {
        let mut preloaded = HashMap::new();
        preloaded.insert("AGENTS.md".into(), "Memory content".into());

        let mw = MemoryMiddleware::new(vec!["AGENTS.md".into()]).with_preloaded(preloaded);
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());
        assert!(update.unwrap().extensions.contains_key("memory_contents"));
    }

    #[test]
    fn test_format_agent_memory() {
        let mut contents = HashMap::new();
        contents.insert("AGENTS.md".into(), "Be helpful.".into());
        let formatted = MemoryMiddleware::format_agent_memory(&contents);
        assert!(formatted.contains("<agent_memory>"));
        assert!(formatted.contains("Be helpful."));
        assert!(formatted.contains("<memory_guidelines>"));
    }

    #[test]
    fn test_wrap_model_call_no_memory() {
        let mw = MemoryMiddleware::new(vec![]);
        let request = ModelRequest::new(vec![]);
        let handler = PassthroughHandler;
        let response = mw.wrap_model_call(request, &handler);
        assert!(response.message.content.is_empty());
    }

    #[test]
    fn test_default_security_policy() {
        assert_eq!(SecurityPolicy::default(), SecurityPolicy::WarnUntrusted);
    }
}
