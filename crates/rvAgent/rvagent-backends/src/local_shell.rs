//! LocalShellBackend — filesystem backend with shell execution (ADR-103 C2).
//!
//! Extends `FilesystemBackend` with `execute()` using `tokio::process::Command`.
//! Implements environment sanitization, optional command allowlisting,
//! configurable timeout, and output truncation.

use crate::filesystem::FilesystemBackend;
use crate::protocol::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Environment variable name patterns that must be stripped (ADR-103 C2/SEC-005).
const SENSITIVE_ENV_PATTERNS: &[&str] = &[
    "SECRET",
    "KEY",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "AWS_",
    "AZURE_",
    "GCP_",
    "DATABASE_URL",
    "PRIVATE",
    "API_KEY",
    "AUTH",
    "BEARER",
    "JWT",
    "SESSION",
];

/// Safe environment variables that are explicitly allowed.
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ",
    "EDITOR", "HOSTNAME",
];

/// Optional command allowlist configuration.
#[derive(Debug, Clone, Default)]
pub struct CommandAllowlist {
    /// If non-empty, only these command prefixes are allowed.
    pub allowed_prefixes: Vec<String>,
}

impl CommandAllowlist {
    /// Create a new allowlist with the given command prefixes.
    pub fn new(prefixes: Vec<String>) -> Self {
        Self {
            allowed_prefixes: prefixes,
        }
    }

    /// Check if a command is allowed by this allowlist.
    /// Returns `true` if the allowlist is empty (all commands allowed)
    /// or if the command matches one of the allowed prefixes.
    pub fn is_allowed(&self, command: &str) -> bool {
        if self.allowed_prefixes.is_empty() {
            return true;
        }
        let trimmed = command.trim();
        self.allowed_prefixes
            .iter()
            .any(|prefix| trimmed.starts_with(prefix.as_str()))
    }
}

/// Configuration for the local shell backend.
#[derive(Debug, Clone)]
pub struct LocalShellConfig {
    /// Default command timeout in seconds.
    pub default_timeout_secs: u32,
    /// Maximum output size in bytes before truncation.
    pub max_output_bytes: usize,
    /// Optional command allowlist.
    pub allowlist: Option<CommandAllowlist>,
    /// Additional safe environment variables to pass through.
    pub extra_env: HashMap<String, String>,
}

impl Default for LocalShellConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 30,
            max_output_bytes: 1024 * 1024, // 1 MB
            allowlist: None,
            extra_env: HashMap::new(),
        }
    }
}

/// Local shell backend with execution hardening.
///
/// - Environment sanitization: strips SECRET, KEY, TOKEN, etc. (SEC-005)
/// - `env_clear()` + explicit safe env (SEC-008)
/// - Optional command allowlist
/// - Configurable timeout
/// - Output truncation at configurable limit
/// - Uses `tokio::process::Command` (ADR-103 A3)
#[derive(Clone)]
pub struct LocalShellBackend {
    inner: FilesystemBackend,
    config: LocalShellConfig,
    sandbox_id: String,
    safe_env: HashMap<String, String>,
}

impl LocalShellBackend {
    /// Create a new local shell backend.
    pub fn new(cwd: PathBuf, config: LocalShellConfig) -> Self {
        let safe_env = build_safe_env(&config.extra_env);
        Self {
            inner: FilesystemBackend::new(cwd),
            config,
            sandbox_id: uuid::Uuid::new_v4().to_string(),
            safe_env,
        }
    }

    /// Create with a specific sandbox ID.
    pub fn with_id(cwd: PathBuf, config: LocalShellConfig, sandbox_id: String) -> Self {
        let safe_env = build_safe_env(&config.extra_env);
        Self {
            inner: FilesystemBackend::new(cwd),
            config,
            sandbox_id,
            safe_env,
        }
    }

    /// Get a reference to the inner filesystem backend.
    pub fn filesystem(&self) -> &FilesystemBackend {
        &self.inner
    }
}

/// Build the sanitized environment map.
///
/// Starts with env_clear() semantics — only passes through SAFE_ENV_VARS
/// from the current environment, then adds extra_env, and filters out
/// any variable matching SENSITIVE_ENV_PATTERNS.
fn build_safe_env(extra_env: &HashMap<String, String>) -> HashMap<String, String> {
    let mut env = HashMap::new();

    // Only include known-safe vars from current environment
    for var_name in SAFE_ENV_VARS {
        if let Ok(val) = std::env::var(var_name) {
            env.insert(var_name.to_string(), val);
        }
    }

    // Add extra env vars (user-provided overrides)
    for (k, v) in extra_env {
        env.insert(k.clone(), v.clone());
    }

    // Strip anything matching sensitive patterns
    env.retain(|key, _| !is_sensitive_env_var(key));

    env
}

/// Check if an environment variable name matches any sensitive pattern.
pub fn is_sensitive_env_var(name: &str) -> bool {
    let upper = name.to_uppercase();
    SENSITIVE_ENV_PATTERNS
        .iter()
        .any(|pattern| upper.contains(pattern))
}

#[async_trait]
impl Backend for LocalShellBackend {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        self.inner.ls_info(path).await
    }

    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        self.inner.read_file(file_path, offset, limit).await
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        self.inner.write_file(file_path, content).await
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        self.inner
            .edit_file(file_path, old_string, new_string, replace_all)
            .await
    }

    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        self.inner.glob_info(pattern, path).await
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        self.inner.grep(pattern, path, include_glob).await
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        self.inner.download_files(paths).await
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        self.inner.upload_files(files).await
    }
}

#[async_trait]
impl SandboxBackend for LocalShellBackend {
    /// Execute a shell command with environment sanitization and timeout.
    ///
    /// Uses `tokio::process::Command` (not `std::process::Command`) per ADR-103 A3.
    /// Applies `env_clear()` + explicit safe env per ADR-103 C2.
    async fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
        // Check allowlist
        if let Some(ref allowlist) = self.config.allowlist {
            if !allowlist.is_allowed(command) {
                return ExecuteResponse {
                    output: format!("Command not allowed: {}", command),
                    exit_code: Some(1),
                    truncated: false,
                };
            }
        }

        let timeout_secs = timeout.unwrap_or(self.config.default_timeout_secs);
        let timeout_duration = Duration::from_secs(timeout_secs as u64);

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.current_dir(self.inner.cwd());

        // env_clear() + explicit safe env (SEC-008)
        cmd.env_clear();
        for (k, v) in &self.safe_env {
            cmd.env(k, v);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return ExecuteResponse {
                    output: format!("Failed to spawn command: {}", e),
                    exit_code: Some(1),
                    truncated: false,
                };
            }
        };

        // Wait with timeout
        let result = tokio::time::timeout(timeout_duration, child.wait_with_output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                // Combine stdout and stderr, prefixing stderr lines
                let mut combined = String::new();
                if !stdout.is_empty() {
                    combined.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    for line in stderr.lines() {
                        combined.push_str("[stderr] ");
                        combined.push_str(line);
                        combined.push('\n');
                    }
                }

                // Truncate if over limit
                let truncated = combined.len() > self.config.max_output_bytes;
                if truncated {
                    combined.truncate(self.config.max_output_bytes);
                    combined.push_str("\n... [output truncated]");
                }

                ExecuteResponse {
                    output: combined,
                    exit_code: output.status.code(),
                    truncated,
                }
            }
            Ok(Err(e)) => ExecuteResponse {
                output: format!("Command failed: {}", e),
                exit_code: Some(1),
                truncated: false,
            },
            Err(_) => ExecuteResponse {
                output: format!("Command timed out after {} seconds", timeout_secs),
                exit_code: None,
                truncated: false,
            },
        }
    }

    fn id(&self) -> &str {
        &self.sandbox_id
    }

    fn sandbox_root(&self) -> &Path {
        self.inner.cwd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, LocalShellBackend) {
        let tmp = TempDir::new().unwrap();
        let config = LocalShellConfig::default();
        let backend = LocalShellBackend::new(tmp.path().to_path_buf(), config);
        (tmp, backend)
    }

    #[test]
    fn test_env_sanitization_strips_secrets() {
        assert!(is_sensitive_env_var("MY_SECRET_KEY"));
        assert!(is_sensitive_env_var("AWS_ACCESS_KEY_ID"));
        assert!(is_sensitive_env_var("AZURE_CLIENT_SECRET"));
        assert!(is_sensitive_env_var("GCP_SERVICE_ACCOUNT"));
        assert!(is_sensitive_env_var("DATABASE_URL"));
        assert!(is_sensitive_env_var("api_token"));
        assert!(is_sensitive_env_var("PRIVATE_KEY"));
        assert!(is_sensitive_env_var("my_password"));
        assert!(is_sensitive_env_var("credential_file"));
        assert!(is_sensitive_env_var("OPENAI_API_KEY"));
        assert!(is_sensitive_env_var("OAUTH_TOKEN"));
        assert!(is_sensitive_env_var("BEARER_TOKEN"));
        assert!(is_sensitive_env_var("JWT_SECRET"));
        assert!(is_sensitive_env_var("SESSION_ID"));
    }

    #[test]
    fn test_env_sanitization_allows_safe_vars() {
        assert!(!is_sensitive_env_var("PATH"));
        assert!(!is_sensitive_env_var("HOME"));
        assert!(!is_sensitive_env_var("USER"));
        assert!(!is_sensitive_env_var("SHELL"));
        assert!(!is_sensitive_env_var("LANG"));
        assert!(!is_sensitive_env_var("TERM"));
    }

    #[test]
    fn test_build_safe_env_excludes_sensitive() {
        let mut extra = HashMap::new();
        extra.insert("MY_SECRET".to_string(), "hidden".to_string());
        extra.insert("CUSTOM_VAR".to_string(), "visible".to_string());

        let env = build_safe_env(&extra);
        assert!(!env.contains_key("MY_SECRET"));
        assert!(env.contains_key("CUSTOM_VAR"));
    }

    #[test]
    fn test_command_allowlist_empty_allows_all() {
        let al = CommandAllowlist::default();
        assert!(al.is_allowed("rm -rf /"));
        assert!(al.is_allowed("ls"));
    }

    #[test]
    fn test_command_allowlist_restricts() {
        let al = CommandAllowlist::new(vec![
            "ls".to_string(),
            "cat".to_string(),
            "grep".to_string(),
        ]);
        assert!(al.is_allowed("ls -la"));
        assert!(al.is_allowed("cat file.txt"));
        assert!(al.is_allowed("grep pattern file"));
        assert!(!al.is_allowed("rm -rf /"));
        assert!(!al.is_allowed("curl evil.com"));
    }

    #[tokio::test]
    async fn test_execute_simple_command() {
        let (_tmp, backend) = setup();
        let result = backend.execute("echo hello", None).await;
        assert_eq!(result.exit_code, Some(0));
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_with_stderr() {
        let (_tmp, backend) = setup();
        let result = backend.execute("echo err >&2", None).await;
        assert!(result.output.contains("[stderr]"));
    }

    #[tokio::test]
    async fn test_execute_exit_code() {
        let (_tmp, backend) = setup();
        let result = backend.execute("exit 42", None).await;
        assert_eq!(result.exit_code, Some(42));
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let tmp = TempDir::new().unwrap();
        let config = LocalShellConfig {
            default_timeout_secs: 1,
            ..Default::default()
        };
        let backend = LocalShellBackend::new(tmp.path().to_path_buf(), config);
        let result = backend.execute("sleep 30", Some(1)).await;
        assert!(result.output.contains("timed out"));
    }

    #[tokio::test]
    async fn test_execute_allowlist_blocked() {
        let tmp = TempDir::new().unwrap();
        let config = LocalShellConfig {
            allowlist: Some(CommandAllowlist::new(vec!["echo".to_string()])),
            ..Default::default()
        };
        let backend = LocalShellBackend::new(tmp.path().to_path_buf(), config);
        let result = backend.execute("rm -rf /", None).await;
        assert!(result.output.contains("not allowed"));
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_execute_truncation() {
        let tmp = TempDir::new().unwrap();
        let config = LocalShellConfig {
            max_output_bytes: 20,
            ..Default::default()
        };
        let backend = LocalShellBackend::new(tmp.path().to_path_buf(), config);
        let result = backend
            .execute(
                "echo 'this is a very long output string that should be truncated'",
                None,
            )
            .await;
        assert!(result.truncated);
        assert!(result.output.contains("[output truncated]"));
    }

    #[tokio::test]
    async fn test_execute_env_cleared() {
        let (_tmp, backend) = setup();
        // The command should not see arbitrary parent env vars
        let result = backend.execute("env", None).await;
        // Should not contain any sensitive patterns from parent env
        for line in result.output.lines() {
            if line.starts_with("[stderr]") {
                continue;
            }
            let var_name = line.split('=').next().unwrap_or("");
            assert!(
                !is_sensitive_env_var(var_name),
                "Sensitive env var leaked: {}",
                line
            );
        }
    }

    #[test]
    fn test_sandbox_id() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalShellBackend::with_id(
            tmp.path().to_path_buf(),
            LocalShellConfig::default(),
            "test-id-123".to_string(),
        );
        assert_eq!(backend.id(), "test-id-123");
    }

    #[test]
    fn test_sandbox_root() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalShellBackend::new(tmp.path().to_path_buf(), LocalShellConfig::default());
        assert_eq!(backend.sandbox_root(), tmp.path());
    }
}
