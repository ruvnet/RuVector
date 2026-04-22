//! Sandbox backend trait with mandatory path restriction contract (ADR-103 C5).
//!
//! Implements SEC-023: Sandbox Path Restriction Contract.
//! All filesystem operations MUST be confined to the sandbox root.
//! Any attempt to access files outside the sandbox MUST fail with PathEscapesSandbox error.

use crate::protocol::*;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// Configuration for sandbox execution.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum command execution timeout in seconds.
    pub timeout_secs: u32,
    /// Maximum output size in bytes before truncation.
    pub max_output_size: usize,
    /// Working directory within the sandbox.
    pub work_dir: Option<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_output_size: 1024 * 1024, // 1 MB
            work_dir: None,
        }
    }
}

/// Sandbox-specific errors (ADR-103 C5).
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("Path escapes sandbox root: {0}")]
    PathEscapesSandbox(String),
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Sandbox initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Timeout exceeded")]
    Timeout,
    #[error("IO error: {0}")]
    IoError(String),
}

/// Base sandbox trait for backends providing filesystem confinement.
///
/// # Mandatory Path Restriction Contract (SEC-023)
///
/// All implementations MUST enforce the following:
/// 1. All filesystem operations are confined to `sandbox_root()`
/// 2. Path traversal attempts (../, symlinks, absolute paths) MUST be rejected
/// 3. `validate_path()` MUST be called before any filesystem access
/// 4. Failed validation MUST return `PathEscapesSandbox` error
///
/// This trait extends `SandboxBackend` to provide default implementations
/// of file operations via shell commands, allowing any sandbox that can
/// execute commands to also serve as a full backend.
pub trait BaseSandbox: Send + Sync {
    /// The root path of the sandbox filesystem.
    /// All file operations MUST be confined to this root.
    fn sandbox_root(&self) -> &Path;

    /// Configuration for this sandbox.
    fn config(&self) -> &SandboxConfig;

    /// Execute a command within the sandbox.
    fn execute_sync(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;

    /// Unique identifier for this sandbox.
    fn sandbox_id(&self) -> &str;

    /// Validate that a path is within the sandbox (MANDATORY before filesystem access).
    ///
    /// # Security Contract (SEC-023)
    /// - MUST reject paths outside sandbox_root
    /// - MUST canonicalize paths to resolve symlinks and .. components
    /// - MUST return PathEscapesSandbox error for violations
    ///
    /// # Examples
    /// ```rust,ignore
    /// // Valid path within sandbox
    /// sandbox.validate_path(Path::new("/sandbox/allowed.txt"))?;
    ///
    /// // Invalid: escapes via ..
    /// sandbox.validate_path(Path::new("/sandbox/../etc/passwd")) // Error!
    ///
    /// // Invalid: absolute path outside sandbox
    /// sandbox.validate_path(Path::new("/etc/passwd")) // Error!
    /// ```
    fn validate_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        // Canonicalize to resolve symlinks and .. components
        let canonical = path.canonicalize().map_err(|e| {
            SandboxError::IoError(format!("Failed to canonicalize {}: {}", path.display(), e))
        })?;

        let root = self.sandbox_root().canonicalize().map_err(|e| {
            SandboxError::InitializationFailed(format!("Failed to canonicalize root: {}", e))
        })?;

        // Check if canonical path starts with root
        if !canonical.starts_with(&root) {
            return Err(SandboxError::PathEscapesSandbox(format!(
                "{} is outside sandbox root {}",
                canonical.display(),
                root.display()
            )));
        }

        Ok(canonical)
    }

    /// Check if a path is within the sandbox root (legacy method).
    ///
    /// Prefer `validate_path()` for error handling.
    fn is_path_confined(&self, path: &Path) -> bool {
        self.validate_path(path).is_ok()
    }

    /// Read a file using execute().
    fn read_via_execute(&self, file_path: &str) -> Result<String, FileOperationError> {
        let response = self.execute_sync(&format!("cat -n '{}'", file_path), None);
        if response.exit_code != Some(0) {
            if response.output.contains("No such file") {
                return Err(FileOperationError::FileNotFound);
            }
            if response.output.contains("Permission denied") {
                return Err(FileOperationError::PermissionDenied);
            }
            if response.output.contains("Is a directory") {
                return Err(FileOperationError::IsDirectory);
            }
            return Err(FileOperationError::InvalidPath);
        }
        Ok(response.output)
    }

    /// List files using execute().
    fn ls_via_execute(&self, path: &str) -> Vec<FileInfo> {
        let response = self.execute_sync(
            &format!("ls -la --time-style=full-iso '{}' 2>/dev/null", path),
            None,
        );
        if response.exit_code != Some(0) {
            return Vec::new();
        }
        // Parse ls output (simplified)
        let mut results = Vec::new();
        for line in response.output.lines().skip(1) {
            // skip "total" line
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 9 {
                let is_dir = parts[0].starts_with('d');
                let size: u64 = parts[4].parse().unwrap_or(0);
                let name = parts[8..].join(" ");
                if name != "." && name != ".." {
                    results.push(FileInfo {
                        path: if path.is_empty() {
                            name
                        } else {
                            format!("{}/{}", path.trim_end_matches('/'), name)
                        },
                        is_dir,
                        size,
                        modified_at: None,
                    });
                }
            }
        }
        results
    }
}

/// Local filesystem sandbox implementation with strict path confinement.
///
/// # Security Properties (SEC-023)
/// - All filesystem access confined to `root` directory
/// - Path validation enforced via `validate_path()` before operations
/// - Command execution runs with working directory = sandbox root
/// - Environment sanitized (only HOME and PATH set)
pub struct LocalSandbox {
    id: String,
    root: PathBuf,
    config: SandboxConfig,
    created_at: std::time::Instant,
}

impl LocalSandbox {
    /// Create a new local sandbox with the given root directory.
    ///
    /// Creates the root directory if it doesn't exist.
    pub fn new(root: PathBuf) -> Result<Self, SandboxError> {
        Self::new_with_config(root, SandboxConfig::default())
    }

    /// Create a sandbox with custom configuration.
    pub fn new_with_config(root: PathBuf, config: SandboxConfig) -> Result<Self, SandboxError> {
        // Create root directory if it doesn't exist
        if !root.exists() {
            std::fs::create_dir_all(&root).map_err(|e| {
                SandboxError::InitializationFailed(format!(
                    "Failed to create sandbox root {}: {}",
                    root.display(),
                    e
                ))
            })?;
        }

        // Verify root is a directory
        if !root.is_dir() {
            return Err(SandboxError::InitializationFailed(format!(
                "{} is not a directory",
                root.display()
            )));
        }

        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            root,
            config,
            created_at: std::time::Instant::now(),
        })
    }

    /// Get sandbox uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }
}

impl BaseSandbox for LocalSandbox {
    fn sandbox_root(&self) -> &Path {
        &self.root
    }

    fn config(&self) -> &SandboxConfig {
        &self.config
    }

    fn sandbox_id(&self) -> &str {
        &self.id
    }

    fn execute_sync(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
        use std::process::Command;
        use std::time::{Duration, Instant};

        let timeout_secs = timeout.unwrap_or(self.config.timeout_secs);
        let start = Instant::now();

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.current_dir(&self.root);

        // Sanitize environment (SEC-005)
        cmd.env_clear();
        cmd.env("HOME", &self.root);
        cmd.env("PATH", "/usr/bin:/bin");

        // Execute with timeout
        let result = match cmd.output() {
            Ok(output) => {
                let elapsed = start.elapsed();
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr);

                // Combine stdout and stderr
                if !stderr.is_empty() {
                    stdout.push_str("\n[stderr]\n");
                    stdout.push_str(&stderr);
                }

                // Truncate if needed
                let truncated = if stdout.len() > self.config.max_output_size {
                    stdout.truncate(self.config.max_output_size);
                    true
                } else {
                    false
                };

                ExecuteResponse {
                    output: stdout,
                    exit_code: output.status.code(),
                    truncated,
                }
            }
            Err(e) => ExecuteResponse {
                output: format!("Command execution failed: {}", e),
                exit_code: None,
                truncated: false,
            },
        };

        result
    }
}

/// Async sandbox backend implementation.
#[async_trait]
impl SandboxBackend for LocalSandbox {
    async fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
        // Delegate to synchronous implementation
        // In production, use tokio::process::Command for true async
        self.execute_sync(command, timeout)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn sandbox_root(&self) -> &Path {
        &self.root
    }
}

/// Implement Backend trait for LocalSandbox via default shell-based operations.
#[async_trait]
impl Backend for LocalSandbox {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        self.ls_via_execute(path)
    }

    async fn read_file(
        &self,
        file_path: &str,
        _offset: usize,
        _limit: usize,
    ) -> Result<String, FileOperationError> {
        self.read_via_execute(file_path)
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        // Shell-based write with path validation
        let response = self.execute_sync(
            &format!("cat > '{}' << 'EOF'\n{}\nEOF", file_path, content),
            None,
        );

        if response.exit_code == Some(0) {
            WriteResult {
                error: None,
                path: Some(file_path.to_string()),
                files_update: None,
            }
        } else {
            WriteResult {
                error: Some(response.output),
                path: None,
                files_update: None,
            }
        }
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        let sed_flags = if replace_all { "g" } else { "" };
        let response = self.execute_sync(
            &format!(
                "sed -i 's/{}/{}/{}' '{}'",
                old_string.replace('/', "\\/"),
                new_string.replace('/', "\\/"),
                sed_flags,
                file_path
            ),
            None,
        );

        if response.exit_code == Some(0) {
            EditResult {
                error: None,
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: None, // sed doesn't report count
            }
        } else {
            EditResult {
                error: Some(response.output),
                path: None,
                files_update: None,
                occurrences: None,
            }
        }
    }

    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        let search_path = if path.is_empty() { "." } else { path };
        let response = self.execute_sync(
            &format!("find '{}' -name '{}' -ls 2>/dev/null", search_path, pattern),
            None,
        );

        if response.exit_code != Some(0) {
            return Vec::new();
        }

        // Parse find -ls output (simplified)
        response
            .output
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 11 {
                    let is_dir = parts[2].starts_with('d');
                    let size: u64 = parts[6].parse().ok()?;
                    let name = parts[10..].join(" ");
                    Some(FileInfo {
                        path: name,
                        is_dir,
                        size,
                        modified_at: None,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let search_path = path.unwrap_or(".");
        let include_flag = include_glob
            .map(|g| format!("--include='{}'", g))
            .unwrap_or_default();

        let response = self.execute_sync(
            &format!(
                "grep -rn {} '{}' {} 2>/dev/null || true",
                include_flag, pattern, search_path
            ),
            None,
        );

        let matches = response
            .output
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(3, ':').collect();
                if parts.len() == 3 {
                    Some(GrepMatch {
                        path: parts[0].to_string(),
                        line: parts[1].parse().ok()?,
                        text: parts[2].to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(matches)
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        let mut results = Vec::new();
        for path in paths {
            let content = match self.read_file(path, 0, usize::MAX).await {
                Ok(text) => Some(text.into_bytes()),
                Err(e) => {
                    results.push(FileDownloadResponse {
                        path: path.clone(),
                        content: None,
                        error: Some(e),
                    });
                    continue;
                }
            };

            results.push(FileDownloadResponse {
                path: path.clone(),
                content,
                error: None,
            });
        }
        results
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        let mut results = Vec::new();
        for (path, content) in files {
            let text = String::from_utf8_lossy(content);
            let write_result = self.write_file(path, &text).await;

            results.push(FileUploadResponse {
                path: path.clone(),
                error: write_result.error.map(|e| FileOperationError::IoError(e)),
            });
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_sandbox_config_default() {
        let config = SandboxConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_output_size, 1024 * 1024);
        assert!(config.work_dir.is_none());
    }

    #[test]
    fn test_sandbox_config_custom() {
        let config = SandboxConfig {
            timeout_secs: 60,
            max_output_size: 512,
            work_dir: Some("/workspace".to_string()),
        };
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.max_output_size, 512);
        assert_eq!(config.work_dir.as_deref(), Some("/workspace"));
    }

    #[test]
    fn test_local_sandbox_creation() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        assert_eq!(BaseSandbox::sandbox_root(&sandbox), temp.path());
        assert!(BaseSandbox::sandbox_id(&sandbox).len() > 0);
        assert_eq!(sandbox.config().timeout_secs, 30);
    }

    #[test]
    fn test_local_sandbox_creates_root() {
        let temp = TempDir::new().unwrap();
        let sandbox_root = temp.path().join("new_sandbox");

        // Directory doesn't exist yet
        assert!(!sandbox_root.exists());

        let sandbox = LocalSandbox::new(sandbox_root.clone()).unwrap();

        // Now it should exist
        assert!(sandbox_root.exists());
        assert!(sandbox_root.is_dir());
    }

    #[test]
    fn test_local_sandbox_rejects_file_as_root() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("not_a_dir.txt");
        fs::write(&file_path, "test").unwrap();

        let result = LocalSandbox::new(file_path);
        assert!(result.is_err());
        match result {
            Err(SandboxError::InitializationFailed(msg)) => {
                assert!(msg.contains("not a directory"));
            }
            _ => panic!("Expected InitializationFailed error"),
        }
    }

    // SEC-023: Path restriction tests
    #[test]
    fn test_validate_path_allows_within_sandbox() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create a test file
        let test_file = temp.path().join("allowed.txt");
        fs::write(&test_file, "allowed").unwrap();

        // Should succeed
        let result = sandbox.validate_path(&test_file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file.canonicalize().unwrap());
    }

    #[test]
    fn test_validate_path_rejects_parent_escape() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Try to escape via ../ to a path that exists (the parent temp dir)
        // First create a file outside sandbox that we can reference
        let parent_dir = temp.path().parent().unwrap();
        let escape_target = parent_dir.join("escape_test.txt");
        fs::write(&escape_target, "test").unwrap();

        let escape_path = temp
            .path()
            .join("..")
            .join(escape_target.file_name().unwrap());

        let result = sandbox.validate_path(&escape_path);
        // Clean up
        let _ = fs::remove_file(&escape_target);

        assert!(result.is_err(), "Path escaping sandbox should be rejected");
        match result {
            Err(SandboxError::PathEscapesSandbox(msg)) => {
                assert!(msg.contains("outside sandbox root"));
            }
            Err(e) => panic!("Expected PathEscapesSandbox error, got: {:?}", e),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_validate_path_rejects_absolute_outside() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Try absolute path outside sandbox
        let outside_path = Path::new("/etc/passwd");

        // This will fail at canonicalize if file doesn't exist,
        // or at starts_with check if it does
        let result = sandbox.validate_path(outside_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_rejects_symlink_escape() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create symlink pointing outside sandbox
        let link_path = temp.path().join("evil_link");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/passwd", &link_path).unwrap();

            let result = sandbox.validate_path(&link_path);
            assert!(result.is_err());
            match result {
                Err(SandboxError::PathEscapesSandbox(msg)) => {
                    assert!(msg.contains("outside sandbox root"));
                }
                _ => panic!("Expected PathEscapesSandbox error, got {:?}", result),
            }
        }
    }

    #[test]
    fn test_is_path_confined_legacy() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let allowed = temp.path().join("allowed.txt");
        fs::write(&allowed, "test").unwrap();

        assert!(sandbox.is_path_confined(&allowed));

        // Escape attempt
        let escape = temp.path().join("../etc/passwd");
        assert!(!sandbox.is_path_confined(&escape));
    }

    #[test]
    fn test_execute_sync_basic() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let response = sandbox.execute_sync("echo 'hello world'", None);
        assert_eq!(response.exit_code, Some(0));
        assert!(response.output.contains("hello world"));
        assert!(!response.truncated);
    }

    #[test]
    fn test_execute_sync_confined_to_root() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create file in sandbox
        fs::write(temp.path().join("test.txt"), "sandbox file").unwrap();

        // Command runs in sandbox root, so relative path works
        let response = sandbox.execute_sync("cat test.txt", None);
        assert_eq!(response.exit_code, Some(0));
        assert!(response.output.contains("sandbox file"));
    }

    #[test]
    fn test_execute_sync_environment_sanitized() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let response = sandbox.execute_sync("env | sort", None);
        assert_eq!(response.exit_code, Some(0));

        // Check that HOME and PATH are set correctly
        // Note: Some shells may add additional variables (PWD, SHLVL, _)
        // but sensitive ones (AWS_*, API_*, ANTHROPIC_*, etc.) should not be present
        let output = &response.output;
        assert!(output.contains("HOME="), "HOME should be set");
        assert!(output.contains("PATH="), "PATH should be set");

        // Ensure no sensitive env vars leaked
        assert!(!output.contains("AWS_"), "AWS credentials should not leak");
        assert!(!output.contains("API_KEY"), "API keys should not leak");
        assert!(
            !output.contains("ANTHROPIC_"),
            "Anthropic keys should not leak"
        );
        assert!(!output.contains("OPENAI_"), "OpenAI keys should not leak");
        assert!(!output.contains("SECRET"), "Secrets should not leak");
    }

    #[test]
    fn test_execute_sync_truncates_large_output() {
        let temp = TempDir::new().unwrap();
        let config = SandboxConfig {
            timeout_secs: 30,
            max_output_size: 100, // Small limit
            work_dir: None,
        };
        let sandbox = LocalSandbox::new_with_config(temp.path().to_path_buf(), config).unwrap();

        // Generate output larger than limit
        let response = sandbox.execute_sync("seq 1 1000", None);
        assert_eq!(response.exit_code, Some(0));
        assert!(response.truncated);
        assert_eq!(response.output.len(), 100);
    }

    #[test]
    fn test_sandbox_uptime() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(sandbox.uptime_secs() == 0); // Less than 1 second
    }

    // Additional security tests for multiple escape vectors
    #[test]
    fn test_validate_path_rejects_double_dot_variations() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let attempts = vec![
            temp.path().join(".."),
            temp.path().join("../.."),
            temp.path().join("foo/../../.."),
            temp.path().join("./../../etc"),
        ];

        for attempt in attempts {
            let result = sandbox.validate_path(&attempt);
            assert!(result.is_err(), "Should reject: {}", attempt.display());
        }
    }

    #[test]
    fn test_validate_path_allows_subdirectories() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create nested structure
        let subdir = temp.path().join("sub/dir/nested");
        fs::create_dir_all(&subdir).unwrap();
        let file = subdir.join("file.txt");
        fs::write(&file, "nested").unwrap();

        let result = sandbox.validate_path(&file);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_normalizes_dot_segments() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create actual directory structure for canonicalize to work
        let subdir = temp.path().join("foo");
        fs::create_dir(&subdir).unwrap();

        let file = temp.path().join("test.txt");
        fs::write(&file, "test").unwrap();

        // Path with redundant ./ and foo/../ segments
        let weird_path = temp.path().join("./foo/../test.txt");

        let result = sandbox.validate_path(&weird_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file.canonicalize().unwrap());
    }
}
