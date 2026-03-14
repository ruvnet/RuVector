//! Sandbox trait and configuration (ADR-103 C5).
//!
//! Defines the `BaseSandbox` trait for backends that provide
//! filesystem confinement, and `SandboxConfig` for configuration.

use crate::protocol::*;
use std::path::Path;

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

/// Base sandbox trait for backends providing filesystem confinement.
///
/// Concrete implementations MUST confine filesystem access to the
/// sandbox root path (SEC-023). The `sandbox_root()` method defines
/// the confinement boundary.
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

    /// Check if a path is within the sandbox root.
    fn is_path_confined(&self, path: &Path) -> bool {
        // Normalize the path and check it stays within sandbox_root
        let root = self.sandbox_root();
        match path.canonicalize() {
            Ok(canonical) => canonical.starts_with(root),
            Err(_) => {
                // If we can't canonicalize, check lexically
                let normalized = crate::filesystem::FilesystemBackend::new(root.to_path_buf())
                    .resolve_path(&path.to_string_lossy());
                normalized.is_ok()
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
