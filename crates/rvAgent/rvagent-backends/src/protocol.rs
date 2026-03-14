//! Core backend traits and types (ADR-094).
//!
//! Defines the `Backend` and `SandboxBackend` async traits that all
//! backend implementations must satisfy, plus the associated error
//! and response types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Standardized error codes for file operations (LLM-actionable).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, thiserror::Error)]
pub enum FileOperationError {
    #[error("file not found")]
    FileNotFound,
    #[error("permission denied")]
    PermissionDenied,
    #[error("is a directory")]
    IsDirectory,
    #[error("invalid path")]
    InvalidPath,
    #[error("security violation: {0}")]
    SecurityViolation(String),
}

/// Metadata about a file or directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    #[serde(default)]
    pub is_dir: bool,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub modified_at: Option<String>,
}

/// Response from downloading a file.
#[derive(Debug, Clone)]
pub struct FileDownloadResponse {
    pub path: String,
    pub content: Option<Vec<u8>>,
    pub error: Option<FileOperationError>,
}

/// Response from uploading a file.
#[derive(Debug, Clone)]
pub struct FileUploadResponse {
    pub path: String,
    pub error: Option<FileOperationError>,
}

/// A single grep match result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepMatch {
    pub path: String,
    pub line: u32,
    pub text: String,
}

/// Result of a write operation.
#[derive(Debug, Clone)]
pub struct WriteResult {
    pub error: Option<String>,
    pub path: Option<String>,
    pub files_update: Option<HashMap<String, serde_json::Value>>,
}

/// Result of an edit operation.
#[derive(Debug, Clone)]
pub struct EditResult {
    pub error: Option<String>,
    pub path: Option<String>,
    pub files_update: Option<HashMap<String, serde_json::Value>>,
    pub occurrences: Option<u32>,
}

/// Response from executing a command.
#[derive(Debug, Clone)]
pub struct ExecuteResponse {
    pub output: String,
    pub exit_code: Option<i32>,
    pub truncated: bool,
}

/// In-memory file data representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    pub content: Vec<String>,
    pub created_at: String,
    pub modified_at: String,
}

/// Core backend trait — all file operations.
///
/// Maps to Python's `BackendProtocol`. Provides both synchronous and
/// asynchronous variants of each method.
#[async_trait]
pub trait Backend: Send + Sync {
    /// List files/directories at the given path.
    async fn ls_info(&self, path: &str) -> Vec<FileInfo>;

    /// Read file content with optional offset and line limit.
    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError>;

    /// Write content to a file, creating it if necessary.
    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult;

    /// Edit a file by replacing occurrences of old_string with new_string.
    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult;

    /// Search for files matching a glob pattern.
    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo>;

    /// Search file contents for a pattern.
    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String>;

    /// Download files, returning their content.
    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse>;

    /// Upload files with the given content.
    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse>;
}

/// Extension trait for backends with shell execution capability.
///
/// Maps to Python's `SandboxBackendProtocol` (ADR-103 C5).
#[async_trait]
pub trait SandboxBackend: Backend {
    /// Execute a shell command within the sandbox.
    async fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;

    /// Unique identifier for this sandbox instance.
    fn id(&self) -> &str;

    /// Root path of the sandbox filesystem. Implementations MUST confine
    /// filesystem access to this root (ADR-103 C5/SEC-023).
    fn sandbox_root(&self) -> &Path;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_operation_error_display() {
        assert_eq!(FileOperationError::FileNotFound.to_string(), "file not found");
        assert_eq!(FileOperationError::PermissionDenied.to_string(), "permission denied");
        assert_eq!(FileOperationError::IsDirectory.to_string(), "is a directory");
        assert_eq!(FileOperationError::InvalidPath.to_string(), "invalid path");
        assert_eq!(
            FileOperationError::SecurityViolation("bad".into()).to_string(),
            "security violation: bad"
        );
    }

    #[test]
    fn test_file_info_serde() {
        let info = FileInfo {
            path: "/tmp/test.txt".to_string(),
            is_dir: false,
            size: 42,
            modified_at: Some("2026-01-01T00:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: FileInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.path, "/tmp/test.txt");
        assert_eq!(back.size, 42);
    }

    #[test]
    fn test_file_info_defaults() {
        let json = r#"{"path": "/foo"}"#;
        let info: FileInfo = serde_json::from_str(json).unwrap();
        assert!(!info.is_dir);
        assert_eq!(info.size, 0);
        assert!(info.modified_at.is_none());
    }

    #[test]
    fn test_grep_match_serde() {
        let m = GrepMatch {
            path: "src/main.rs".to_string(),
            line: 10,
            text: "fn main()".to_string(),
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("fn main()"));
    }

    #[test]
    fn test_write_result() {
        let r = WriteResult {
            error: None,
            path: Some("/tmp/out.txt".to_string()),
            files_update: None,
        };
        assert!(r.error.is_none());
        assert_eq!(r.path.as_deref(), Some("/tmp/out.txt"));
    }

    #[test]
    fn test_edit_result() {
        let r = EditResult {
            error: None,
            path: Some("/tmp/out.txt".to_string()),
            files_update: None,
            occurrences: Some(3),
        };
        assert_eq!(r.occurrences, Some(3));
    }

    #[test]
    fn test_execute_response() {
        let r = ExecuteResponse {
            output: "hello".to_string(),
            exit_code: Some(0),
            truncated: false,
        };
        assert_eq!(r.exit_code, Some(0));
        assert!(!r.truncated);
    }

    #[test]
    fn test_file_data() {
        let fd = FileData {
            content: vec!["line 1".to_string(), "line 2".to_string()],
            created_at: "2026-01-01".to_string(),
            modified_at: "2026-01-02".to_string(),
        };
        assert_eq!(fd.content.len(), 2);
    }

    #[test]
    fn test_file_operation_error_equality() {
        assert_eq!(FileOperationError::FileNotFound, FileOperationError::FileNotFound);
        assert_ne!(FileOperationError::FileNotFound, FileOperationError::InvalidPath);
    }
}
