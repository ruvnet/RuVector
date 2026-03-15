//! StateBackend — ephemeral in-memory backend storing files in agent state.
//!
//! Maps to Python's `StateBackend`. Uses `Arc<RwLock<HashMap<String, FileData>>>`
//! for thread-safe concurrent access to the file store.

use crate::protocol::*;
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Ephemeral in-memory file store backend.
///
/// Stores files as `FileData` structs in a shared `HashMap`. Suitable for
/// WASM targets and testing where no filesystem access is available.
#[derive(Clone)]
pub struct StateBackend {
    files: Arc<RwLock<HashMap<String, FileData>>>,
}

impl StateBackend {
    /// Create a new empty state backend.
    pub fn new() -> Self {
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a state backend pre-populated with the given files.
    pub fn with_files(files: HashMap<String, FileData>) -> Self {
        Self {
            files: Arc::new(RwLock::new(files)),
        }
    }

    /// Get a snapshot of all stored file paths.
    pub fn file_paths(&self) -> Vec<String> {
        self.files.read().keys().cloned().collect()
    }

    /// Check if a file exists in the store.
    pub fn contains(&self, path: &str) -> bool {
        self.files.read().contains_key(path)
    }
}

impl Default for StateBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for StateBackend {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        let files = self.files.read();
        let prefix = if path.ends_with('/') || path.is_empty() {
            path.to_string()
        } else {
            format!("{}/", path)
        };

        let mut results = Vec::new();
        let mut seen_dirs = std::collections::HashSet::new();

        for (file_path, data) in files.iter() {
            if path.is_empty() || file_path.starts_with(&prefix) || file_path == path {
                // Direct file match
                if file_path == path {
                    let content_size: usize = data.content.iter().map(|l| l.len() + 1).sum();
                    results.push(FileInfo {
                        path: file_path.clone(),
                        is_dir: false,
                        size: content_size as u64,
                        modified_at: Some(data.modified_at.clone()),
                    });
                } else if file_path.starts_with(&prefix) {
                    // Check if there's a subdirectory
                    let rest = &file_path[prefix.len()..];
                    if let Some(slash_pos) = rest.find('/') {
                        let dir_name = &rest[..slash_pos];
                        let dir_path = format!("{}{}", prefix, dir_name);
                        if seen_dirs.insert(dir_path.clone()) {
                            results.push(FileInfo {
                                path: dir_path,
                                is_dir: true,
                                size: 0,
                                modified_at: None,
                            });
                        }
                    } else {
                        let content_size: usize =
                            data.content.iter().map(|l| l.len() + 1).sum();
                        results.push(FileInfo {
                            path: file_path.clone(),
                            is_dir: false,
                            size: content_size as u64,
                            modified_at: Some(data.modified_at.clone()),
                        });
                    }
                }
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        results
    }

    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        let files = self.files.read();
        let data = files
            .get(file_path)
            .ok_or(FileOperationError::FileNotFound)?;

        let lines: Vec<&str> = data.content.iter().map(|s| s.as_str()).collect();
        let total = lines.len();
        let start = offset.min(total);
        let end = if limit == 0 {
            total
        } else {
            (start + limit).min(total)
        };

        let selected: Vec<&str> = lines[start..end].to_vec();
        let content = selected.join("\n");

        use crate::utils::format_content_with_line_numbers;
        Ok(format_content_with_line_numbers(&content, start + 1, 2000))
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        let now = Utc::now().to_rfc3339();
        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();

        let mut files = self.files.write();
        let existed = files.contains_key(file_path);
        let created_at = if existed {
            files.get(file_path).map(|f| f.created_at.clone()).unwrap_or_else(|| now.clone())
        } else {
            now.clone()
        };

        files.insert(
            file_path.to_string(),
            FileData {
                content: lines,
                created_at,
                modified_at: now,
            },
        );

        WriteResult {
            error: None,
            path: Some(file_path.to_string()),
            files_update: None,
        }
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        let mut files = self.files.write();
        let data = match files.get_mut(file_path) {
            Some(d) => d,
            None => {
                return EditResult {
                    error: Some(format!("File not found: {}", file_path)),
                    path: None,
                    files_update: None,
                    occurrences: None,
                };
            }
        };

        let full_content = data.content.join("\n");
        let count = full_content.matches(old_string).count() as u32;

        if count == 0 {
            return EditResult {
                error: Some(format!(
                    "old_string not found in {}",
                    file_path
                )),
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(0),
            };
        }

        if !replace_all && count > 1 {
            return EditResult {
                error: Some(format!(
                    "old_string found {} times in {} — must be unique (or use replace_all)",
                    count, file_path
                )),
                path: Some(file_path.to_string()),
                files_update: None,
                occurrences: Some(count),
            };
        }

        let new_content = if replace_all {
            full_content.replace(old_string, new_string)
        } else {
            full_content.replacen(old_string, new_string, 1)
        };

        let replaced_count = if replace_all { count } else { 1 };
        data.content = new_content.lines().map(|l| l.to_string()).collect();
        data.modified_at = Utc::now().to_rfc3339();

        EditResult {
            error: None,
            path: Some(file_path.to_string()),
            files_update: None,
            occurrences: Some(replaced_count),
        }
    }

    async fn glob_info(&self, pattern: &str, _path: &str) -> Vec<FileInfo> {
        let files = self.files.read();
        let glob_pattern = match glob::Pattern::new(pattern) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();
        for (file_path, data) in files.iter() {
            if glob_pattern.matches(file_path) {
                let content_size: usize = data.content.iter().map(|l| l.len() + 1).sum();
                results.push(FileInfo {
                    path: file_path.clone(),
                    is_dir: false,
                    size: content_size as u64,
                    modified_at: Some(data.modified_at.clone()),
                });
            }
        }
        results.sort_by(|a, b| a.path.cmp(&b.path));
        results
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        _include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let files = self.files.read();
        let mut matches = Vec::new();

        for (file_path, data) in files.iter() {
            if let Some(search_path) = path {
                if !file_path.starts_with(search_path) {
                    continue;
                }
            }

            for (line_idx, line) in data.content.iter().enumerate() {
                // Literal string matching (ADR-103 C13)
                if line.contains(pattern) {
                    matches.push(GrepMatch {
                        path: file_path.clone(),
                        line: (line_idx + 1) as u32,
                        text: line.clone(),
                    });
                }
            }
        }

        Ok(matches)
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        let files = self.files.read();
        paths
            .iter()
            .map(|p| {
                if let Some(data) = files.get(p) {
                    FileDownloadResponse {
                        path: p.clone(),
                        content: Some(data.content.join("\n").into_bytes()),
                        error: None,
                    }
                } else {
                    FileDownloadResponse {
                        path: p.clone(),
                        content: None,
                        error: Some(FileOperationError::FileNotFound),
                    }
                }
            })
            .collect()
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        let now = Utc::now().to_rfc3339();
        let mut store = self.files.write();
        files
            .iter()
            .map(|(path, content)| {
                let text = String::from_utf8_lossy(content);
                let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
                store.insert(
                    path.clone(),
                    FileData {
                        content: lines,
                        created_at: now.clone(),
                        modified_at: now.clone(),
                    },
                );
                FileUploadResponse {
                    path: path.clone(),
                    error: None,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_and_read() {
        let backend = StateBackend::new();
        backend.write_file("test.txt", "hello\nworld").await;
        let content = backend.read_file("test.txt", 0, 0).await.unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("world"));
    }

    #[tokio::test]
    async fn test_read_not_found() {
        let backend = StateBackend::new();
        let result = backend.read_file("missing.txt", 0, 0).await;
        assert_eq!(result.unwrap_err(), FileOperationError::FileNotFound);
    }

    #[tokio::test]
    async fn test_edit_replace_one() {
        let backend = StateBackend::new();
        backend.write_file("test.txt", "hello world").await;
        let result = backend
            .edit_file("test.txt", "hello", "goodbye", false)
            .await;
        assert!(result.error.is_none());
        assert_eq!(result.occurrences, Some(1));

        let content = backend.read_file("test.txt", 0, 0).await.unwrap();
        assert!(content.contains("goodbye"));
    }

    #[tokio::test]
    async fn test_edit_not_unique() {
        let backend = StateBackend::new();
        backend.write_file("test.txt", "aaa bbb aaa").await;
        let result = backend
            .edit_file("test.txt", "aaa", "ccc", false)
            .await;
        assert!(result.error.is_some());
        assert_eq!(result.occurrences, Some(2));
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let backend = StateBackend::new();
        backend.write_file("test.txt", "aaa bbb aaa").await;
        let result = backend
            .edit_file("test.txt", "aaa", "ccc", true)
            .await;
        assert!(result.error.is_none());
        assert_eq!(result.occurrences, Some(2));
    }

    #[tokio::test]
    async fn test_ls_info() {
        let backend = StateBackend::new();
        backend.write_file("src/main.rs", "fn main() {}").await;
        backend.write_file("src/lib.rs", "pub mod foo;").await;
        backend.write_file("README.md", "# Hello").await;

        let items = backend.ls_info("src").await;
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn test_glob_info() {
        let backend = StateBackend::new();
        backend.write_file("src/main.rs", "fn main()").await;
        backend.write_file("src/lib.rs", "pub mod").await;
        backend.write_file("Cargo.toml", "[package]").await;

        let results = backend.glob_info("src/*.rs", "").await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_grep_literal() {
        let backend = StateBackend::new();
        backend
            .write_file("test.rs", "fn main() {}\nfn helper() {}")
            .await;
        let results = backend.grep("fn ", None, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_upload_download() {
        let backend = StateBackend::new();
        let upload_result = backend
            .upload_files(&[("doc.txt".to_string(), b"content here".to_vec())])
            .await;
        assert!(upload_result[0].error.is_none());

        let download_result = backend
            .download_files(&["doc.txt".to_string()])
            .await;
        assert!(download_result[0].error.is_none());
        assert!(download_result[0].content.is_some());
    }

    #[tokio::test]
    async fn test_contains_and_file_paths() {
        let backend = StateBackend::new();
        backend.write_file("a.txt", "data").await;
        assert!(backend.contains("a.txt"));
        assert!(!backend.contains("b.txt"));
        assert_eq!(backend.file_paths(), vec!["a.txt".to_string()]);
    }
}
