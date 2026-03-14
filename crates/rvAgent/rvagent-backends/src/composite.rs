//! CompositeBackend — path-prefix routing to sub-backends (ADR-103 C11).
//!
//! Routes file operations to different backends based on path prefixes.
//! After prefix stripping, re-validates the resulting path against
//! traversal attacks (SEC-003).

use crate::protocol::*;
use async_trait::async_trait;
use std::sync::Arc;

/// A reference to a backend, shared across routes.
pub type BackendRef = Arc<dyn Backend>;

/// Composite backend that routes operations to sub-backends based on path prefix.
///
/// Routes are sorted by prefix length (longest first) to ensure the most
/// specific match is used. A default backend handles unmatched paths.
pub struct CompositeBackend {
    default: BackendRef,
    routes: Vec<(String, BackendRef)>,
}

impl CompositeBackend {
    /// Create a new composite backend with a default backend and a set of routes.
    ///
    /// Routes are automatically sorted by prefix length (longest first).
    pub fn new(default: BackendRef, mut routes: Vec<(String, BackendRef)>) -> Self {
        // Sort by prefix length descending for longest-prefix-first matching
        routes.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Self { default, routes }
    }

    /// Add a route. Re-sorts routes after insertion.
    pub fn add_route(&mut self, prefix: String, backend: BackendRef) {
        self.routes.push((prefix, backend));
        self.routes.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }

    /// Route a path to the appropriate backend, stripping the matched prefix.
    ///
    /// After prefix stripping, re-validates the resulting path against
    /// traversal attacks (ADR-103 C11/SEC-003).
    fn route_path(&self, path: &str) -> Result<(BackendRef, String), FileOperationError> {
        for (prefix, backend) in &self.routes {
            if path.starts_with(prefix.as_str()) {
                let stripped = &path[prefix.len()..];
                // Strip leading '/' from the remainder
                let stripped = stripped.strip_prefix('/').unwrap_or(stripped);

                // Re-validate against traversal after stripping (ADR-103 C11)
                if stripped.contains("..") || stripped.starts_with('~') {
                    return Err(FileOperationError::SecurityViolation(
                        "path traversal detected after prefix stripping".to_string(),
                    ));
                }

                return Ok((backend.clone(), stripped.to_string()));
            }
        }

        // Default backend — no stripping, but still validate
        if path.contains("..") && crate::utils::contains_traversal(path) {
            return Err(FileOperationError::SecurityViolation(
                "path traversal detected".to_string(),
            ));
        }

        Ok((self.default.clone(), path.to_string()))
    }

    /// Re-map a path from the sub-backend's relative path back to the
    /// composite's full path.
    fn remap_path(prefix: &str, relative_path: &str) -> String {
        if prefix.is_empty() {
            relative_path.to_string()
        } else {
            format!("{}/{}", prefix.trim_end_matches('/'), relative_path)
        }
    }

    /// Find the prefix used for a given path.
    fn find_prefix(&self, path: &str) -> String {
        for (prefix, _) in &self.routes {
            if path.starts_with(prefix.as_str()) {
                return prefix.clone();
            }
        }
        String::new()
    }
}

#[async_trait]
impl Backend for CompositeBackend {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        let prefix = self.find_prefix(path);
        match self.route_path(path) {
            Ok((backend, stripped)) => {
                let mut results = backend.ls_info(&stripped).await;
                // Remap paths back to composite namespace
                for info in &mut results {
                    info.path = Self::remap_path(&prefix, &info.path);
                }
                results
            }
            Err(_) => Vec::new(),
        }
    }

    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        let (backend, stripped) = self.route_path(file_path)?;
        backend.read_file(&stripped, offset, limit).await
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        match self.route_path(file_path) {
            Ok((backend, stripped)) => {
                let mut result = backend.write_file(&stripped, content).await;
                if let Some(ref mut p) = result.path {
                    let prefix = self.find_prefix(file_path);
                    *p = Self::remap_path(&prefix, p);
                }
                result
            }
            Err(e) => WriteResult {
                error: Some(e.to_string()),
                path: None,
                files_update: None,
            },
        }
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        match self.route_path(file_path) {
            Ok((backend, stripped)) => {
                let mut result = backend
                    .edit_file(&stripped, old_string, new_string, replace_all)
                    .await;
                if let Some(ref mut p) = result.path {
                    let prefix = self.find_prefix(file_path);
                    *p = Self::remap_path(&prefix, p);
                }
                result
            }
            Err(e) => EditResult {
                error: Some(e.to_string()),
                path: None,
                files_update: None,
                occurrences: None,
            },
        }
    }

    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        let prefix = self.find_prefix(path);
        match self.route_path(path) {
            Ok((backend, stripped)) => {
                let mut results = backend.glob_info(pattern, &stripped).await;
                for info in &mut results {
                    info.path = Self::remap_path(&prefix, &info.path);
                }
                results
            }
            Err(_) => Vec::new(),
        }
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        let search_path = path.unwrap_or("");
        let prefix = self.find_prefix(search_path);
        let (backend, stripped) = self
            .route_path(search_path)
            .map_err(|e| e.to_string())?;
        let stripped_opt = if stripped.is_empty() {
            None
        } else {
            Some(stripped.as_str())
        };
        let mut results = backend.grep(pattern, stripped_opt, include_glob).await?;
        for m in &mut results {
            m.path = Self::remap_path(&prefix, &m.path);
        }
        Ok(results)
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        let mut responses = Vec::with_capacity(paths.len());
        for path in paths {
            match self.route_path(path) {
                Ok((backend, stripped)) => {
                    let mut result = backend.download_files(&[stripped]).await;
                    if let Some(resp) = result.pop() {
                        responses.push(FileDownloadResponse {
                            path: path.clone(),
                            ..resp
                        });
                    }
                }
                Err(e) => {
                    responses.push(FileDownloadResponse {
                        path: path.clone(),
                        content: None,
                        error: Some(e),
                    });
                }
            }
        }
        responses
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        let mut responses = Vec::with_capacity(files.len());
        for (path, content) in files {
            match self.route_path(path) {
                Ok((backend, stripped)) => {
                    let mut result = backend
                        .upload_files(&[(stripped, content.clone())])
                        .await;
                    if let Some(resp) = result.pop() {
                        responses.push(FileUploadResponse {
                            path: path.clone(),
                            ..resp
                        });
                    }
                }
                Err(e) => {
                    responses.push(FileUploadResponse {
                        path: path.clone(),
                        error: Some(e),
                    });
                }
            }
        }
        responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateBackend;

    fn make_composite() -> CompositeBackend {
        let default: BackendRef = Arc::new(StateBackend::new());
        let workspace: BackendRef = Arc::new(StateBackend::new());
        let routes = vec![("workspace/".to_string(), workspace)];
        CompositeBackend::new(default, routes)
    }

    #[test]
    fn test_route_path_default() {
        let composite = make_composite();
        let (_, stripped) = composite.route_path("file.txt").unwrap();
        assert_eq!(stripped, "file.txt");
    }

    #[test]
    fn test_route_path_prefix_match() {
        let composite = make_composite();
        let (_, stripped) = composite.route_path("workspace/src/main.rs").unwrap();
        assert_eq!(stripped, "src/main.rs");
    }

    #[test]
    fn test_route_path_traversal_after_strip() {
        let composite = make_composite();
        let result = composite.route_path("workspace/../../../etc/passwd");
        assert!(result.is_err());
        match result.unwrap_err() {
            FileOperationError::SecurityViolation(msg) => {
                assert!(msg.contains("traversal"));
            }
            other => panic!("Expected SecurityViolation, got {:?}", other),
        }
    }

    #[test]
    fn test_route_path_tilde_after_strip() {
        let composite = make_composite();
        let result = composite.route_path("workspace/~/.ssh/id_rsa");
        assert!(result.is_err());
    }

    #[test]
    fn test_remap_path() {
        assert_eq!(
            CompositeBackend::remap_path("workspace", "src/main.rs"),
            "workspace/src/main.rs"
        );
        assert_eq!(
            CompositeBackend::remap_path("", "file.txt"),
            "file.txt"
        );
    }

    #[test]
    fn test_longest_prefix_first() {
        let default: BackendRef = Arc::new(StateBackend::new());
        let short: BackendRef = Arc::new(StateBackend::new());
        let long: BackendRef = Arc::new(StateBackend::new());
        let routes = vec![
            ("a/".to_string(), short),
            ("a/b/c/".to_string(), long),
        ];
        let composite = CompositeBackend::new(default, routes);
        // Should match the longer prefix
        assert_eq!(composite.routes[0].0, "a/b/c/");
        assert_eq!(composite.routes[1].0, "a/");
    }

    #[tokio::test]
    async fn test_composite_write_read() {
        let default: BackendRef = Arc::new(StateBackend::new());
        let workspace: BackendRef = Arc::new(StateBackend::new());
        let routes = vec![("ws/".to_string(), workspace.clone())];
        let composite = CompositeBackend::new(default, routes);

        // Write to workspace backend via composite
        composite.write_file("ws/test.txt", "hello").await;

        // Read via composite
        let content = composite.read_file("ws/test.txt", 0, 0).await.unwrap();
        assert!(content.contains("hello"));

        // Should also be readable directly from the workspace backend
        let direct = workspace.read_file("test.txt", 0, 0).await.unwrap();
        assert!(direct.contains("hello"));
    }

    #[tokio::test]
    async fn test_composite_traversal_blocked() {
        let composite = make_composite();
        let result = composite.read_file("workspace/../../etc/shadow", 0, 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_composite_grep_remaps_paths() {
        let default: BackendRef = Arc::new(StateBackend::new());
        let ws: BackendRef = Arc::new(StateBackend::new());

        // Write a file to the workspace backend
        ws.write_file("code.rs", "fn main() {}").await;

        let routes = vec![("ws/".to_string(), ws)];
        let composite = CompositeBackend::new(default, routes);

        let results = composite.grep("fn main", Some("ws/"), None).await.unwrap();
        assert!(!results.is_empty());
        // Path should be remapped to include the prefix
        assert!(results[0].path.starts_with("ws/"));
    }
}
