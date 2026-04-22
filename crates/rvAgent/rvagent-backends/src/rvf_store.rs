//! RVF Store Backend — ADR-106 Layer 3 runtime bridge.
//!
//! Provides an `RvfStoreBackend` that wraps agent operations with RVF package
//! awareness. It routes `rvf://` paths to the mount table and delegates all
//! other operations to an inner backend.

use async_trait::async_trait;
use std::sync::{Arc, Mutex};

use rvagent_core::rvf_bridge::{
    MountTable, RvfBridgeConfig, RvfManifest, RvfMountHandle, RvfVerifyStatus,
};

use crate::protocol::{
    Backend, EditResult, ExecuteResponse, FileDownloadResponse, FileInfo, FileOperationError,
    FileUploadResponse, GrepMatch, SandboxBackend, WriteResult,
};

// ---------------------------------------------------------------------------
// RVF Store Backend
// ---------------------------------------------------------------------------

/// A backend that wraps RVF package operations.
///
/// This is the rvAgent-side adapter from ADR-106 Layer 3. It translates
/// agent backend operations into RVF package operations:
///
/// - `read_file()` can read files from mounted RVF packages
/// - `ls_info()` lists mounted packages and their contents
///
/// The mount table is shared across the agent for consistent package state.
pub struct RvfStoreBackend<B: Backend> {
    /// Shared mount table.
    mount_table: Arc<Mutex<MountTable>>,
    /// Inner backend for non-RVF operations.
    inner: B,
    /// Bridge configuration.
    _config: RvfBridgeConfig,
}

impl<B: Backend> RvfStoreBackend<B> {
    /// Create a new RVF store backend wrapping an inner backend.
    pub fn new(inner: B, config: RvfBridgeConfig) -> Self {
        Self {
            mount_table: Arc::new(Mutex::new(MountTable::new())),
            inner,
            _config: config,
        }
    }

    /// Create with an existing mount table (for sharing across components).
    pub fn with_mount_table(
        inner: B,
        config: RvfBridgeConfig,
        mount_table: Arc<Mutex<MountTable>>,
    ) -> Self {
        Self {
            mount_table,
            inner,
            _config: config,
        }
    }

    /// Get a reference to the mount table.
    pub fn mount_table(&self) -> &Arc<Mutex<MountTable>> {
        &self.mount_table
    }

    /// Mount an RVF package from a manifest.
    pub fn mount_package(
        &self,
        manifest: RvfManifest,
        verify_status: RvfVerifyStatus,
    ) -> RvfMountHandle {
        let mut table = self.mount_table.lock().unwrap();
        table.mount(manifest, verify_status)
    }

    /// Unmount a package by handle.
    pub fn unmount_package(&self, handle: RvfMountHandle) -> bool {
        let mut table = self.mount_table.lock().unwrap();
        table.unmount(handle)
    }

    /// List all tools from mounted packages.
    pub fn mounted_tools(&self) -> Vec<MountedToolInfo> {
        let table = self.mount_table.lock().unwrap();
        table
            .all_tools()
            .into_iter()
            .map(|(handle, entry)| MountedToolInfo {
                mount_handle: *handle,
                name: entry.name.clone(),
                description: entry.description.clone(),
                parameters_schema: entry.parameters_schema.clone(),
            })
            .collect()
    }

    /// Check if a path refers to an RVF-mounted resource.
    fn is_rvf_path(path: &str) -> bool {
        path.starts_with("rvf://") || path.starts_with("/rvf/")
    }

    /// Parse an RVF path into (package_name, internal_path).
    fn parse_rvf_path(path: &str) -> Option<(&str, &str)> {
        let stripped = path
            .strip_prefix("rvf://")
            .or_else(|| path.strip_prefix("/rvf/"))?;
        let slash_pos = stripped.find('/');
        match slash_pos {
            Some(pos) => Some((&stripped[..pos], &stripped[pos + 1..])),
            None => Some((stripped, "")),
        }
    }
}

/// Information about a tool from a mounted RVF package.
#[derive(Debug, Clone)]
pub struct MountedToolInfo {
    pub mount_handle: RvfMountHandle,
    pub name: String,
    pub description: String,
    pub parameters_schema: Option<serde_json::Value>,
}

#[async_trait]
impl<B: Backend + 'static> Backend for RvfStoreBackend<B> {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo> {
        if Self::is_rvf_path(path) {
            let table = self.mount_table.lock().unwrap();
            if let Some((pkg_name, _internal)) = Self::parse_rvf_path(path) {
                if pkg_name.is_empty() {
                    // List all mounted packages
                    return table
                        .list()
                        .iter()
                        .map(|e| FileInfo {
                            path: format!("rvf://{}", e.package_name),
                            is_dir: true,
                            size: 0,
                            modified_at: None,
                        })
                        .collect();
                }
                // O(1) lookup by name via index
                if let Some(entry) = table.get_by_name(pkg_name) {
                    return entry
                        .manifest
                        .entries
                        .iter()
                        .map(|e| FileInfo {
                            path: format!("rvf://{}/{}", pkg_name, e.name),
                            is_dir: false,
                            size: 0,
                            modified_at: None,
                        })
                        .collect();
                }
                return vec![];
            }
            // List all mounted packages
            return table
                .list()
                .iter()
                .map(|e| FileInfo {
                    path: format!("rvf://{}", e.package_name),
                    is_dir: true,
                    size: 0,
                    modified_at: None,
                })
                .collect();
        }
        self.inner.ls_info(path).await
    }

    async fn read_file(
        &self,
        file_path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, FileOperationError> {
        if Self::is_rvf_path(file_path) {
            let table = self.mount_table.lock().unwrap();
            if let Some((pkg_name, internal_path)) = Self::parse_rvf_path(file_path) {
                // O(1) lookup by name via index
                if let Some(entry) = table.get_by_name(pkg_name) {
                    if internal_path.is_empty() {
                        let json = serde_json::to_string_pretty(&entry.manifest)
                            .unwrap_or_else(|_| "Error serializing manifest".into());
                        return Ok(json);
                    }
                    if let Some(manifest_entry) = entry
                        .manifest
                        .entries
                        .iter()
                        .find(|e| e.name == internal_path)
                    {
                        return Ok(format!(
                            "RVF entry: {} (type: {:?}, version: {})\n{}",
                            manifest_entry.name,
                            manifest_entry.entry_type,
                            manifest_entry.version,
                            manifest_entry.description
                        ));
                    }
                    return Err(FileOperationError::FileNotFound);
                }
                return Err(FileOperationError::FileNotFound);
            }
            return Err(FileOperationError::InvalidPath);
        }
        self.inner.read_file(file_path, offset, limit).await
    }

    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult {
        if Self::is_rvf_path(file_path) {
            return WriteResult {
                error: Some("RVF packages are read-only".into()),
                path: None,
                files_update: None,
            };
        }
        self.inner.write_file(file_path, content).await
    }

    async fn edit_file(
        &self,
        file_path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> EditResult {
        if Self::is_rvf_path(file_path) {
            return EditResult {
                error: Some("RVF packages are read-only".into()),
                path: None,
                files_update: None,
                occurrences: None,
            };
        }
        self.inner
            .edit_file(file_path, old_string, new_string, replace_all)
            .await
    }

    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo> {
        if Self::is_rvf_path(path) {
            let table = self.mount_table.lock().unwrap();
            let search = pattern.trim_start_matches('*').trim_end_matches('*');
            let mut results = Vec::new();
            for entry in table.list() {
                for manifest_entry in &entry.manifest.entries {
                    if manifest_entry.name.contains(search) {
                        results.push(FileInfo {
                            path: format!("rvf://{}/{}", entry.package_name, manifest_entry.name),
                            is_dir: false,
                            size: 0,
                            modified_at: None,
                        });
                    }
                }
            }
            return results;
        }
        self.inner.glob_info(pattern, path).await
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        include_glob: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        // RVF packages don't support content grep — delegate to inner
        self.inner.grep(pattern, path, include_glob).await
    }

    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse> {
        // Separate RVF paths from regular paths
        let mut rvf_responses = Vec::new();
        let mut regular_paths = Vec::new();

        for path in paths {
            if Self::is_rvf_path(path) {
                rvf_responses.push(FileDownloadResponse {
                    path: path.clone(),
                    content: None,
                    error: Some(FileOperationError::PermissionDenied),
                });
            } else {
                regular_paths.push(path.clone());
            }
        }

        let mut results = rvf_responses;
        if !regular_paths.is_empty() {
            results.extend(self.inner.download_files(&regular_paths).await);
        }
        results
    }

    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse> {
        let mut rvf_responses = Vec::new();
        let mut regular_files = Vec::new();

        for (path, data) in files {
            if Self::is_rvf_path(path) {
                rvf_responses.push(FileUploadResponse {
                    path: path.clone(),
                    error: Some(FileOperationError::PermissionDenied),
                });
            } else {
                regular_files.push((path.clone(), data.clone()));
            }
        }

        let mut results = rvf_responses;
        if !regular_files.is_empty() {
            let refs: Vec<(String, Vec<u8>)> = regular_files;
            results.extend(self.inner.upload_files(&refs).await);
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateBackend;
    use rvagent_core::rvf_bridge::{RvfManifestEntry, RvfManifestEntryType};

    fn make_rvf_backend() -> RvfStoreBackend<StateBackend> {
        let inner = StateBackend::new();
        let config = RvfBridgeConfig {
            enabled: true,
            ..Default::default()
        };
        RvfStoreBackend::new(inner, config)
    }

    fn sample_manifest() -> RvfManifest {
        let mut manifest = RvfManifest::new("test-tools", "0.1.0");
        manifest.entries.push(RvfManifestEntry {
            name: "analyzer".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Analyze code quality".into(),
            version: "0.1.0".into(),
            parameters_schema: Some(serde_json::json!({"type": "object"})),
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "formatter".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Format code".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "deploy-skill".into(),
            entry_type: RvfManifestEntryType::Skill,
            description: "Deploy to production".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec!["execute".into()],
        });
        manifest
    }

    #[test]
    fn test_mount_and_list() {
        let backend = make_rvf_backend();
        let manifest = sample_manifest();
        let handle = backend.mount_package(manifest, RvfVerifyStatus::SignatureValid);
        assert!(!handle.is_null());

        let tools = backend.mounted_tools();
        assert_eq!(tools.len(), 2); // Only Tool entries, not Skill
    }

    #[tokio::test]
    async fn test_ls_rvf_packages() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let entries = backend.ls_info("rvf://").await;
        assert_eq!(entries.len(), 1);
        assert!(entries[0].path.contains("test-tools"));
        assert!(entries[0].is_dir);
    }

    #[tokio::test]
    async fn test_ls_rvf_package_contents() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let entries = backend.ls_info("rvf://test-tools").await;
        assert_eq!(entries.len(), 3); // 2 tools + 1 skill
    }

    #[tokio::test]
    async fn test_read_rvf_manifest() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let content = backend.read_file("rvf://test-tools", 0, 100).await.unwrap();
        assert!(content.contains("test-tools"));
        assert!(content.contains("analyzer"));
    }

    #[tokio::test]
    async fn test_read_rvf_entry() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let content = backend
            .read_file("rvf://test-tools/analyzer", 0, 100)
            .await
            .unwrap();
        assert!(content.contains("analyzer"));
        assert!(content.contains("Analyze code quality"));
    }

    #[tokio::test]
    async fn test_write_to_rvf_forbidden() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let result = backend
            .write_file("rvf://test-tools/new_file", "content")
            .await;
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("read-only"));
    }

    #[tokio::test]
    async fn test_edit_rvf_forbidden() {
        let backend = make_rvf_backend();
        let result = backend
            .edit_file("rvf://test-tools/x", "old", "new", false)
            .await;
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_glob_rvf() {
        let backend = make_rvf_backend();
        backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let results = backend.glob_info("*format*", "rvf://").await;
        assert_eq!(results.len(), 1);
        assert!(results[0].path.contains("formatter"));
    }

    #[test]
    fn test_unmount() {
        let backend = make_rvf_backend();
        let handle = backend.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);
        assert!(backend.unmount_package(handle));

        let table = backend.mount_table().lock().unwrap();
        assert!(table.is_empty());
    }

    #[tokio::test]
    async fn test_fallthrough_to_inner() {
        let backend = make_rvf_backend();
        // Non-RVF paths should delegate to inner backend
        let result = backend.read_file("/some/file.txt", 0, 100).await;
        // StateBackend returns error for missing files
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_rvf_path() {
        assert_eq!(
            RvfStoreBackend::<StateBackend>::parse_rvf_path("rvf://pkg-a/tool_x"),
            Some(("pkg-a", "tool_x"))
        );
        assert_eq!(
            RvfStoreBackend::<StateBackend>::parse_rvf_path("rvf://pkg-a"),
            Some(("pkg-a", ""))
        );
        assert_eq!(
            RvfStoreBackend::<StateBackend>::parse_rvf_path("/rvf/pkg-b/sub/path"),
            Some(("pkg-b", "sub/path"))
        );
        assert_eq!(
            RvfStoreBackend::<StateBackend>::parse_rvf_path("/other/path"),
            None
        );
    }

    #[test]
    fn test_shared_mount_table() {
        let config = RvfBridgeConfig::default();
        let mount_table = Arc::new(Mutex::new(MountTable::new()));

        let backend1 = RvfStoreBackend::with_mount_table(
            StateBackend::new(),
            config.clone(),
            mount_table.clone(),
        );
        let backend2 = RvfStoreBackend::with_mount_table(StateBackend::new(), config, mount_table);

        // Mount via backend1, visible in backend2
        backend1.mount_package(sample_manifest(), RvfVerifyStatus::SignatureValid);

        let tools = backend2.mounted_tools();
        assert_eq!(tools.len(), 2);
    }
}
