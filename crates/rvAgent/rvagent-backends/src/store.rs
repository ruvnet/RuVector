//! StoreBackend — persistent storage backend using the filesystem.
//!
//! Provides a key-value-style storage abstraction backed by the
//! local filesystem. Used for persistent agent state, checkpoints,
//! and artifact storage.

use crate::filesystem::FilesystemBackend;
use crate::protocol::*;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// Persistent storage backend using the filesystem.
///
/// Wraps `FilesystemBackend` to provide persistent key-value storage.
/// Keys are mapped to file paths within the store root directory.
#[derive(Clone)]
pub struct StoreBackend {
    inner: FilesystemBackend,
    store_root: PathBuf,
}

impl StoreBackend {
    /// Create a new store backend at the given root directory.
    ///
    /// Creates the root directory if it doesn't exist.
    pub fn new(store_root: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&store_root)?;
        Ok(Self {
            inner: FilesystemBackend::new(store_root.clone()),
            store_root,
        })
    }

    /// Get the store root directory.
    pub fn store_root(&self) -> &Path {
        &self.store_root
    }

    /// Store a value under the given key.
    pub async fn store(&self, key: &str, value: &str) -> WriteResult {
        self.inner.write_file(key, value).await
    }

    /// Retrieve a value by key.
    pub async fn retrieve(&self, key: &str) -> Result<String, FileOperationError> {
        // Read without line numbers — return raw content
        let resolved = self
            .inner
            .resolve_path(key)
            .map_err(|_| FileOperationError::FileNotFound)?;
        let content = tokio::task::spawn_blocking(move || std::fs::read_to_string(&resolved))
            .await
            .map_err(|_| FileOperationError::InvalidPath)?
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
                std::io::ErrorKind::PermissionDenied => FileOperationError::PermissionDenied,
                _ => FileOperationError::InvalidPath,
            })?;
        Ok(content)
    }

    /// Delete a stored value.
    pub async fn delete(&self, key: &str) -> Result<(), FileOperationError> {
        let resolved = self
            .inner
            .resolve_path(key)
            .map_err(|_| FileOperationError::InvalidPath)?;
        tokio::task::spawn_blocking(move || std::fs::remove_file(&resolved))
            .await
            .map_err(|_| FileOperationError::InvalidPath)?
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => FileOperationError::FileNotFound,
                std::io::ErrorKind::PermissionDenied => FileOperationError::PermissionDenied,
                _ => FileOperationError::InvalidPath,
            })?;
        Ok(())
    }

    /// Check if a key exists in the store.
    pub async fn exists(&self, key: &str) -> bool {
        let resolved = match self.inner.resolve_path(key) {
            Ok(p) => p,
            Err(_) => return false,
        };
        tokio::task::spawn_blocking(move || resolved.exists())
            .await
            .unwrap_or(false)
    }

    /// List all keys in the store (relative paths).
    pub async fn list_keys(&self) -> Vec<String> {
        self.inner
            .ls_info("")
            .await
            .into_iter()
            .filter(|info| !info.is_dir)
            .map(|info| info.path)
            .collect()
    }
}

#[async_trait]
impl Backend for StoreBackend {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        store.store("key1.txt", "value1").await;
        let retrieved = store.retrieve("key1.txt").await.unwrap();
        assert_eq!(retrieved, "value1");
    }

    #[tokio::test]
    async fn test_retrieve_not_found() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        let result = store.retrieve("nonexistent.txt").await;
        assert_eq!(result.unwrap_err(), FileOperationError::FileNotFound);
    }

    #[tokio::test]
    async fn test_delete() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        store.store("deleteme.txt", "gone").await;
        assert!(store.exists("deleteme.txt").await);

        store.delete("deleteme.txt").await.unwrap();
        assert!(!store.exists("deleteme.txt").await);
    }

    #[tokio::test]
    async fn test_exists() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        assert!(!store.exists("nope.txt").await);
        store.store("yep.txt", "data").await;
        assert!(store.exists("yep.txt").await);
    }

    #[tokio::test]
    async fn test_list_keys() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        store.store("a.txt", "aaa").await;
        store.store("b.txt", "bbb").await;

        let keys = store.list_keys().await;
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_store_root() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();
        assert_eq!(store.store_root(), tmp.path());
    }

    #[tokio::test]
    async fn test_path_traversal_blocked() {
        let tmp = TempDir::new().unwrap();
        let store = StoreBackend::new(tmp.path().to_path_buf()).unwrap();

        let result = store.retrieve("../../../etc/passwd").await;
        assert!(result.is_err());
    }
}
