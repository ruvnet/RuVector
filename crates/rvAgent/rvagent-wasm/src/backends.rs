//! WASM-compatible backend implementations.
//!
//! These backends operate entirely in-memory or via web-sys fetch,
//! since direct filesystem access is unavailable in the browser sandbox.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmStateBackend — in-memory virtual filesystem
// ---------------------------------------------------------------------------

/// In-memory state backend for WASM environments.
///
/// Stores files in a `HashMap<String, String>` keyed by virtual path.
/// No real filesystem access is performed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WasmStateBackend {
    /// Virtual filesystem: path -> content.
    files: HashMap<String, String>,
}

impl WasmStateBackend {
    /// Create a new empty state backend.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Read a file from the virtual filesystem.
    pub fn read_file(&self, path: &str) -> Result<String, WasmBackendError> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| WasmBackendError::NotFound(path.to_string()))
    }

    /// Write a file to the virtual filesystem. Creates or overwrites.
    pub fn write_file(&mut self, path: &str, content: &str) -> Result<(), WasmBackendError> {
        // Security: Check content size
        if content.len() > MAX_FILE_SIZE {
            return Err(WasmBackendError::LimitExceeded(format!(
                "File size {} exceeds maximum {}",
                content.len(),
                MAX_FILE_SIZE
            )));
        }

        // Security: Check file count limit
        if !self.files.contains_key(path) && self.files.len() >= MAX_FILES {
            return Err(WasmBackendError::LimitExceeded(format!(
                "File count {} exceeds maximum {}",
                self.files.len(),
                MAX_FILES
            )));
        }

        let normalized = normalize_path(path)?;
        self.files.insert(normalized, content.to_string());
        Ok(())
    }

    /// Apply an edit to an existing file: replace `old` with `new` in the file content.
    pub fn edit_file(&mut self, path: &str, old: &str, new: &str) -> Result<(), WasmBackendError> {
        let content = self.read_file(path)?;
        if !content.contains(old) {
            return Err(WasmBackendError::EditMismatch {
                path: path.to_string(),
                needle: old.to_string(),
            });
        }
        let updated = content.replacen(old, new, 1);
        self.files.insert(path.to_string(), updated);
        Ok(())
    }

    /// Delete a file from the virtual filesystem.
    pub fn delete_file(&mut self, path: &str) -> Result<(), WasmBackendError> {
        self.files
            .remove(path)
            .map(|_| ())
            .ok_or_else(|| WasmBackendError::NotFound(path.to_string()))
    }

    /// List all file paths in the virtual filesystem.
    pub fn list_files(&self) -> Vec<String> {
        let mut paths: Vec<String> = self.files.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// Check whether a file exists.
    pub fn file_exists(&self, path: &str) -> bool {
        self.files.contains_key(path)
    }

    /// Clear all files from the virtual filesystem.
    pub fn clear(&mut self) {
        self.files.clear();
    }

    /// Get the number of files.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Serialize the entire state to JSON for persistence / export.
    pub fn to_json(&self) -> Result<String, WasmBackendError> {
        serde_json::to_string(&self.files).map_err(WasmBackendError::Serialization)
    }

    /// Restore state from a JSON snapshot.
    pub fn from_json(json: &str) -> Result<Self, WasmBackendError> {
        let files: HashMap<String, String> =
            serde_json::from_str(json).map_err(WasmBackendError::Serialization)?;
        Ok(Self { files })
    }
}

// ---------------------------------------------------------------------------
// WasmFetchBackend — remote file operations via web-sys fetch
// ---------------------------------------------------------------------------

/// Backend that uses the browser Fetch API for remote file operations.
///
/// Suitable for loading files from a remote server or API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFetchBackend {
    /// Base URL for fetch requests (e.g. "https://api.example.com/files").
    pub base_url: String,
    /// Optional authorization header value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
}

impl WasmFetchBackend {
    /// Create a new fetch backend with the given base URL.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_header: None,
        }
    }

    /// Set an authorization header (e.g. "Bearer <token>").
    pub fn with_auth(mut self, auth: &str) -> Self {
        self.auth_header = Some(auth.to_string());
        self
    }

    /// Fetch a file from `{base_url}/{path}`.
    pub async fn fetch_file(&self, path: &str) -> Result<String, WasmBackendError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let resp_value = self.do_fetch(&url, "GET", None).await?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| WasmBackendError::FetchError("response cast failed".into()))?;

        if !resp.ok() {
            return Err(WasmBackendError::FetchError(format!(
                "HTTP {} for {}",
                resp.status(),
                url
            )));
        }

        let text_promise = resp
            .text()
            .map_err(|_| WasmBackendError::FetchError("text() failed".into()))?;
        let text_value = wasm_bindgen_futures::JsFuture::from(text_promise)
            .await
            .map_err(|e| WasmBackendError::FetchError(format!("{:?}", e)))?;

        text_value
            .as_string()
            .ok_or_else(|| WasmBackendError::FetchError("response was not a string".into()))
    }

    /// PUT a file to `{base_url}/{path}`.
    pub async fn put_file(&self, path: &str, content: &str) -> Result<(), WasmBackendError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let resp_value = self.do_fetch(&url, "PUT", Some(content)).await?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| WasmBackendError::FetchError("response cast failed".into()))?;

        if !resp.ok() {
            return Err(WasmBackendError::FetchError(format!(
                "HTTP {} for PUT {}",
                resp.status(),
                url
            )));
        }
        Ok(())
    }

    /// Internal: perform a fetch request.
    async fn do_fetch(
        &self,
        url: &str,
        method: &str,
        body: Option<&str>,
    ) -> Result<JsValue, WasmBackendError> {
        let opts = web_sys::RequestInit::new();
        opts.set_method(method);
        opts.set_mode(web_sys::RequestMode::Cors);

        if let Some(body_str) = body {
            opts.set_body(&JsValue::from_str(body_str));
        }

        let request = web_sys::Request::new_with_str_and_init(url, &opts)
            .map_err(|e| WasmBackendError::FetchError(format!("Request::new failed: {:?}", e)))?;

        if let Some(ref auth) = self.auth_header {
            request.headers().set("Authorization", auth).map_err(|e| {
                WasmBackendError::FetchError(format!("set auth header failed: {:?}", e))
            })?;
        }

        request
            .headers()
            .set("Content-Type", "application/json")
            .map_err(|e| {
                WasmBackendError::FetchError(format!("set content-type failed: {:?}", e))
            })?;

        // Use global fetch (works in both Window and Worker scopes).
        let global = js_sys::global();
        let promise = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
            .map_err(|_| WasmBackendError::FetchError("global.fetch not found".into()))?;
        let fetch_fn: js_sys::Function = promise
            .dyn_into()
            .map_err(|_| WasmBackendError::FetchError("fetch is not a function".into()))?;

        let resp_promise = fetch_fn
            .call1(&JsValue::NULL, &request)
            .map_err(|e| WasmBackendError::FetchError(format!("fetch call failed: {:?}", e)))?;
        let resp_promise: js_sys::Promise = resp_promise
            .dyn_into()
            .map_err(|_| WasmBackendError::FetchError("fetch did not return a promise".into()))?;

        wasm_bindgen_futures::JsFuture::from(resp_promise)
            .await
            .map_err(|e| WasmBackendError::FetchError(format!("fetch rejected: {:?}", e)))
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from WASM backend operations.
#[derive(Debug, thiserror::Error)]
pub enum WasmBackendError {
    /// File not found in the virtual filesystem.
    #[error("file not found: {0}")]
    NotFound(String),

    /// Edit target string not found in file.
    #[error("edit target not found in {path}: {needle}")]
    EditMismatch { path: String, needle: String },

    /// Serialization / deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Fetch API error.
    #[error("fetch error: {0}")]
    FetchError(String),

    /// Security violation.
    #[error("security error: {0}")]
    SecurityError(String),

    /// Resource limit exceeded.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),
}

impl From<WasmBackendError> for JsValue {
    fn from(err: WasmBackendError) -> JsValue {
        JsValue::from_str(&err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Security Constants
// ---------------------------------------------------------------------------

/// Maximum path length (256 characters)
pub const MAX_PATH_LENGTH: usize = 256;

/// Maximum file content size (1 MB per file)
pub const MAX_FILE_SIZE: usize = 1024 * 1024;

/// Maximum number of files in the virtual filesystem
pub const MAX_FILES: usize = 10000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalize a virtual file path (remove leading `./`, collapse double slashes).
/// Returns error for path traversal attempts.
fn normalize_path(path: &str) -> Result<String, WasmBackendError> {
    // Security: Check path length
    if path.len() > MAX_PATH_LENGTH {
        return Err(WasmBackendError::SecurityError(format!(
            "Path length {} exceeds maximum {}",
            path.len(),
            MAX_PATH_LENGTH
        )));
    }

    // Security: Reject path traversal attempts
    if path.contains("..") {
        return Err(WasmBackendError::SecurityError(
            "Path traversal (..) is not allowed".to_string(),
        ));
    }

    // Security: Reject absolute paths that could escape sandbox
    if path.starts_with('/') && path.contains("etc") {
        return Err(WasmBackendError::SecurityError(
            "Suspicious path pattern detected".to_string(),
        ));
    }

    let p = path.trim_start_matches("./");
    let p = p.replace("//", "/");
    if p.is_empty() {
        Ok("/".to_string())
    } else {
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_backend_read_write() {
        let mut backend = WasmStateBackend::new();
        assert!(backend.read_file("test.txt").is_err());

        backend.write_file("test.txt", "hello").unwrap();
        assert_eq!(backend.read_file("test.txt").unwrap(), "hello");
    }

    #[test]
    fn test_state_backend_edit() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("main.rs", "fn main() {}").unwrap();
        backend
            .edit_file("main.rs", "fn main()", "fn main() -> i32")
            .unwrap();
        assert_eq!(backend.read_file("main.rs").unwrap(), "fn main() -> i32 {}");
    }

    #[test]
    fn test_state_backend_edit_not_found() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("a.txt", "abc").unwrap();
        let err = backend.edit_file("a.txt", "xyz", "replaced").unwrap_err();
        assert!(matches!(err, WasmBackendError::EditMismatch { .. }));
    }

    #[test]
    fn test_state_backend_delete() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("f.txt", "data").unwrap();
        backend.delete_file("f.txt").unwrap();
        assert!(!backend.file_exists("f.txt"));
    }

    #[test]
    fn test_state_backend_list() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("b.txt", "b").unwrap();
        backend.write_file("a.txt", "a").unwrap();
        let files = backend.list_files();
        assert_eq!(files, vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn test_state_backend_clear() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("x.txt", "x").unwrap();
        assert_eq!(backend.file_count(), 1);
        backend.clear();
        assert_eq!(backend.file_count(), 0);
    }

    #[test]
    fn test_state_backend_json_roundtrip() {
        let mut backend = WasmStateBackend::new();
        backend.write_file("a.rs", "code").unwrap();
        backend.write_file("b.rs", "more code").unwrap();
        let json = backend.to_json().unwrap();
        let restored = WasmStateBackend::from_json(&json).unwrap();
        assert_eq!(restored.read_file("a.rs").unwrap(), "code");
        assert_eq!(restored.read_file("b.rs").unwrap(), "more code");
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("./src/main.rs").unwrap(), "src/main.rs");
        assert_eq!(normalize_path("a//b.txt").unwrap(), "a/b.txt");
        assert_eq!(normalize_path("").unwrap(), "/");
    }

    #[test]
    fn test_normalize_path_security() {
        // Path traversal should be rejected
        assert!(normalize_path("../etc/passwd").is_err());
        assert!(normalize_path("foo/../bar").is_err());

        // Long paths should be rejected
        let long_path = "a".repeat(300);
        assert!(normalize_path(&long_path).is_err());
    }

    #[test]
    fn test_write_file_size_limit() {
        let mut backend = WasmStateBackend::new();
        let huge_content = "x".repeat(MAX_FILE_SIZE + 1);
        let result = backend.write_file("huge.txt", &huge_content);
        assert!(matches!(result, Err(WasmBackendError::LimitExceeded(_))));
    }

    #[test]
    fn test_fetch_backend_new() {
        let fb = WasmFetchBackend::new("https://api.example.com/files/");
        assert_eq!(fb.base_url, "https://api.example.com/files");
        assert!(fb.auth_header.is_none());
    }

    #[test]
    fn test_fetch_backend_with_auth() {
        let fb = WasmFetchBackend::new("https://api.example.com").with_auth("Bearer tok123");
        assert_eq!(fb.auth_header.as_deref(), Some("Bearer tok123"));
    }
}
