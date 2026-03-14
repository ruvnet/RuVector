# ADR-094: Backend Protocol & Trait System

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crate**   | `ruvector-deep-backends`                        |

## Context

DeepAgents defines a `BackendProtocol` ABC with 12 methods (sync + async pairs) for file operations, plus `SandboxBackendProtocol` extending it with `execute()`. Five concrete implementations exist:

1. **StateBackend** — Ephemeral, stores files in LangGraph state dict
2. **FilesystemBackend** — Local disk with ripgrep integration
3. **LocalShellBackend** — Filesystem + unrestricted shell execution
4. **CompositeBackend** — Path-prefix routing to multiple backends
5. **BaseSandbox** — Abstract, implements all file ops via `execute()` shell commands

## Decision

### Core Traits

```rust
// crates/ruvector-deep-backends/src/protocol.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Standardized error codes for file operations (LLM-actionable).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileOperationError {
    FileNotFound,
    PermissionDenied,
    IsDirectory,
    InvalidPath,
}

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

#[derive(Debug, Clone)]
pub struct FileDownloadResponse {
    pub path: String,
    pub content: Option<Vec<u8>>,
    pub error: Option<FileOperationError>,
}

#[derive(Debug, Clone)]
pub struct FileUploadResponse {
    pub path: String,
    pub error: Option<FileOperationError>,
}

#[derive(Debug, Clone)]
pub struct GrepMatch {
    pub path: String,
    pub line: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct WriteResult {
    pub error: Option<String>,
    pub path: Option<String>,
    pub files_update: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct EditResult {
    pub error: Option<String>,
    pub path: Option<String>,
    pub files_update: Option<HashMap<String, serde_json::Value>>,
    pub occurrences: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ExecuteResponse {
    pub output: String,
    pub exit_code: Option<i32>,
    pub truncated: bool,
}

/// Core backend trait — all file operations.
/// Python: BackendProtocol
#[async_trait]
pub trait Backend: Send + Sync {
    fn ls_info(&self, path: &str) -> Vec<FileInfo>;
    async fn als_info(&self, path: &str) -> Vec<FileInfo> {
        tokio::task::spawn_blocking({
            let this = self.clone_box();
            let path = path.to_string();
            move || this.ls_info(&path)
        }).await.unwrap()
    }

    fn read(&self, file_path: &str, offset: usize, limit: usize) -> String;
    async fn aread(&self, file_path: &str, offset: usize, limit: usize) -> String;

    fn grep_raw(&self, pattern: &str, path: Option<&str>, glob: Option<&str>)
        -> Result<Vec<GrepMatch>, String>;
    async fn agrep_raw(&self, pattern: &str, path: Option<&str>, glob: Option<&str>)
        -> Result<Vec<GrepMatch>, String>;

    fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo>;
    async fn aglob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo>;

    fn write(&self, file_path: &str, content: &str) -> WriteResult;
    async fn awrite(&self, file_path: &str, content: &str) -> WriteResult;

    fn edit(&self, file_path: &str, old_string: &str, new_string: &str, replace_all: bool)
        -> EditResult;
    async fn aedit(&self, file_path: &str, old_string: &str, new_string: &str, replace_all: bool)
        -> EditResult;

    fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse>;
    async fn aupload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse>;

    fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse>;
    async fn adownload_files(&self, paths: &[String]) -> Vec<FileDownloadResponse>;
}

/// Extension trait for backends with shell execution.
/// Python: SandboxBackendProtocol
#[async_trait]
pub trait SandboxBackend: Backend {
    fn id(&self) -> &str;
    fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;
    async fn aexecute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;
}
```

### Backend Implementations

#### StateBackend

```rust
// Python: StateBackend — stores files in agent state (HashMap)
pub struct StateBackend {
    state: Arc<RwLock<AgentState>>,
}
```

Maps directly: Python's `runtime.state.get("files", {})` → Rust `state.read().files`.

#### FilesystemBackend

```rust
pub struct FilesystemBackend {
    cwd: PathBuf,
    virtual_mode: bool,
    max_file_size_bytes: u64,
}
```

Key mappings:
- `_resolve_path()` → `resolve_path()` with same virtual_mode logic
- `_ripgrep_search()` → Shell out to `rg --json -F` (same as Python)
- `_python_search()` → Native Rust `walkdir` + `regex` fallback
- `wcmatch.glob` → `globset` crate

#### LocalShellBackend

```rust
pub struct LocalShellBackend {
    inner: FilesystemBackend,
    default_timeout: u32,
    max_output_bytes: usize,
    env: HashMap<String, String>,
    sandbox_id: String,
}

impl SandboxBackend for LocalShellBackend {
    fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
        // std::process::Command with shell=true equivalent
        // Combined stdout/stderr with [stderr] prefix — same as Python
    }
}
```

#### CompositeBackend

```rust
pub struct CompositeBackend {
    default: Box<dyn Backend>,
    routes: Vec<(String, Box<dyn Backend>)>, // sorted by prefix length desc
}
```

Preserves exact routing logic: longest-prefix-first matching, path stripping, result remapping.

#### BaseSandbox

```rust
pub trait BaseSandbox: SandboxBackend {
    // Default implementations for all Backend methods using execute()
    // Same Python command templates (_GLOB_COMMAND_TEMPLATE, etc.)
}
```

### Type Mapping Details

| Python Type | Rust Type |
|---|---|
| `dict[str, Any]` (file data) | `FileData { content: Vec<String>, created_at: String, modified_at: String }` |
| `list[FileInfo]` | `Vec<FileInfo>` |
| `list[GrepMatch] \| str` | `Result<Vec<GrepMatch>, String>` |
| `WriteResult` (dataclass) | `WriteResult` (struct) |
| `EditResult` (dataclass) | `EditResult` (struct) |
| `ExecuteResponse` (dataclass) | `ExecuteResponse` (struct) |
| `BackendFactory` (Callable) | `Box<dyn Fn(&ToolRuntime) -> Box<dyn Backend>>` |

### Crate Dependencies

```toml
[dependencies]
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
walkdir = "2"
globset = "0.4"
regex = "1"
chrono = "0.4"
```

## Fidelity Verification

For each backend method, we verify:

1. **Path resolution** — Same behavior for absolute, relative, virtual paths
2. **Error codes** — Same `FileOperationError` variants for same conditions
3. **Line numbering** — `cat -n` format (1-indexed, 6-char width, tab separator)
4. **Grep output** — Identical `GrepMatch` structs for same input
5. **Edit semantics** — Same replace_all=false uniqueness check
6. **Execute output** — Same `[stderr]` prefixing, truncation, exit code formatting

## Consequences

- All 5 backend implementations fully ported with identical behavior
- `async_trait` provides async/sync parity matching Python's `asyncio.to_thread` pattern
- `CompositeBackend` routing is zero-cost (sorted prefix matching)
- WASM targets can use `StateBackend` (no filesystem needed)
