//! rvAgent backends — filesystem, shell, composite, state, store, and sandbox protocols.
//!
//! This crate provides all backend implementations for rvAgent, following
//! ADR-094 (Backend Protocol & Trait System) and ADR-103 (Review Amendments).
//!
//! # Backend implementations
//!
//! - [`StateBackend`](state::StateBackend) — Ephemeral in-memory file store
//! - [`FilesystemBackend`](filesystem::FilesystemBackend) — Local disk with security hardening
//! - [`LocalShellBackend`](local_shell::LocalShellBackend) — Filesystem + shell execution
//! - [`CompositeBackend`](composite::CompositeBackend) — Path-prefix routing to sub-backends
//! - [`StoreBackend`](store::StoreBackend) — Persistent key-value storage
//!
//! # Security features (ADR-103)
//!
//! - Path traversal protection with atomic resolve+open (SEC-001)
//! - Environment sanitization for shell execution (SEC-005)
//! - Unicode security detection and stripping (SEC-016)
//! - Composite path re-validation after prefix stripping (SEC-003)
//! - Literal grep mode to prevent ReDoS (SEC-021)

pub mod anthropic;
pub mod composite;
pub mod filesystem;
pub mod gemini;
pub mod local_shell;
pub mod protocol;
pub mod rvf_store;
pub mod sandbox;
pub mod security;
pub mod state;
pub mod store;
pub mod unicode_security;
pub mod utils;

// Re-export core types for convenience.
pub use anthropic::AnthropicClient;
pub use composite::{BackendRef, CompositeBackend};
pub use filesystem::FilesystemBackend;
pub use local_shell::{CommandAllowlist, LocalShellBackend, LocalShellConfig};
pub use protocol::{
    Backend, EditResult, ExecuteResponse, FileData, FileDownloadResponse, FileInfo,
    FileOperationError, FileUploadResponse, GrepMatch, SandboxBackend, WriteResult,
};
pub use rvf_store::MountedToolInfo;
pub use sandbox::{BaseSandbox, LocalSandbox, SandboxConfig, SandboxError};
pub use state::StateBackend;
pub use store::StoreBackend;
