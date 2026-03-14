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

pub mod protocol;
pub mod utils;
pub mod unicode_security;
pub mod state;
pub mod filesystem;
pub mod local_shell;
pub mod composite;
pub mod sandbox;
pub mod store;

// Re-export core types for convenience.
pub use protocol::{
    Backend, SandboxBackend, FileOperationError, FileInfo, FileData,
    FileDownloadResponse, FileUploadResponse, GrepMatch,
    WriteResult, EditResult, ExecuteResponse,
};
pub use state::StateBackend;
pub use filesystem::FilesystemBackend;
pub use local_shell::{LocalShellBackend, LocalShellConfig, CommandAllowlist};
pub use composite::{CompositeBackend, BackendRef};
pub use sandbox::{BaseSandbox, SandboxConfig};
pub use store::StoreBackend;
