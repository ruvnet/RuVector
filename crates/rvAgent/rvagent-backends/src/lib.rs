//! `rvagent-backends` — Backend implementations for the rvAgent framework.
//!
//! This crate provides:
//!
//! - [`protocol`] — Core backend traits (`Backend`, `SandboxBackend`) and types
//! - [`unicode_security`] — Unicode security detection and validation (ADR-103 C7)
//! - [`utils`] — Shared utility functions (line formatting, path validation)

pub mod protocol;
pub mod unicode_security;
pub mod utils;
