//! rvAgent ACP — Agent Communication Protocol server.
//!
//! Provides an axum-based HTTP server implementing the ACP protocol
//! with authentication, rate limiting, and body size enforcement
//! per ADR-099 and ADR-103 C6.

pub mod agent;
pub mod auth;
pub mod server;
pub mod types;
