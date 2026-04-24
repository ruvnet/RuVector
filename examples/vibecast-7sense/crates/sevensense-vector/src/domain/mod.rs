//! Domain layer for the Vector Space bounded context.
//!
//! Contains:
//! - Entities: Core domain objects with identity
//! - Value Objects: Immutable objects defined by their attributes
//! - Repository Traits: Abstractions for persistence
//! - Domain Errors: Error types specific to this context

pub mod entities;
pub mod error;
pub mod repository;

pub use entities::*;
pub use error::*;
pub use repository::*;
