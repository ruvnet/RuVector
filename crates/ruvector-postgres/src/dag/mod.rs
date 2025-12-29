//! Neural DAG learning for PostgreSQL query optimization
//!
//! This module integrates the SONA (Scalable On-device Neural Adaptation) engine
//! with PostgreSQL's query planner to provide learned query optimization.

pub mod state;
pub mod functions;

pub use state::{DAG_STATE, DagState, DagConfig};
