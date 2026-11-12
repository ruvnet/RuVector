//! The §16.3 private operation contract: envelope types, per-op
//! authorization, executors and the dispatcher, running against an
//! in-process [`LocalCluster`] of ledger and shard stores.

pub mod authz;
pub mod cluster;
pub mod dispatch;
mod exec;
pub mod types;
mod vector;

pub use authz::{authorize, requirement, OpRequirement};
pub use cluster::LocalCluster;
pub use dispatch::{Dispatcher, OpReply};
pub use types::{canonical_json, Op, OpRequest, OpResponse, OpUsage, WireError};
