//! Portable proximity-graph sharding for edge deployment and agent memory.
//!
//! Extracts a coherent subgraph (shard) from a large vector proximity graph,
//! serializes it to a compact binary format, and enables standalone ANN search
//! on the shard without loading the full index.

pub mod error;
pub mod graph;
pub mod search;
pub mod shard;
pub mod wire;

pub use error::{ShardError, ShardResult};
pub use graph::{cosine_distance, GraphConfig, KnnGraph};
pub use search::{brute_force_knn, recall_at_k, search_shard};
pub use shard::{
    BfsShard, CoherenceShard, HubShard, Shard, ShardExtractor, ShardMeta, ShardVariant,
};
pub use wire::{read_shard, write_shard};
