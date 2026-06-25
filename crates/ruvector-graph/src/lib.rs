//! # RuVector Graph Database
//!
//! A high-performance graph database layer built on RuVector with Neo4j compatibility.
//! Supports property graphs, hypergraphs, Cypher queries, ACID transactions, and distributed queries.

pub mod bm25;
pub mod codegen;
pub mod cypher;
pub mod edge;
pub mod embed;
pub mod error;
pub mod executor;
pub mod graph;
pub mod hyperedge;
pub mod index;
pub mod node;
pub mod schema;
pub mod storage;
pub mod transaction;
pub mod typed_graph;
pub mod types;

// Performance optimization modules
pub mod optimization;

// Vector-graph hybrid query capabilities
pub mod hybrid;

// Distributed graph capabilities
#[cfg(feature = "distributed")]
pub mod distributed;

// Core type re-exports
pub use bm25::{Bm25Index, Bm25Params};
pub use edge::{Edge, EdgeBuilder};
pub use embed::{Embedder, HashEmbedder};
pub use error::{GraphError, Result};
pub use graph::GraphDB;
pub use hyperedge::{Hyperedge, HyperedgeBuilder, HyperedgeId};
pub use node::{Node, NodeBuilder};
pub use schema::{
    reciprocal_rank_fusion, DistanceMetric, EdgeSchema, GraphSchema, NodeSchema, PropertySchema,
    PropertyType, VectorSchema,
};
#[cfg(feature = "storage")]
pub use storage::GraphStorage;
pub use transaction::{IsolationLevel, Transaction, TransactionManager};
pub use typed_graph::{Direction, TraversalResult, TraverseSpec, TypedGraph};
pub use types::{EdgeId, Label, NodeId, Properties, PropertyValue, RelationType};

// Re-export hybrid query types when available
#[cfg(not(feature = "minimal"))]
pub use hybrid::{
    EmbeddingConfig, GnnConfig, GraphNeuralEngine, HybridIndex, RagConfig, RagEngine,
    SemanticSearch, VectorCypherParser,
};

// Re-export distributed types when feature is enabled
#[cfg(feature = "distributed")]
pub use distributed::{
    Coordinator, Federation, GossipMembership, GraphReplication, GraphShard, RpcClient, RpcServer,
    ShardCoordinator, ShardStrategy,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        // Placeholder test to allow compilation
        assert!(true);
    }
}
