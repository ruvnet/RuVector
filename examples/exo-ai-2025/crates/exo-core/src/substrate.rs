//! Substrate implementation using ruvector as backend

use crate::error::{Error, Result};
use crate::types::*;
use ruvector_core::{DbOptions, DistanceMetric, VectorDB, VectorEntry};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Cognitive substrate instance
pub struct SubstrateInstance {
    /// Vector database backend
    db: Arc<RwLock<VectorDB>>,
    /// Configuration
    config: SubstrateConfig,
}

impl SubstrateInstance {
    /// Create a new substrate instance
    pub fn new(config: SubstrateConfig) -> Result<Self> {
        let db_options = DbOptions {
            dimensions: config.dimensions,
            distance_metric: DistanceMetric::Cosine,
            storage_path: config.storage_path.clone(),
            hnsw_config: None,
            quantization: None,
        };

        let db = VectorDB::new(db_options)
            .map_err(|e| Error::Backend(format!("Failed to create VectorDB: {}", e)))?;

        Ok(Self {
            db: Arc::new(RwLock::new(db)),
            config,
        })
    }

    /// Store a pattern in the substrate
    pub async fn store(&self, pattern: Pattern) -> Result<String> {
        let entry = VectorEntry {
            id: None,
            vector: pattern.embedding.clone(),
            metadata: Some(serde_json::to_value(&pattern.metadata)?),
        };

        let db = self.db.read().await;
        let id = db
            .insert(entry)
            .map_err(|e| Error::Backend(format!("Failed to insert pattern: {}", e)))?;

        Ok(id)
    }

    /// Search for similar patterns
    pub async fn search(&self, query: Query) -> Result<Vec<SearchResult>> {
        let search_query = ruvector_core::SearchQuery {
            vector: query.embedding.clone(),
            k: query.k,
            filter: None,
            ef_search: None,
        };

        let db = self.db.read().await;
        let results = db
            .search(search_query)
            .map_err(|e| Error::Backend(format!("Failed to search: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.score,
                pattern: None, // TODO: Retrieve full pattern if needed
            })
            .collect())
    }

    /// Query hypergraph topology
    pub async fn hypergraph_query(&self, _query: TopologicalQuery) -> Result<HypergraphResult> {
        if !self.config.enable_hypergraph {
            return Ok(HypergraphResult::NotSupported);
        }

        // TODO: Implement hypergraph queries
        Ok(HypergraphResult::NotSupported)
    }

    /// Get substrate statistics
    pub async fn stats(&self) -> Result<SubstrateStats> {
        let db = self.db.read().await;
        let len = db
            .len()
            .map_err(|e| Error::Backend(format!("Failed to get length: {}", e)))?;

        Ok(SubstrateStats {
            total_patterns: len,
            dimensions: self.config.dimensions,
        })
    }
}

/// Substrate statistics
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubstrateStats {
    /// Total number of patterns
    pub total_patterns: usize,
    /// Vector dimensions
    pub dimensions: usize,
}
