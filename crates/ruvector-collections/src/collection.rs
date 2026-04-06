//! Collection types and operations

use ruvector_core::types::{DistanceMetric, HnswConfig, QuantizationConfig};
use ruvector_core::vector_db::VectorDB;
use serde::{Deserialize, Serialize};

use crate::error::{CollectionError, Result};

/// Configuration for creating a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Vector dimensions
    pub dimensions: usize,

    /// Distance metric for similarity calculation
    pub distance_metric: DistanceMetric,

    /// HNSW index configuration
    pub hnsw_config: Option<HnswConfig>,

    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,

    /// Whether to store payload data on disk
    pub on_disk_payload: bool,
}

impl CollectionConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.dimensions == 0 {
            return Err(CollectionError::InvalidConfiguration {
                message: "Dimensions must be greater than 0".to_string(),
            });
        }

        if self.dimensions > 100_000 {
            return Err(CollectionError::InvalidConfiguration {
                message: "Dimensions exceeds maximum of 100,000".to_string(),
            });
        }

        // Validate HNSW config if present
        if let Some(ref hnsw_config) = self.hnsw_config {
            if hnsw_config.m == 0 {
                return Err(CollectionError::InvalidConfiguration {
                    message: "HNSW M parameter must be greater than 0".to_string(),
                });
            }

            if hnsw_config.ef_construction < hnsw_config.m {
                return Err(CollectionError::InvalidConfiguration {
                    message: "HNSW ef_construction must be >= M".to_string(),
                });
            }

            if hnsw_config.ef_search == 0 {
                return Err(CollectionError::InvalidConfiguration {
                    message: "HNSW ef_search must be greater than 0".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Create a default configuration for the given dimensions
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            dimensions,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: Some(HnswConfig::default()),
            quantization: Some(QuantizationConfig::Scalar),
            on_disk_payload: true,
        }
    }
}

/// A collection of vectors with its own configuration
pub struct Collection {
    /// Collection name
    pub name: String,

    /// Collection configuration
    pub config: CollectionConfig,

    /// Underlying vector database
    pub db: VectorDB,

    /// When the collection was created (Unix timestamp in seconds)
    pub created_at: i64,

    /// When the collection was last updated (Unix timestamp in seconds)
    pub updated_at: i64,
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("config", &self.config)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .field("db", &"<VectorDB>")
            .finish()
    }
}

impl Collection {
    /// Create a new collection
    pub fn new(name: String, config: CollectionConfig, storage_path: String) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Create VectorDB with the configuration
        let db_options = ruvector_core::types::DbOptions {
            dimensions: config.dimensions,
            distance_metric: config.distance_metric,
            storage_path,
            hnsw_config: config.hnsw_config.clone(),
            quantization: config.quantization.clone(),
        };

        let db = VectorDB::new(db_options)?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Ok(Self {
            name,
            config,
            db,
            created_at: now,
            updated_at: now,
        })
    }

    /// Get collection statistics
    pub fn stats(&self) -> Result<CollectionStats> {
        let vectors_count = self.db.len()?;

        Ok(CollectionStats {
            vectors_count,
            segments_count: 1,  // Single segment for now
            disk_size_bytes: 0, // TODO: Implement disk size calculation
            ram_size_bytes: 0,  // TODO: Implement RAM size calculation
        })
    }

    /// Update the last modified timestamp
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
    }
}

/// Statistics about a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    /// Number of vectors in the collection
    pub vectors_count: usize,

    /// Number of segments (partitions) in the collection
    pub segments_count: usize,

    /// Total disk space used (bytes)
    pub disk_size_bytes: u64,

    /// Total RAM used (bytes)
    pub ram_size_bytes: u64,
}

impl CollectionStats {
    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.vectors_count == 0
    }

    /// Get human-readable disk size
    pub fn disk_size_human(&self) -> String {
        format_bytes(self.disk_size_bytes)
    }

    /// Get human-readable RAM size
    pub fn ram_size_human(&self) -> String {
        format_bytes(self.ram_size_bytes)
    }
}

/// Format bytes into human-readable size
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_core::types::HnswConfig;

    // ===== CollectionConfig validation tests =====

    #[test]
    fn test_collection_config_validation() {
        // Valid config
        let config = CollectionConfig::with_dimensions(384);
        assert!(config.validate().is_ok());

        // Invalid: zero dimensions
        let config = CollectionConfig {
            dimensions: 0,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: true,
        };
        assert!(config.validate().is_err());

        // Invalid: dimensions too large
        let config = CollectionConfig {
            dimensions: 200_000,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: true,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_at_boundary_dimensions() {
        // Exactly 1 dimension -- minimum valid
        let config = CollectionConfig {
            dimensions: 1,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: false,
        };
        assert!(config.validate().is_ok());

        // Exactly 100_000 -- maximum valid
        let config = CollectionConfig {
            dimensions: 100_000,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: false,
        };
        assert!(config.validate().is_ok());

        // 100_001 -- just over the limit
        let config = CollectionConfig {
            dimensions: 100_001,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: false,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_hnsw_m_zero() {
        let config = CollectionConfig {
            dimensions: 128,
            distance_metric: DistanceMetric::Euclidean,
            hnsw_config: Some(HnswConfig {
                m: 0,
                ef_construction: 200,
                ef_search: 100,
                max_elements: 1000,
            }),
            quantization: None,
            on_disk_payload: false,
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("M parameter"));
    }

    #[test]
    fn test_config_validates_hnsw_ef_construction_less_than_m() {
        let config = CollectionConfig {
            dimensions: 128,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: Some(HnswConfig {
                m: 32,
                ef_construction: 16, // less than m
                ef_search: 100,
                max_elements: 1000,
            }),
            quantization: None,
            on_disk_payload: false,
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("ef_construction"));
    }

    #[test]
    fn test_config_validates_hnsw_ef_search_zero() {
        let config = CollectionConfig {
            dimensions: 128,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 0,
                max_elements: 1000,
            }),
            quantization: None,
            on_disk_payload: false,
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("ef_search"));
    }

    #[test]
    fn test_config_valid_hnsw_passes() {
        let config = CollectionConfig {
            dimensions: 64,
            distance_metric: DistanceMetric::DotProduct,
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 128,
                ef_search: 50,
                max_elements: 5000,
            }),
            quantization: None,
            on_disk_payload: true,
        };
        assert!(config.validate().is_ok());
    }

    // ===== CollectionConfig::with_dimensions tests =====

    #[test]
    fn test_with_dimensions_sets_fields() {
        let config = CollectionConfig::with_dimensions(256);
        assert_eq!(config.dimensions, 256);
        assert!(matches!(config.distance_metric, DistanceMetric::Cosine));
        assert!(config.hnsw_config.is_some());
        assert!(config.quantization.is_some());
        assert!(config.on_disk_payload);
    }

    // ===== CollectionConfig serde tests =====

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = CollectionConfig::with_dimensions(384);
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: CollectionConfig =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.dimensions, 384);
    }

    // ===== Collection creation tests =====

    #[test]
    fn test_collection_new_with_valid_config() {
        let temp = std::env::temp_dir().join("ruvector_test_coll_new_valid");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let db_path = temp.join("vectors.db").to_string_lossy().to_string();
        let config = CollectionConfig::with_dimensions(64);
        let coll = Collection::new("test_coll".to_string(), config, db_path);
        assert!(coll.is_ok());

        let coll = coll.unwrap();
        assert_eq!(coll.name, "test_coll");
        assert_eq!(coll.config.dimensions, 64);
        assert!(coll.created_at > 0);
        assert_eq!(coll.created_at, coll.updated_at);

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_collection_new_rejects_zero_dimensions() {
        let temp = std::env::temp_dir().join("ruvector_test_coll_new_zero");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let db_path = temp.join("vectors.db").to_string_lossy().to_string();
        let config = CollectionConfig {
            dimensions: 0,
            distance_metric: DistanceMetric::Cosine,
            hnsw_config: None,
            quantization: None,
            on_disk_payload: false,
        };
        let result = Collection::new("bad".to_string(), config, db_path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&temp);
    }

    // ===== Collection stats tests =====

    #[test]
    fn test_collection_stats_on_empty() {
        let temp = std::env::temp_dir().join("ruvector_test_coll_stats_empty");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let db_path = temp.join("vectors.db").to_string_lossy().to_string();
        let config = CollectionConfig::with_dimensions(32);
        let coll = Collection::new("stats_test".to_string(), config, db_path).unwrap();

        let stats = coll.stats().unwrap();
        assert_eq!(stats.vectors_count, 0);
        assert!(stats.is_empty());

        let _ = std::fs::remove_dir_all(&temp);
    }

    // ===== Collection touch tests =====

    #[test]
    fn test_collection_touch_updates_timestamp() {
        let temp = std::env::temp_dir().join("ruvector_test_coll_touch");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let db_path = temp.join("vectors.db").to_string_lossy().to_string();
        let config = CollectionConfig::with_dimensions(32);
        let mut coll = Collection::new("touch_test".to_string(), config, db_path).unwrap();

        let before = coll.updated_at;
        // Touch with a small pause to ensure timestamp can differ
        coll.touch();
        assert!(coll.updated_at >= before);

        let _ = std::fs::remove_dir_all(&temp);
    }

    // ===== Collection Debug impl test =====

    #[test]
    fn test_collection_debug_format() {
        let temp = std::env::temp_dir().join("ruvector_test_coll_debug");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let db_path = temp.join("vectors.db").to_string_lossy().to_string();
        let config = CollectionConfig::with_dimensions(16);
        let coll = Collection::new("debug_test".to_string(), config, db_path).unwrap();

        let debug_str = format!("{:?}", coll);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("<VectorDB>"));

        let _ = std::fs::remove_dir_all(&temp);
    }

    // ===== CollectionStats tests =====

    #[test]
    fn test_collection_stats_is_empty() {
        let stats = CollectionStats {
            vectors_count: 0,
            segments_count: 1,
            disk_size_bytes: 0,
            ram_size_bytes: 0,
        };
        assert!(stats.is_empty());

        let stats = CollectionStats {
            vectors_count: 5,
            segments_count: 1,
            disk_size_bytes: 1024,
            ram_size_bytes: 512,
        };
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_collection_stats_human_readable_sizes() {
        let stats = CollectionStats {
            vectors_count: 100,
            segments_count: 1,
            disk_size_bytes: 1048576, // 1 MB
            ram_size_bytes: 2048,     // 2 KB
        };
        assert_eq!(stats.disk_size_human(), "1.00 MB");
        assert_eq!(stats.ram_size_human(), "2.00 KB");
    }

    #[test]
    fn test_collection_stats_zero_bytes_human() {
        let stats = CollectionStats {
            vectors_count: 0,
            segments_count: 0,
            disk_size_bytes: 0,
            ram_size_bytes: 0,
        };
        assert_eq!(stats.disk_size_human(), "0 B");
        assert_eq!(stats.ram_size_human(), "0 B");
    }

    #[test]
    fn test_collection_stats_serde_roundtrip() {
        let stats = CollectionStats {
            vectors_count: 42,
            segments_count: 3,
            disk_size_bytes: 999,
            ram_size_bytes: 888,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: CollectionStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.vectors_count, 42);
        assert_eq!(deserialized.segments_count, 3);
        assert_eq!(deserialized.disk_size_bytes, 999);
        assert_eq!(deserialized.ram_size_bytes, 888);
    }

    // ===== format_bytes tests =====

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_terabyte() {
        assert_eq!(format_bytes(1099511627776), "1.00 TB");
    }

    #[test]
    fn test_format_bytes_small_values() {
        assert_eq!(format_bytes(1), "1.00 B");
        assert_eq!(format_bytes(1023), "1023.00 B");
    }

    // ===== All distance metrics with valid config =====

    #[test]
    fn test_config_all_distance_metrics_validate() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let config = CollectionConfig {
                dimensions: 128,
                distance_metric: metric,
                hnsw_config: None,
                quantization: None,
                on_disk_payload: false,
            };
            assert!(config.validate().is_ok(), "Failed for metric {:?}", metric);
        }
    }
}
