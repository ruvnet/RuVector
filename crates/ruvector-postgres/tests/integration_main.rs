//! Integration Test Entry Point for RuVector Postgres v2
//!
//! This file serves as the main entry point for Docker-based integration tests.
//! Tests are organized into modules that correspond to test categories.
//!
//! # Running Tests
//!
//! ## Using Docker (Recommended)
//!
//! ```bash
//! cd crates/ruvector-postgres
//! ./docker/run-integration-tests.sh
//! ```
//!
//! ## Using Cargo Directly
//!
//! Requires a running PostgreSQL instance with RuVector extension:
//!
//! ```bash
//! export DATABASE_URL="postgresql://ruvector:ruvector@localhost:5432/ruvector_test"
//! cargo test --test integration --features pg17
//! ```
//!
//! ## Running Specific Categories
//!
//! ```bash
//! # pgvector compatibility
//! cargo test --test integration pgvector_compat
//!
//! # Performance tests
//! cargo test --test integration perf_tests
//!
//! # Integrity system
//! cargo test --test integration integrity_tests
//! ```
//!
//! # Test Categories
//!
//! | Category | Description |
//! |----------|-------------|
//! | `pgvector_compat` | pgvector SQL syntax compatibility |
//! | `integrity_tests` | Contracted graph and integrity monitoring |
//! | `hybrid_search_tests` | BM25 + vector hybrid search |
//! | `tenancy_tests` | Multi-tenant isolation and RLS |
//! | `healing_tests` | Self-healing and recovery |
//! | `perf_tests` | Performance benchmarks |

// Include all test modules
mod integration;

// Re-export test modules for cargo test filtering
pub use integration::harness;
pub use integration::healing_tests;
pub use integration::hybrid_search_tests;
pub use integration::integrity_tests;
pub use integration::perf_tests;
pub use integration::pgvector_compat;
pub use integration::tenancy_tests;

#[cfg(test)]
mod integration_entry {
    use super::*;

    /// Verify test harness is working
    #[test]
    fn test_harness_config() {
        let config = harness::TestConfig::default();

        assert!(!config.host.is_empty());
        assert!(config.port > 0);
        assert!(!config.user.is_empty());
        assert!(!config.database.is_empty());
    }

    /// Verify test context creation
    #[test]
    fn test_context_creation() {
        let ctx = harness::TestContext::new("test_example");

        assert!(ctx.schema_name.starts_with("test_"));
        assert!(ctx.init_sql().contains("CREATE SCHEMA"));
        assert!(ctx.cleanup_sql().contains("DROP SCHEMA"));
    }

    /// Verify vector generation utilities
    #[test]
    fn test_vector_generation() {
        let vectors = harness::generate_random_vectors(100, 128);

        assert_eq!(vectors.len(), 100);
        for v in &vectors {
            assert_eq!(v.len(), 128);
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }

    /// Verify normalized vector generation
    #[test]
    fn test_normalized_vector_generation() {
        let vectors = harness::generate_normalized_vectors(50, 64);

        assert_eq!(vectors.len(), 50);
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5 || norm == 0.0,
                "Vector should be normalized"
            );
        }
    }

    /// Verify SQL helpers
    #[test]
    fn test_sql_helpers() {
        let schema = "test_schema";
        let table = "test_table";

        let create_sql = harness::sql::create_vector_table(schema, table, 128);
        assert!(create_sql.contains("CREATE TABLE"));
        assert!(create_sql.contains("vector(128)"));

        let hnsw_sql = harness::sql::create_hnsw_index(schema, table, 16, 64);
        assert!(hnsw_sql.contains("USING hnsw"));
        assert!(hnsw_sql.contains("m = 16"));

        let ivfflat_sql = harness::sql::create_ivfflat_index(schema, table, 100);
        assert!(ivfflat_sql.contains("USING ivfflat"));
        assert!(ivfflat_sql.contains("lists = 100"));
    }

    /// Verify latency statistics calculation
    #[test]
    fn test_latency_stats() {
        let mut measurements: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        let stats = harness::LatencyStats::from_measurements(&mut measurements);

        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
        assert!((stats.mean - 50.5).abs() < 0.1);
        assert!((stats.p50 - 50.0).abs() < 1.0);
        assert!(stats.p95 >= 95.0);
        assert!(stats.p99 >= 99.0);
    }

    /// Verify assertion helpers
    #[test]
    fn test_assertion_helpers() {
        // approx_eq
        harness::assertions::assert_approx_eq(1.0001, 1.0, 0.001);

        // recall
        harness::assertions::assert_recall_above(0.95, 0.9);

        // precision
        harness::assertions::assert_precision_above(0.88, 0.85);
    }

    /// Verify percentile calculation
    #[test]
    fn test_percentile() {
        let mut values: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        assert_eq!(harness::percentile(&mut values, 0.0), 1.0);
        assert!((harness::percentile(&mut values, 50.0) - 50.0).abs() < 1.0);
        assert_eq!(harness::percentile(&mut values, 100.0), 100.0);
    }

    /// Verify PostgreSQL array formatting
    #[test]
    fn test_pg_array_formatting() {
        let v = vec![1.0, 2.0, 3.0];

        let pg_vector = harness::vec_to_pg_array(&v);
        assert_eq!(pg_vector, "[1.000000,2.000000,3.000000]");

        let pg_array = harness::vec_to_pg_real_array(&v);
        assert!(pg_array.starts_with("ARRAY["));
        assert!(pg_array.ends_with("::real[]"));
    }
}
