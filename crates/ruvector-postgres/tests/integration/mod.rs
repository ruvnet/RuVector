//! Docker-based integration tests for RuVector Postgres v2
//!
//! These tests require a running PostgreSQL instance with the RuVector extension.
//! Use the Docker Compose setup in the `docker/` directory to run these tests.
//!
//! # Test Categories
//!
//! - `pgvector_compat`: pgvector SQL syntax compatibility
//! - `integrity_tests`: Contracted graph and integrity system
//! - `hybrid_search_tests`: BM25, RRF, and fusion search
//! - `tenancy_tests`: Multi-tenancy and RLS isolation
//! - `healing_tests`: Self-healing and recovery
//! - `perf_tests`: Performance benchmarks
//!
//! # Running Tests
//!
//! ```bash
//! # Start Docker environment
//! cd docker && docker-compose up -d
//!
//! # Run all integration tests
//! cargo test --test integration --features pg_test
//!
//! # Run specific test category
//! cargo test --test integration pgvector_compat --features pg_test
//! ```

pub mod harness;
pub mod healing_tests;
pub mod hybrid_search_tests;
pub mod integrity_tests;
pub mod perf_tests;
pub mod pgvector_compat;
pub mod tenancy_tests;
