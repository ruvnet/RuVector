//! Test harness for Docker-based PostgreSQL integration tests
//!
//! Provides connection management, test utilities, and assertion helpers
//! for running tests against a live PostgreSQL instance with RuVector.

use std::env;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Database connection configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub database: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            host: env::var("POSTGRES_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: env::var("POSTGRES_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(5432),
            user: env::var("POSTGRES_USER").unwrap_or_else(|_| "ruvector".to_string()),
            password: env::var("POSTGRES_PASSWORD").unwrap_or_else(|_| "ruvector".to_string()),
            database: env::var("POSTGRES_DB").unwrap_or_else(|_| "ruvector_test".to_string()),
        }
    }
}

impl TestConfig {
    /// Get database URL for connection
    pub fn database_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, self.database
        )
    }

    /// Get connection URL from DATABASE_URL env var or use default
    pub fn from_env() -> Self {
        if let Ok(url) = env::var("DATABASE_URL") {
            Self::parse_url(&url).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    fn parse_url(url: &str) -> Option<Self> {
        // Parse postgresql://user:password@host:port/database
        let url = url.trim_start_matches("postgresql://");
        let url = url.trim_start_matches("postgres://");

        let (auth, rest) = url.split_once('@')?;
        let (user, password) = auth.split_once(':')?;
        let (host_port, database) = rest.split_once('/')?;
        let (host, port_str) = host_port.split_once(':').unwrap_or((host_port, "5432"));
        let port = port_str.parse().ok()?;

        Some(Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
            password: password.to_string(),
            database: database.to_string(),
        })
    }
}

/// Global test configuration singleton
static CONFIG: OnceLock<TestConfig> = OnceLock::new();

pub fn get_config() -> &'static TestConfig {
    CONFIG.get_or_init(TestConfig::from_env)
}

/// Test context for managing database connections and state
pub struct TestContext {
    pub config: TestConfig,
    pub schema_name: String,
    initialized: bool,
}

impl TestContext {
    /// Create a new test context with isolated schema
    pub fn new(test_name: &str) -> Self {
        let config = TestConfig::from_env();
        let schema_name = format!("test_{}", test_name.replace("::", "_").replace(" ", "_"));

        Self {
            config,
            schema_name,
            initialized: false,
        }
    }

    /// Get SQL to initialize the test schema
    pub fn init_sql(&self) -> String {
        format!(
            r#"
            DROP SCHEMA IF EXISTS {} CASCADE;
            CREATE SCHEMA {};
            SET search_path TO {}, public;
            "#,
            self.schema_name, self.schema_name, self.schema_name
        )
    }

    /// Get SQL to clean up the test schema
    pub fn cleanup_sql(&self) -> String {
        format!("DROP SCHEMA IF EXISTS {} CASCADE;", self.schema_name)
    }

    /// Connection string for this test context
    pub fn connection_string(&self) -> String {
        self.config.database_url()
    }
}

/// Timing utilities for performance tests
#[derive(Debug, Clone)]
pub struct TimingResult {
    pub operation: String,
    pub duration: Duration,
    pub iterations: usize,
}

impl TimingResult {
    pub fn new(operation: &str, duration: Duration, iterations: usize) -> Self {
        Self {
            operation: operation.to_string(),
            duration,
            iterations,
        }
    }

    /// Average duration per operation
    pub fn avg_duration(&self) -> Duration {
        self.duration / self.iterations as u32
    }

    /// Operations per second
    pub fn ops_per_sec(&self) -> f64 {
        self.iterations as f64 / self.duration.as_secs_f64()
    }

    /// Latency in microseconds
    pub fn latency_us(&self) -> f64 {
        self.duration.as_micros() as f64 / self.iterations as f64
    }
}

/// Timer for measuring operation durations
pub struct Timer {
    start: Instant,
    operation: String,
}

impl Timer {
    pub fn start(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
        }
    }

    pub fn stop(self, iterations: usize) -> TimingResult {
        TimingResult::new(&self.operation, self.start.elapsed(), iterations)
    }
}

/// Percentile calculation for latency analysis
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = (p / 100.0 * (values.len() - 1) as f64).round() as usize;
    values[index.min(values.len() - 1)]
}

/// Calculate statistics for a series of latency measurements
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl LatencyStats {
    pub fn from_measurements(measurements: &mut [f64]) -> Self {
        if measurements.is_empty() {
            return Self {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let count = measurements.len();
        let min = *measurements
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = *measurements
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mean = measurements.iter().sum::<f64>() / count as f64;

        Self {
            count,
            min,
            max,
            mean,
            p50: percentile(measurements, 50.0),
            p95: percentile(measurements, 95.0),
            p99: percentile(measurements, 99.0),
        }
    }
}

/// Generate random test vectors
pub fn generate_random_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Generate normalized random vectors
pub fn generate_normalized_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    generate_random_vectors(count, dimensions)
        .into_iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect()
}

/// Format vector as PostgreSQL array literal
pub fn vec_to_pg_array(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Format vector as PostgreSQL array literal for array type
pub fn vec_to_pg_real_array(v: &[f32]) -> String {
    format!(
        "ARRAY[{}]::real[]",
        v.iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Assertion helpers for test results
pub mod assertions {
    use super::*;

    /// Assert that two f32 values are approximately equal
    pub fn assert_approx_eq(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() < epsilon,
            "Expected {} to be approximately equal to {} (epsilon: {})",
            actual,
            expected,
            epsilon
        );
    }

    /// Assert that latency is within acceptable bounds
    pub fn assert_latency_within(stats: &LatencyStats, max_p99_us: f64) {
        assert!(
            stats.p99 <= max_p99_us,
            "p99 latency {} us exceeds maximum {} us",
            stats.p99,
            max_p99_us
        );
    }

    /// Assert that throughput meets minimum requirements
    pub fn assert_throughput_meets(result: &TimingResult, min_ops_per_sec: f64) {
        let actual = result.ops_per_sec();
        assert!(
            actual >= min_ops_per_sec,
            "Throughput {} ops/sec is below minimum {} ops/sec",
            actual,
            min_ops_per_sec
        );
    }

    /// Assert that recall is above threshold
    pub fn assert_recall_above(actual: f64, threshold: f64) {
        assert!(
            actual >= threshold,
            "Recall {} is below threshold {}",
            actual,
            threshold
        );
    }

    /// Assert that precision is above threshold
    pub fn assert_precision_above(actual: f64, threshold: f64) {
        assert!(
            actual >= threshold,
            "Precision {} is below threshold {}",
            actual,
            threshold
        );
    }
}

/// SQL query templates for common test operations
pub mod sql {
    /// Create a table with vector column
    pub fn create_vector_table(schema: &str, table: &str, dimensions: usize) -> String {
        format!(
            r#"
            CREATE TABLE {}.{} (
                id SERIAL PRIMARY KEY,
                embedding vector({}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            "#,
            schema, table, dimensions
        )
    }

    /// Create HNSW index on vector column
    pub fn create_hnsw_index(
        schema: &str,
        table: &str,
        m: usize,
        ef_construction: usize,
    ) -> String {
        format!(
            r#"
            CREATE INDEX ON {}.{} USING hnsw (embedding vector_l2_ops)
            WITH (m = {}, ef_construction = {});
            "#,
            schema, table, m, ef_construction
        )
    }

    /// Create IVFFlat index on vector column
    pub fn create_ivfflat_index(schema: &str, table: &str, lists: usize) -> String {
        format!(
            r#"
            CREATE INDEX ON {}.{} USING ivfflat (embedding vector_l2_ops)
            WITH (lists = {});
            "#,
            schema, table, lists
        )
    }

    /// Insert a vector with metadata
    pub fn insert_vector(schema: &str, table: &str, vector: &str, metadata: &str) -> String {
        format!(
            r#"
            INSERT INTO {}.{} (embedding, metadata)
            VALUES ('{}', '{}')
            RETURNING id;
            "#,
            schema, table, vector, metadata
        )
    }

    /// Batch insert vectors
    pub fn batch_insert_vectors(schema: &str, table: &str, vectors: &[String]) -> String {
        let values = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| format!("('{}', '{{\"idx\": {}}}')", v, i))
            .collect::<Vec<_>>()
            .join(",\n");

        format!(
            r#"
            INSERT INTO {}.{} (embedding, metadata)
            VALUES {};
            "#,
            schema, table, values
        )
    }

    /// Nearest neighbor search with L2 distance
    pub fn nn_search_l2(schema: &str, table: &str, query: &str, limit: usize) -> String {
        format!(
            r#"
            SELECT id, embedding <-> '{}' AS distance
            FROM {}.{}
            ORDER BY embedding <-> '{}'
            LIMIT {};
            "#,
            query, schema, table, query, limit
        )
    }

    /// Nearest neighbor search with cosine distance
    pub fn nn_search_cosine(schema: &str, table: &str, query: &str, limit: usize) -> String {
        format!(
            r#"
            SELECT id, embedding <=> '{}' AS distance
            FROM {}.{}
            ORDER BY embedding <=> '{}'
            LIMIT {};
            "#,
            query, schema, table, query, limit
        )
    }

    /// Nearest neighbor search with inner product
    pub fn nn_search_ip(schema: &str, table: &str, query: &str, limit: usize) -> String {
        format!(
            r#"
            SELECT id, embedding <#> '{}' AS distance
            FROM {}.{}
            ORDER BY embedding <#> '{}'
            LIMIT {};
            "#,
            query, schema, table, query, limit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let config = TestConfig::parse_url("postgresql://user:pass@localhost:5432/testdb").unwrap();
        assert_eq!(config.user, "user");
        assert_eq!(config.password, "pass");
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5432);
        assert_eq!(config.database, "testdb");
    }

    #[test]
    fn test_percentile_calculation() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&mut values, 50.0), 5.0);
        assert_eq!(percentile(&mut values, 0.0), 1.0);
        assert_eq!(percentile(&mut values, 100.0), 10.0);
    }

    #[test]
    fn test_vector_generation() {
        let vectors = generate_random_vectors(10, 128);
        assert_eq!(vectors.len(), 10);
        assert!(vectors.iter().all(|v| v.len() == 128));
    }

    #[test]
    fn test_normalized_vectors() {
        let vectors = generate_normalized_vectors(10, 128);
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
        }
    }

    #[test]
    fn test_vec_to_pg_array() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(vec_to_pg_array(&v), "[1.000000,2.000000,3.000000]");
    }
}
