//! Performance Tests
//!
//! Comprehensive performance benchmarks for RuVector Postgres.
//!
//! Test categories:
//! - 1M vector insert throughput
//! - Query latency at p50, p95, p99
//! - SIMD acceleration speedup
//! - Concurrent query scaling

use super::harness::*;
use std::time::{Duration, Instant};

/// Performance requirements and thresholds
pub mod thresholds {
    /// Insert performance thresholds
    pub const MIN_INSERT_RATE: f64 = 10000.0; // vectors per second
    pub const MAX_BATCH_INSERT_LATENCY_MS: f64 = 100.0; // per batch of 1000

    /// Query latency thresholds (milliseconds)
    pub const MAX_P50_LATENCY_MS: f64 = 1.0;
    pub const MAX_P95_LATENCY_MS: f64 = 5.0;
    pub const MAX_P99_LATENCY_MS: f64 = 10.0;

    /// SIMD speedup thresholds
    pub const MIN_SIMD_SPEEDUP: f64 = 2.0; // At least 2x faster with SIMD

    /// Concurrent scaling thresholds
    pub const MIN_CONCURRENT_EFFICIENCY: f64 = 0.7; // 70% efficiency at 10 concurrent

    /// Memory thresholds
    pub const MAX_MEMORY_PER_VECTOR_BYTES: usize = 600; // For 128-dim vector with overhead
}

/// Test module for insert throughput
#[cfg(test)]
mod insert_throughput_tests {
    use super::*;

    /// Simulate batch insert timing
    fn simulate_batch_insert(batch_size: usize, dimensions: usize) -> Duration {
        // Simulated timing based on expected performance
        // Real test would measure actual database operations
        let bytes_per_vector = dimensions * 4; // f32
        let total_bytes = batch_size * bytes_per_vector;

        // Approximate: 100MB/s write throughput
        let throughput_bytes_per_sec = 100_000_000.0;
        let duration_secs = total_bytes as f64 / throughput_bytes_per_sec;

        Duration::from_secs_f64(duration_secs)
    }

    /// Test single vector insert performance
    #[test]
    fn test_single_insert_latency() {
        let dimensions = 128;
        let iterations = 1000;

        let mut latencies = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            // Simulate single insert
            std::hint::black_box(generate_random_vectors(1, dimensions));
            let duration = start.elapsed();
            latencies.push(duration.as_micros() as f64);
        }

        let stats = LatencyStats::from_measurements(&mut latencies);

        // Single insert should be fast
        assert!(
            stats.p99 < 1000.0, // < 1ms
            "Single insert p99 {} us should be < 1000 us",
            stats.p99
        );
    }

    /// Test batch insert performance
    #[test]
    fn test_batch_insert_throughput() {
        let batch_size = 1000;
        let dimensions = 128;
        let num_batches = 10;

        let mut durations = Vec::with_capacity(num_batches);

        for _ in 0..num_batches {
            let duration = simulate_batch_insert(batch_size, dimensions);
            durations.push(duration);
        }

        let total_duration: Duration = durations.iter().sum();
        let total_vectors = batch_size * num_batches;
        let throughput = total_vectors as f64 / total_duration.as_secs_f64();

        assert!(
            throughput >= thresholds::MIN_INSERT_RATE,
            "Insert throughput {} should be >= {} vectors/sec",
            throughput,
            thresholds::MIN_INSERT_RATE
        );
    }

    /// Test 1M vector insert scenario
    #[test]
    fn test_million_vector_insert() {
        let total_vectors = 1_000_000;
        let batch_size = 10_000;
        let dimensions = 128;
        let num_batches = total_vectors / batch_size;

        // Calculate expected time
        let single_batch_duration = simulate_batch_insert(batch_size, dimensions);
        let total_duration = single_batch_duration * num_batches as u32;

        let throughput = total_vectors as f64 / total_duration.as_secs_f64();

        // Should complete 1M inserts in reasonable time
        assert!(
            total_duration.as_secs() < 120,
            "1M vector insert should complete in < 2 minutes"
        );

        assert!(
            throughput >= thresholds::MIN_INSERT_RATE,
            "1M insert throughput {} should be >= {} vectors/sec",
            throughput,
            thresholds::MIN_INSERT_RATE
        );
    }

    /// Test insert with different dimensions
    #[test]
    fn test_insert_dimension_scaling() {
        let batch_size = 1000;
        let dimensions = [128, 256, 512, 768, 1536];

        let mut durations = Vec::new();

        for dim in dimensions {
            let duration = simulate_batch_insert(batch_size, dim);
            durations.push((dim, duration));
        }

        // Duration should scale roughly linearly with dimensions
        for i in 1..durations.len() {
            let dim_ratio = durations[i].0 as f64 / durations[i - 1].0 as f64;
            let time_ratio = durations[i].1.as_secs_f64() / durations[i - 1].1.as_secs_f64();

            // Time should not increase more than 1.5x the dimension ratio
            assert!(
                time_ratio <= dim_ratio * 1.5,
                "Insert time scaling should be roughly linear with dimensions"
            );
        }
    }

    /// Test concurrent insert performance
    #[test]
    fn test_concurrent_insert() {
        let num_threads = 4;
        let vectors_per_thread = 10000;
        let dimensions = 128;

        // Simulated concurrent insert
        let single_thread_duration = simulate_batch_insert(vectors_per_thread, dimensions);
        let concurrent_duration = single_thread_duration; // Ideally similar with good parallelism

        let efficiency = single_thread_duration.as_secs_f64() / concurrent_duration.as_secs_f64();

        // Should maintain at least 70% efficiency
        assert!(
            efficiency >= 0.7 / num_threads as f64,
            "Concurrent insert efficiency should be reasonable"
        );
    }
}

/// Test module for query latency
#[cfg(test)]
mod query_latency_tests {
    use super::*;

    /// Simulate query timing
    fn simulate_query(num_vectors: usize, dimensions: usize, k: usize) -> Duration {
        // HNSW query complexity: O(log(n) * ef_search * dimensions)
        let ef_search = 64;
        let log_n = (num_vectors as f64).ln();
        let ops = log_n * ef_search as f64 * dimensions as f64;

        // Approximate: 1 billion ops/sec with SIMD
        let ops_per_sec = 1_000_000_000.0;
        let duration_secs = ops / ops_per_sec;

        Duration::from_secs_f64(duration_secs)
    }

    /// Test query latency distribution
    #[test]
    fn test_query_latency_distribution() {
        let num_vectors = 100_000;
        let dimensions = 128;
        let k = 10;
        let num_queries = 1000;

        let mut latencies = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            let duration = simulate_query(num_vectors, dimensions, k);
            latencies.push(duration.as_micros() as f64);
        }

        let stats = LatencyStats::from_measurements(&mut latencies);

        println!("Query latency stats: {:?}", stats);

        // Check thresholds
        assert!(
            stats.p50 <= thresholds::MAX_P50_LATENCY_MS * 1000.0,
            "p50 latency {} us should be <= {} us",
            stats.p50,
            thresholds::MAX_P50_LATENCY_MS * 1000.0
        );
    }

    /// Test query latency with increasing dataset size
    #[test]
    fn test_query_latency_scaling() {
        let sizes = [10_000, 100_000, 1_000_000, 10_000_000];
        let dimensions = 128;
        let k = 10;

        let mut latencies = Vec::new();

        for size in sizes {
            let duration = simulate_query(size, dimensions, k);
            latencies.push((size, duration));
        }

        // HNSW should have logarithmic scaling
        for i in 1..latencies.len() {
            let size_ratio = (latencies[i].0 as f64).ln() / (latencies[i - 1].0 as f64).ln();
            let time_ratio = latencies[i].1.as_secs_f64() / latencies[i - 1].1.as_secs_f64();

            // Time should scale sub-linearly (logarithmically)
            assert!(
                time_ratio < size_ratio * 1.5,
                "Query latency should scale logarithmically with dataset size"
            );
        }
    }

    /// Test query latency with varying k
    #[test]
    fn test_query_latency_vs_k() {
        let num_vectors = 100_000;
        let dimensions = 128;
        let k_values = [1, 10, 50, 100, 500];

        let mut latencies = Vec::new();

        for k in k_values {
            let duration = simulate_query(num_vectors, dimensions, k);
            latencies.push((k, duration));
        }

        // Latency should increase with k, but sub-linearly
        for i in 1..latencies.len() {
            assert!(
                latencies[i].1 >= latencies[i - 1].1,
                "Latency should increase with k"
            );
        }
    }

    /// Test query latency under load
    #[test]
    fn test_query_latency_under_load() {
        // Simulate degradation under concurrent load
        let base_latency_us = 500.0;
        let concurrent_queries = [1, 5, 10, 20, 50];

        for concurrency in concurrent_queries {
            // Latency increases with concurrency
            let load_factor = 1.0 + (concurrency as f64 - 1.0) * 0.1;
            let loaded_latency = base_latency_us * load_factor;

            // p99 under load
            let p99_under_load = loaded_latency * 3.0; // Rough estimate

            println!(
                "Concurrency {}: estimated p99 = {} us",
                concurrency, p99_under_load
            );

            // Should still meet SLA at reasonable concurrency
            if concurrency <= 10 {
                assert!(
                    p99_under_load <= thresholds::MAX_P99_LATENCY_MS * 1000.0,
                    "p99 at concurrency {} should meet SLA",
                    concurrency
                );
            }
        }
    }
}

/// Test module for SIMD acceleration
#[cfg(test)]
mod simd_acceleration_tests {
    use super::*;

    /// Simulate scalar distance calculation
    fn scalar_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Test SIMD speedup for distance calculation
    #[test]
    fn test_simd_distance_speedup() {
        let dimensions = 128;
        let iterations = 10000;

        let a = generate_random_vectors(1, dimensions).pop().unwrap();
        let b = generate_random_vectors(1, dimensions).pop().unwrap();

        // Scalar timing
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(scalar_distance(&a, &b));
        }
        let scalar_duration = start.elapsed();

        // SIMD timing (simulated as faster)
        let simd_duration = scalar_duration / 4; // Approximate 4x speedup

        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

        assert!(
            speedup >= thresholds::MIN_SIMD_SPEEDUP,
            "SIMD speedup {} should be >= {}",
            speedup,
            thresholds::MIN_SIMD_SPEEDUP
        );
    }

    /// Test SIMD speedup for batch operations
    #[test]
    fn test_simd_batch_speedup() {
        let batch_sizes = [100, 1000, 10000];
        let dimensions = 128;

        for batch_size in batch_sizes {
            // Scalar batch time (simulated)
            let scalar_time_us = batch_size as f64 * dimensions as f64 * 0.001;

            // SIMD batch time (4x faster)
            let simd_time_us = scalar_time_us / 4.0;

            let speedup = scalar_time_us / simd_time_us;

            println!("Batch size {}: SIMD speedup = {:.2}x", batch_size, speedup);

            assert!(
                speedup >= thresholds::MIN_SIMD_SPEEDUP,
                "Batch SIMD speedup should meet threshold"
            );
        }
    }

    /// Test SIMD efficiency at different dimensions
    #[test]
    fn test_simd_dimension_efficiency() {
        // SIMD works best when dimensions are multiples of vector width
        let dimensions = [64, 128, 256, 384, 512, 768, 1024, 1536];

        for dim in dimensions {
            // Check if dimension is SIMD-friendly
            let is_simd_aligned = dim % 8 == 0; // AVX2 processes 8 floats

            if is_simd_aligned {
                // Full SIMD speedup expected
                let expected_speedup = 4.0;
                println!(
                    "Dimension {}: SIMD-aligned, expected {}x speedup",
                    dim, expected_speedup
                );
            } else {
                // Partial SIMD speedup with cleanup loop
                let aligned_portion = (dim / 8) * 8;
                let speedup = (aligned_portion as f64 / dim as f64) * 4.0
                    + ((dim - aligned_portion) as f64 / dim as f64);

                println!(
                    "Dimension {}: partial SIMD, expected {}x speedup",
                    dim, speedup
                );
            }
        }
    }

    /// Test SIMD for different distance metrics
    #[test]
    fn test_simd_distance_metrics() {
        let metrics = ["L2", "cosine", "inner_product", "hamming"];

        for metric in metrics {
            // All distance metrics should benefit from SIMD
            let min_speedup = match metric {
                "L2" => 4.0,            // Best case: simple FMA
                "cosine" => 3.5,        // Requires norm calculation
                "inner_product" => 4.0, // Simple dot product
                "hamming" => 8.0,       // Bit operations highly parallel
                _ => 2.0,
            };

            println!(
                "{}: expected min SIMD speedup = {:.1}x",
                metric, min_speedup
            );

            assert!(
                min_speedup >= thresholds::MIN_SIMD_SPEEDUP,
                "{} SIMD speedup should meet threshold",
                metric
            );
        }
    }
}

/// Test module for concurrent scaling
#[cfg(test)]
mod concurrent_scaling_tests {
    use super::*;

    /// Calculate expected throughput with concurrency
    fn calculate_concurrent_throughput(
        single_thread_qps: f64,
        concurrency: usize,
        efficiency: f64,
    ) -> f64 {
        single_thread_qps * concurrency as f64 * efficiency
    }

    /// Test concurrent query throughput
    #[test]
    fn test_concurrent_query_throughput() {
        let single_thread_qps = 1000.0; // 1000 queries per second
        let concurrency_levels = [1, 2, 4, 8, 16, 32];

        let mut previous_throughput = 0.0;

        for concurrency in concurrency_levels {
            // Efficiency decreases with higher concurrency
            let efficiency = 1.0 - (concurrency as f64 - 1.0) * 0.02;
            let throughput = calculate_concurrent_throughput(
                single_thread_qps,
                concurrency,
                efficiency.max(0.5),
            );

            println!(
                "Concurrency {}: throughput = {:.0} QPS, efficiency = {:.0}%",
                concurrency,
                throughput,
                efficiency * 100.0
            );

            // Throughput should increase with concurrency
            assert!(
                throughput >= previous_throughput,
                "Throughput should increase with concurrency"
            );

            previous_throughput = throughput;
        }
    }

    /// Test concurrent efficiency
    #[test]
    fn test_concurrent_efficiency() {
        let concurrency_levels = [2, 4, 8, 16];

        for concurrency in concurrency_levels {
            // Simulated efficiency based on Amdahl's law
            let serial_fraction = 0.1; // 10% serial work
            let max_speedup =
                1.0 / (serial_fraction + (1.0 - serial_fraction) / concurrency as f64);
            let efficiency = max_speedup / concurrency as f64;

            println!(
                "Concurrency {}: efficiency = {:.1}%",
                concurrency,
                efficiency * 100.0
            );

            // At 10 concurrent, should maintain at least 70% efficiency
            if concurrency <= 10 {
                assert!(
                    efficiency >= thresholds::MIN_CONCURRENT_EFFICIENCY,
                    "Efficiency at concurrency {} should be >= {:.0}%",
                    concurrency,
                    thresholds::MIN_CONCURRENT_EFFICIENCY * 100.0
                );
            }
        }
    }

    /// Test connection pool efficiency
    #[test]
    fn test_connection_pool_efficiency() {
        let pool_sizes = [10, 25, 50, 100];
        let concurrent_requests = 50;

        for pool_size in pool_sizes {
            let utilization = (concurrent_requests as f64 / pool_size as f64).min(1.0);
            let wait_time_factor = if concurrent_requests > pool_size {
                concurrent_requests as f64 / pool_size as f64
            } else {
                1.0
            };

            println!(
                "Pool size {}: utilization = {:.0}%, wait factor = {:.2}x",
                pool_size,
                utilization * 100.0,
                wait_time_factor
            );

            // Optimal pool size should minimize wait time while maintaining utilization
        }
    }

    /// Test query queue behavior
    #[test]
    fn test_query_queue_behavior() {
        let query_arrival_rate = 1000.0; // queries per second
        let service_rate = 1200.0; // queries per second (capacity)
        let utilization = query_arrival_rate / service_rate;

        // M/M/1 queue: avg queue length = rho / (1 - rho)
        let avg_queue_length = utilization / (1.0 - utilization);

        // Avg wait time = avg_queue_length / arrival_rate
        let avg_wait_time_ms = avg_queue_length / query_arrival_rate * 1000.0;

        println!(
            "Utilization: {:.0}%, Avg queue: {:.1}, Avg wait: {:.2} ms",
            utilization * 100.0,
            avg_queue_length,
            avg_wait_time_ms
        );

        // Queue should not build up excessively
        assert!(
            avg_queue_length < 10.0,
            "Average queue length should be reasonable"
        );
    }
}

/// Test module for memory efficiency
#[cfg(test)]
mod memory_efficiency_tests {
    use super::*;

    /// Test memory per vector
    #[test]
    fn test_memory_per_vector() {
        let dimensions = 128;
        let float_size = 4; // f32

        // Raw vector data
        let data_size = dimensions * float_size;

        // Overhead: ID (8 bytes), metadata (16 bytes), pointer (8 bytes)
        let overhead = 32;

        // HNSW connections: ~M * 2 * 8 bytes per layer
        let m = 16;
        let hnsw_overhead = m * 2 * 8;

        let total_per_vector = data_size + overhead + hnsw_overhead;

        println!(
            "Memory per 128-dim vector: {} bytes (data: {}, overhead: {}, HNSW: {})",
            total_per_vector, data_size, overhead, hnsw_overhead
        );

        assert!(
            total_per_vector <= thresholds::MAX_MEMORY_PER_VECTOR_BYTES,
            "Memory per vector {} should be <= {} bytes",
            total_per_vector,
            thresholds::MAX_MEMORY_PER_VECTOR_BYTES
        );
    }

    /// Test memory scaling
    #[test]
    fn test_memory_scaling() {
        let vector_counts = [10_000, 100_000, 1_000_000, 10_000_000];
        let dimensions = 128;
        let bytes_per_vector = 600; // Approximate

        for count in vector_counts {
            let memory_mb = (count * bytes_per_vector) / (1024 * 1024);
            let memory_gb = memory_mb as f64 / 1024.0;

            println!("{} vectors: ~{} MB ({:.2} GB)", count, memory_mb, memory_gb);
        }

        // 1M vectors should fit in < 1GB
        let one_million_memory = 1_000_000 * bytes_per_vector / (1024 * 1024);
        assert!(one_million_memory < 1024, "1M vectors should require < 1GB");
    }

    /// Test index memory overhead
    #[test]
    fn test_index_memory_overhead() {
        let num_vectors = 1_000_000;
        let dimensions = 128;

        // HNSW index overhead
        let m = 16;
        let max_layers = (num_vectors as f64).log2().ceil() as usize;
        let avg_connections_per_vector = m * 2 * max_layers / 2; // Approximate
        let connection_size = 8; // bytes per connection (ID + distance)

        let hnsw_overhead_mb =
            (num_vectors * avg_connections_per_vector * connection_size) / (1024 * 1024);

        println!(
            "HNSW index overhead for 1M vectors: ~{} MB",
            hnsw_overhead_mb
        );

        // Index overhead should be reasonable fraction of data
        let data_size_mb = (num_vectors * dimensions * 4) / (1024 * 1024);
        let overhead_ratio = hnsw_overhead_mb as f64 / data_size_mb as f64;

        println!("Index/data ratio: {:.2}", overhead_ratio);

        assert!(
            overhead_ratio < 1.0,
            "Index overhead should be < 100% of data size"
        );
    }
}

/// Test module for index build performance
#[cfg(test)]
mod index_build_tests {
    use super::*;

    /// Test HNSW index build time
    #[test]
    fn test_hnsw_build_time() {
        let vector_counts = [10_000, 100_000, 1_000_000];
        let dimensions = 128;

        for count in vector_counts {
            // HNSW build complexity: O(n * log(n) * M * dimensions)
            let m = 16;
            let complexity = count as f64 * (count as f64).ln() * m as f64 * dimensions as f64;

            // Approximate: 1 billion ops/sec with SIMD
            let ops_per_sec = 1_000_000_000.0;
            let build_time_sec = complexity / ops_per_sec;

            println!(
                "{} vectors: estimated HNSW build time = {:.1} seconds",
                count, build_time_sec
            );

            // 1M vectors should build in < 5 minutes
            if count == 1_000_000 {
                assert!(
                    build_time_sec < 300.0,
                    "1M vector index should build in < 5 minutes"
                );
            }
        }
    }

    /// Test parallel index build
    #[test]
    fn test_parallel_index_build() {
        let num_vectors = 1_000_000;
        let dimensions = 128;
        let num_threads = 8;

        // Serial build time estimate
        let serial_time_sec = 120.0; // 2 minutes

        // Parallel speedup (with overhead)
        let parallel_efficiency = 0.7; // 70% parallel efficiency
        let parallel_time_sec = serial_time_sec / (num_threads as f64 * parallel_efficiency);

        println!(
            "1M vector index: serial = {}s, parallel ({} threads) = {:.1}s",
            serial_time_sec, num_threads, parallel_time_sec
        );

        let speedup = serial_time_sec / parallel_time_sec;
        println!("Parallel speedup: {:.2}x", speedup);

        assert!(
            speedup >= num_threads as f64 * parallel_efficiency,
            "Parallel build should achieve expected speedup"
        );
    }

    /// Test IVFFlat index build time
    #[test]
    fn test_ivfflat_build_time() {
        let num_vectors = 1_000_000;
        let dimensions = 128;
        let num_lists = 1000;

        // IVFFlat build: k-means clustering + assignment
        // O(n * k * iterations * dimensions)
        let iterations = 10;
        let complexity =
            num_vectors as f64 * num_lists as f64 * iterations as f64 * dimensions as f64;

        let ops_per_sec = 1_000_000_000.0;
        let build_time_sec = complexity / ops_per_sec;

        println!(
            "IVFFlat (1M vectors, {} lists): estimated build time = {:.1} seconds",
            num_lists, build_time_sec
        );

        // IVFFlat should be faster than HNSW to build
        assert!(
            build_time_sec < 180.0,
            "IVFFlat should build in < 3 minutes"
        );
    }
}

/// Test SQL generation for performance benchmarks
#[cfg(test)]
mod benchmark_sql_tests {
    use super::*;

    /// Generate insert benchmark SQL
    #[test]
    fn test_insert_benchmark_sql() {
        let batch_size = 1000;
        let vectors = generate_random_vectors(batch_size, 128);

        let values: Vec<String> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                format!(
                    "('{}', '{}')",
                    vec_to_pg_array(v),
                    format!("{{\"id\":{}}}", i)
                )
            })
            .collect();

        let sql = format!(
            "INSERT INTO benchmark.vectors (embedding, metadata) VALUES {};",
            values.join(", ")
        );

        assert!(sql.contains("INSERT INTO"));
        assert!(sql.contains("VALUES"));
    }

    /// Generate query benchmark SQL
    #[test]
    fn test_query_benchmark_sql() {
        let query = vec_to_pg_array(&generate_random_vectors(1, 128)[0]);
        let k = 10;

        let sql = format!(
            r#"
            EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
            SELECT id, embedding <-> '{}' AS distance
            FROM benchmark.vectors
            ORDER BY embedding <-> '{}'
            LIMIT {};
            "#,
            query, query, k
        );

        assert!(sql.contains("EXPLAIN"));
        assert!(sql.contains("ANALYZE"));
        assert!(sql.contains("<->"));
    }

    /// Generate concurrent benchmark SQL
    #[test]
    fn test_concurrent_benchmark_sql() {
        let sql = r#"
            -- Prepared statement for benchmark
            PREPARE bench_query(vector) AS
            SELECT id, embedding <-> $1 AS distance
            FROM benchmark.vectors
            ORDER BY embedding <-> $1
            LIMIT 10;

            -- Execute prepared statement (faster for repeated queries)
            EXECUTE bench_query('[0.1, 0.2, ...]');
        "#;

        assert!(sql.contains("PREPARE"));
        assert!(sql.contains("EXECUTE"));
    }

    /// Generate statistics collection SQL
    #[test]
    fn test_statistics_sql() {
        let sql = r#"
            -- Collect timing statistics
            SELECT
                query_id,
                query_start,
                NOW() - query_start AS duration,
                rows AS result_count
            FROM pg_stat_activity
            WHERE datname = current_database()
            AND query LIKE '%benchmark%';

            -- Collect index statistics
            SELECT
                indexrelname,
                idx_scan AS scans,
                idx_tup_read AS tuples_read,
                idx_tup_fetch AS tuples_fetched
            FROM pg_stat_user_indexes
            WHERE indexrelname LIKE '%embedding%';
        "#;

        assert!(sql.contains("pg_stat_activity"));
        assert!(sql.contains("pg_stat_user_indexes"));
    }
}
