//! Stress tests for concurrent operations and memory pressure
//!
//! These tests verify that the extension handles:
//! - Concurrent insertions and queries
//! - High memory pressure
//! - Large batches of operations
//! - Thread safety and race conditions

use ruvector_postgres::types::RuVector;
use std::sync::{Arc, Barrier};
use std::thread;

#[cfg(test)]
mod stress_tests {
    use super::*;

    // ========================================================================
    // Concurrent Operations Tests
    // ========================================================================

    #[test]
    fn test_concurrent_vector_creation() {
        let num_threads = 8;
        let vectors_per_thread = 100;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();

                    for i in 0..vectors_per_thread {
                        let data: Vec<f32> = (0..128)
                            .map(|j| ((thread_id * 1000 + i * 10 + j) as f32) * 0.01)
                            .collect();

                        let v = RuVector::from_slice(&data);
                        assert_eq!(v.dimensions(), 128);
                        assert_eq!(v.as_slice().len(), 128);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_concurrent_distance_calculations() {
        let num_threads = 16;
        let calculations_per_thread = 1000;

        // Prepare shared test vectors
        let v1 = Arc::new(RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        let v2 = Arc::new(RuVector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let v1 = Arc::clone(&v1);
                let v2 = Arc::clone(&v2);

                thread::spawn(move || {
                    for _ in 0..calculations_per_thread {
                        let norm1 = v1.norm();
                        let norm2 = v2.norm();
                        let dot = v1.dot(&*v2);

                        assert!(norm1.is_finite());
                        assert!(norm2.is_finite());
                        assert!(dot.is_finite());
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_concurrent_normalization() {
        let num_threads = 8;
        let operations_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        let data: Vec<f32> = (0..64)
                            .map(|j| ((thread_id * 100 + i + j) as f32) * 0.1)
                            .collect();

                        let v = RuVector::from_slice(&data);
                        let normalized = v.normalize();

                        let norm = normalized.norm();
                        if !data.iter().all(|&x| x == 0.0) {
                            assert!(
                                (norm - 1.0).abs() < 1e-5,
                                "Normalized vector should have unit norm"
                            );
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    // ========================================================================
    // Memory Pressure Tests
    // ========================================================================

    #[test]
    fn test_large_batch_allocation() {
        let num_vectors = 10_000;
        let dimensions = 128;

        let mut vectors = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let data: Vec<f32> = (0..dimensions)
                .map(|j| ((i * dimensions + j) as f32) * 0.001)
                .collect();

            vectors.push(RuVector::from_slice(&data));
        }

        // Verify all vectors are intact
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(v.dimensions(), dimensions);
            assert!(v.as_slice()[0] == (i * dimensions) as f32 * 0.001 || v.as_slice()[0] == 0.0);
        }
    }

    #[test]
    fn test_large_vector_dimensions() {
        // Test with maximum supported dimensions
        let max_dims = 10_000;

        let data: Vec<f32> = (0..max_dims).map(|i| (i as f32) * 0.0001).collect();

        let v = RuVector::from_slice(&data);
        assert_eq!(v.dimensions(), max_dims);

        let norm = v.norm();
        assert!(norm.is_finite() && norm > 0.0);
    }

    #[test]
    fn test_memory_reuse_pattern() {
        // Simulate a pattern of allocation and deallocation
        let iterations = 1000;
        let dimensions = 256;

        for _ in 0..iterations {
            let data: Vec<f32> = (0..dimensions).map(|i| i as f32).collect();
            let v = RuVector::from_slice(&data);

            assert_eq!(v.dimensions(), dimensions);

            // Do some operations
            let _ = v.norm();
            let _ = v.normalize();

            // Vector drops here, memory should be freed
        }
    }

    #[test]
    fn test_concurrent_allocation_deallocation() {
        let num_threads = 8;
        let iterations_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                thread::spawn(move || {
                    for _ in 0..iterations_per_thread {
                        let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
                        let v = RuVector::from_slice(&data);

                        // Perform operations
                        let _ = v.norm();
                        let _ = v.add(&v);
                        let _ = v.normalize();

                        // Implicit drop here
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    // ========================================================================
    // Batch Operations Tests
    // ========================================================================

    #[test]
    fn test_batch_distance_calculations() {
        let query = RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let num_candidates = 10_000;

        let candidates: Vec<_> = (0..num_candidates)
            .map(|i| {
                let data: Vec<f32> = (0..5).map(|j| ((i * 5 + j) as f32) * 0.01).collect();
                RuVector::from_slice(&data)
            })
            .collect();

        let distances: Vec<_> = candidates
            .iter()
            .map(|c| {
                use ruvector_postgres::distance::euclidean_distance;
                euclidean_distance(query.as_slice(), c.as_slice())
            })
            .collect();

        assert_eq!(distances.len(), num_candidates);
        assert!(distances.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn test_batch_normalization() {
        let num_vectors = 5000;
        let dimensions = 64;

        let vectors: Vec<_> = (0..num_vectors)
            .map(|i| {
                let data: Vec<f32> = (0..dimensions).map(|j| ((i + j) as f32) * 0.1).collect();
                RuVector::from_slice(&data)
            })
            .collect();

        let normalized: Vec<_> = vectors.iter().map(|v| v.normalize()).collect();

        for n in &normalized {
            let norm = n.norm();
            assert!((norm - 1.0).abs() < 1e-4 || n.as_slice().iter().all(|&x| x == 0.0));
        }
    }

    // ========================================================================
    // Stress Tests with Random Data
    // ========================================================================

    #[test]
    fn test_random_operations_single_threaded() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let dim = rng.gen_range(1..256);
            let data1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
            let data2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();

            let v1 = RuVector::from_slice(&data1);
            let v2 = RuVector::from_slice(&data2);

            // Random operations
            let _ = v1.add(&v2);
            let _ = v1.sub(&v2);
            let _ = v1.dot(&v2);
            let _ = v1.norm();
            let _ = v1.normalize();

            use ruvector_postgres::distance::{
                cosine_distance, euclidean_distance, manhattan_distance,
            };

            let d1 = euclidean_distance(&data1, &data2);
            let d2 = manhattan_distance(&data1, &data2);

            assert!(d1.is_finite());
            assert!(d2.is_finite());

            if data1.iter().any(|&x| x != 0.0) && data2.iter().any(|&x| x != 0.0) {
                let d3 = cosine_distance(&data1, &data2);
                assert!(d3.is_finite());
            }
        }
    }

    #[test]
    fn test_extreme_values_handling() {
        // Test with very small values
        let small = RuVector::from_slice(&[1e-10, 1e-10, 1e-10]);
        assert!(small.norm().is_finite());

        // Test with large values
        let large = RuVector::from_slice(&[1e6, 1e6, 1e6]);
        assert!(large.norm().is_finite());

        // Test with mixed scales
        let mixed = RuVector::from_slice(&[1e-10, 1.0, 1e10]);
        assert!(mixed.norm().is_finite());

        // Operations should not overflow/underflow
        let result = small.add(&large);
        assert!(result.as_slice().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_alternating_pattern_stress() {
        // Create a pattern that might trigger SIMD edge cases
        for size in [63, 64, 65, 127, 128, 129, 255, 256, 257] {
            let data: Vec<f32> = (0..size)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();

            let v = RuVector::from_slice(&data);
            let norm = v.norm();

            let expected = (size as f32).sqrt();
            assert!(
                (norm - expected).abs() < 0.01,
                "Size {}: expected {}, got {}",
                size,
                expected,
                norm
            );
        }
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_shared_vector_read_only() {
        let v = Arc::new(RuVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        let num_threads = 16;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let v = Arc::clone(&v);

                thread::spawn(move || {
                    for _ in 0..10000 {
                        assert_eq!(v.dimensions(), 5);
                        let _ = v.norm();
                        let _ = v.as_slice();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    // Note: test_varlena_roundtrip_stress removed - requires PostgreSQL runtime (pgrx)
    // Use `cargo pgrx test` to run varlena-related tests
}
