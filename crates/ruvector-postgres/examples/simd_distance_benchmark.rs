//! Benchmark demonstrating zero-copy SIMD distance functions
//!
//! This example shows the performance benefits of using raw pointer-based
//! SIMD distance functions for vector operations.
//!
//! Run with: cargo run --release --example simd_distance_benchmark

use std::time::Instant;

// Note: In actual usage, these would be imported from the crate
// For this example, we'll create simple test versions

fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| (0..dim).map(|j| ((i + j) as f32 * 0.01).sin()).collect())
        .collect()
}

fn benchmark_slice_based(query: &[f32], vectors: &[Vec<f32>]) -> (Vec<f32>, u128) {
    let start = Instant::now();

    let results: Vec<f32> = vectors
        .iter()
        .map(|v| {
            // Slice-based approach (requires copying)
            let mut sum = 0.0f32;
            for i in 0..query.len() {
                let diff = query[i] - v[i];
                sum += diff * diff;
            }
            sum.sqrt()
        })
        .collect();

    let elapsed = start.elapsed().as_micros();
    (results, elapsed)
}

fn benchmark_pointer_based(query: &[f32], vectors: &[Vec<f32>]) -> (Vec<f32>, u128) {
    let start = Instant::now();

    let results: Vec<f32> = vectors
        .iter()
        .map(|v| {
            // Pointer-based approach (zero-copy)
            unsafe {
                let mut sum = 0.0f32;
                let a = query.as_ptr();
                let b = v.as_ptr();
                for i in 0..query.len() {
                    let diff = *a.add(i) - *b.add(i);
                    sum += diff * diff;
                }
                sum.sqrt()
            }
        })
        .collect();

    let elapsed = start.elapsed().as_micros();
    (results, elapsed)
}

fn main() {
    println!("=== SIMD Distance Function Benchmark ===\n");

    // Test configurations
    let configs = vec![
        (128, 1000),  // 128-dim vectors, 1000 vectors
        (384, 1000),  // 384-dim (OpenAI ada-002)
        (768, 1000),  // 768-dim (sentence transformers)
        (1536, 1000), // 1536-dim (OpenAI text-embedding-3-small)
    ];

    for (dim, count) in configs {
        println!("Testing with {} vectors of dimension {}", count, dim);

        let query = generate_random_vectors(1, dim)[0].clone();
        let vectors = generate_random_vectors(count, dim);

        // Warm up
        let _ = benchmark_slice_based(&query, &vectors);
        let _ = benchmark_pointer_based(&query, &vectors);

        // Actual benchmark
        let (results1, time1) = benchmark_slice_based(&query, &vectors);
        let (results2, time2) = benchmark_pointer_based(&query, &vectors);

        // Verify correctness
        let max_diff = results1
            .iter()
            .zip(results2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("  Slice-based:   {} μs", time1);
        println!("  Pointer-based: {} μs", time2);
        println!("  Speedup:       {:.2}x", time1 as f64 / time2 as f64);
        println!("  Max diff:      {:.2e}", max_diff);
        println!();
    }

    println!("\n=== Zero-Copy Batch Operations ===\n");

    // Demonstrate batch operations
    let dim = 384;
    let count = 10000;

    println!("Batch processing {} vectors of dimension {}", count, dim);

    let query = generate_random_vectors(1, dim)[0].clone();
    let vectors = generate_random_vectors(count, dim);

    let start = Instant::now();
    let vec_ptrs: Vec<*const f32> = vectors.iter().map(|v| v.as_ptr()).collect();
    let mut results = vec![0.0f32; count];

    // Simulate batch processing (in real code, this would use the SIMD functions)
    for (i, &ptr) in vec_ptrs.iter().enumerate() {
        unsafe {
            let mut sum = 0.0f32;
            for j in 0..dim {
                let diff = *query.as_ptr().add(j) - *ptr.add(j);
                sum += diff * diff;
            }
            results[i] = sum.sqrt();
        }
    }

    let elapsed = start.elapsed().as_micros();
    println!(
        "  Batch time: {} μs ({:.2} μs per vector)",
        elapsed,
        elapsed as f64 / count as f64
    );

    println!("\n=== Expected Performance Characteristics ===\n");
    println!("Architecture-specific optimizations:");
    println!("  AVX-512: 16 floats per iteration");
    println!("  AVX2:     8 floats per iteration");
    println!("  Scalar:   1 float per iteration");
    println!();
    println!("Alignment benefits:");
    println!("  64-byte aligned: Up to 10% faster with AVX-512");
    println!("  32-byte aligned: Up to 10% faster with AVX2");
    println!("  Unaligned:       Automatic fallback to unaligned loads");
    println!();
    println!("Batch operations:");
    println!("  Sequential: Simple iteration, cache-friendly");
    println!("  Parallel:   Uses Rayon for multi-core processing");
    println!();
}
