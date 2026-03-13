//! MoE Memory-Aware Routing Benchmarks (ADR-092)
//!
//! Criterion benchmarks for validating ADR-092 performance targets:
//!
//! - **Routing overhead**: Target <= 15 us (baseline ~5 us)
//! - **Affinity update**: EMA computation performance
//! - **Precision allocation**: Lookup and allocation performance
//! - **Cache hit rate simulation**: Compare baseline LRU vs affinity-aware
//! - **Paging simulation**: Expert paging latency
//!
//! Run with: `cargo bench --bench moe_bench`

#![allow(
    clippy::all,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_assignments,
    unexpected_cfgs,
    unused_must_use
)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvllm::bitnet::expert_cache::{
    align_to_cache_line, expert_memory_footprint, EvictionPolicy, ExpertBatch, ExpertCache,
    ExpertCacheConfig, MoeBatchScheduler, NullPrefetcher, Prefetcher,
};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Number of experts in Mixtral-style model
const NUM_EXPERTS: usize = 8;

/// Top-K experts per token
const TOP_K: usize = 2;

/// Hot-set size
const HOT_SET_SIZE: usize = 4;

/// Benchmark iterations for statistical significance
const BENCH_ITERS: usize = 10_000;

// ============================================================================
// Routing Overhead Benchmark
// ============================================================================

/// Benchmark: Compare standard vs memory-aware routing latency
///
/// Target: Memory-aware routing overhead <= 15 us (baseline ~5 us)
fn bench_routing_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_overhead");
    group.measurement_time(Duration::from_secs(5));

    // Baseline: Simple LRU cache access
    group.bench_function("baseline_lru", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.0, // No prefetch
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        let mut expert_id = 0;
        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;
            black_box(cache.access(expert_id))
        });
    });

    // Memory-aware: Adaptive eviction + prefetch checking
    group.bench_function("memory_aware_adaptive", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        let mut expert_id = 0;
        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;
            let hit = cache.access(expert_id);
            let should_prefetch = cache.should_prefetch((expert_id + 1) % NUM_EXPERTS, 0.15);
            black_box((hit, should_prefetch))
        });
    });

    // Full routing with prefetch admission
    group.bench_function("full_routing_with_prefetch", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        let mut expert_id = 0;
        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;
            let hit = cache.access(expert_id);
            let next_expert = (expert_id + 1) % NUM_EXPERTS;
            if cache.should_prefetch(next_expert, 0.15) {
                cache.prefetch_admit(next_expert);
            }
            black_box(hit)
        });
    });

    group.finish();
}

// ============================================================================
// Affinity Update Benchmark
// ============================================================================

/// Benchmark: EMA-based affinity score updates
fn bench_affinity_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("affinity_update");

    // Simulate EMA-based affinity tracking
    struct AffinityTracker {
        scores: Vec<f32>,
        decay: f32,
    }

    impl AffinityTracker {
        fn new(num_experts: usize, decay: f32) -> Self {
            Self {
                scores: vec![0.5; num_experts],
                decay,
            }
        }

        #[inline]
        fn activate(&mut self, expert_id: usize) {
            if expert_id < self.scores.len() {
                // EMA update: score = decay * score + (1 - decay) * 1.0
                self.scores[expert_id] = self.decay * self.scores[expert_id] + (1.0 - self.decay);
            }
        }

        #[inline]
        fn decay_step(&mut self, expert_id: usize) {
            if expert_id < self.scores.len() {
                self.scores[expert_id] *= self.decay;
            }
        }

        #[inline]
        fn decay_all(&mut self) {
            for score in &mut self.scores {
                *score *= self.decay;
            }
        }

        #[inline]
        fn get_top_k(&self, k: usize) -> Vec<usize> {
            let mut indexed: Vec<(usize, f32)> = self.scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
        }
    }

    // Single activation
    group.bench_function("single_activation", |b| {
        let mut tracker = AffinityTracker::new(NUM_EXPERTS, 0.9);
        let mut expert_id = 0;

        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;
            tracker.activate(expert_id);
            black_box(tracker.scores[expert_id])
        });
    });

    // Decay all experts
    group.bench_function("decay_all_experts", |b| {
        let mut tracker = AffinityTracker::new(NUM_EXPERTS, 0.9);

        b.iter(|| {
            tracker.decay_all();
            black_box(tracker.scores[0])
        });
    });

    // Get top-K by affinity
    group.bench_function("get_top_k", |b| {
        let mut tracker = AffinityTracker::new(NUM_EXPERTS, 0.9);

        // Set up varied affinities
        for i in 0..100 {
            tracker.activate(i % NUM_EXPERTS);
        }

        b.iter(|| black_box(tracker.get_top_k(HOT_SET_SIZE)));
    });

    // Combined: activate + decay + get_top_k (full routing step)
    group.bench_function("full_routing_step", |b| {
        let mut tracker = AffinityTracker::new(NUM_EXPERTS, 0.9);
        let mut step = 0;

        b.iter(|| {
            step += 1;
            let expert_id = step % NUM_EXPERTS;
            tracker.activate(expert_id);
            tracker.decay_all();
            let top_k = tracker.get_top_k(HOT_SET_SIZE);
            black_box(top_k)
        });
    });

    // Larger expert counts (Mixtral 8x22B has 8 experts, but future models may have more)
    for num_experts in [8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("decay_all_scaled", num_experts),
            &num_experts,
            |b, &n| {
                let mut tracker = AffinityTracker::new(n, 0.9);

                b.iter(|| {
                    tracker.decay_all();
                    black_box(tracker.scores[0])
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Precision Allocation Benchmark
// ============================================================================

/// Benchmark: Precision allocation lookup and decision
fn bench_precision_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_allocation");

    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(u8)]
    enum Precision {
        FP16 = 0,
        INT8 = 1,
        INT4 = 2,
    }

    // Simple lookup-based allocation
    group.bench_function("lookup_allocation", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up cache
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        let mut expert_id = 0;
        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;
            let precision = if cache.is_hot(expert_id) {
                Precision::FP16
            } else {
                Precision::INT4
            };
            black_box(precision)
        });
    });

    // Batch allocation for all experts
    group.bench_function("batch_allocation", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up cache
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        b.iter(|| {
            let allocations: Vec<Precision> = (0..NUM_EXPERTS)
                .map(|expert_id| {
                    if cache.is_hot(expert_id) {
                        Precision::FP16
                    } else {
                        Precision::INT4
                    }
                })
                .collect();
            black_box(allocations)
        });
    });

    // Memory budget calculation
    group.bench_function("memory_budget_check", |b| {
        let intermediate_size = 11008;
        let hidden_size = 4096;
        let block_size = 256;

        b.iter(|| {
            let expert_footprint =
                expert_memory_footprint(intermediate_size, hidden_size, block_size) * 3;
            let hot_set_budget = expert_footprint * HOT_SET_SIZE;
            black_box(hot_set_budget)
        });
    });

    group.finish();
}

// ============================================================================
// Cache Hit Rate Simulation Benchmark
// ============================================================================

/// Benchmark: Simulate workload and measure hit rate
fn bench_cache_hit_rate_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit_rate");
    group.measurement_time(Duration::from_secs(5));

    // Pre-generate routing decisions for consistent benchmarking
    let routing_decisions: Vec<Vec<(usize, f32)>> = (0..1000)
        .map(|token_idx| {
            let expert1 = (token_idx * 3 + token_idx / 10) % NUM_EXPERTS;
            let expert2 = (expert1 + 1 + token_idx % 3) % NUM_EXPERTS;
            vec![(expert1, 0.6), (expert2, 0.4)]
        })
        .collect();

    // Baseline LRU
    group.bench_function("baseline_lru", |b| {
        b.iter(|| {
            let config = ExpertCacheConfig {
                max_hot_experts: HOT_SET_SIZE,
                prefetch_threshold: 0.0,
                eviction_policy: EvictionPolicy::Lru,
            };
            let mut cache = ExpertCache::new(NUM_EXPERTS, config);

            for experts in &routing_decisions {
                for &(expert_id, _) in experts {
                    cache.access(expert_id);
                }
            }

            black_box(cache.stats().hit_rate())
        });
    });

    // LFU eviction
    group.bench_function("lfu_eviction", |b| {
        b.iter(|| {
            let config = ExpertCacheConfig {
                max_hot_experts: HOT_SET_SIZE,
                prefetch_threshold: 0.0,
                eviction_policy: EvictionPolicy::Lfu,
            };
            let mut cache = ExpertCache::new(NUM_EXPERTS, config);

            for experts in &routing_decisions {
                for &(expert_id, _) in experts {
                    cache.access(expert_id);
                }
            }

            black_box(cache.stats().hit_rate())
        });
    });

    // Adaptive eviction (memory-aware)
    group.bench_function("adaptive_eviction", |b| {
        b.iter(|| {
            let config = ExpertCacheConfig {
                max_hot_experts: HOT_SET_SIZE,
                prefetch_threshold: 0.1,
                eviction_policy: EvictionPolicy::Adaptive,
            };
            let mut cache = ExpertCache::new(NUM_EXPERTS, config);

            for experts in &routing_decisions {
                for &(expert_id, weight) in experts {
                    cache.access(expert_id);
                    // Also check prefetch candidates
                    if cache.should_prefetch((expert_id + 1) % NUM_EXPERTS, weight) {
                        cache.prefetch_admit((expert_id + 1) % NUM_EXPERTS);
                    }
                }
            }

            black_box(cache.stats().hit_rate())
        });
    });

    // Skewed workload (tests adaptive vs LRU more clearly)
    let skewed_routing: Vec<Vec<(usize, f32)>> = (0..1000)
        .map(|token_idx| {
            // 80% of accesses to experts 0, 1, 2
            let primary = if token_idx % 10 < 8 {
                token_idx % 3
            } else {
                3 + token_idx % (NUM_EXPERTS - 3)
            };
            let secondary = (primary + 1) % NUM_EXPERTS;
            vec![(primary, 0.7), (secondary, 0.3)]
        })
        .collect();

    group.bench_function("skewed_workload_lru", |b| {
        b.iter(|| {
            let config = ExpertCacheConfig {
                max_hot_experts: HOT_SET_SIZE,
                prefetch_threshold: 0.0,
                eviction_policy: EvictionPolicy::Lru,
            };
            let mut cache = ExpertCache::new(NUM_EXPERTS, config);

            for experts in &skewed_routing {
                for &(expert_id, _) in experts {
                    cache.access(expert_id);
                }
            }

            black_box(cache.stats().hit_rate())
        });
    });

    group.bench_function("skewed_workload_adaptive", |b| {
        b.iter(|| {
            let config = ExpertCacheConfig {
                max_hot_experts: HOT_SET_SIZE,
                prefetch_threshold: 0.1,
                eviction_policy: EvictionPolicy::Adaptive,
            };
            let mut cache = ExpertCache::new(NUM_EXPERTS, config);

            for experts in &skewed_routing {
                for &(expert_id, weight) in experts {
                    cache.access(expert_id);
                    if cache.should_prefetch((expert_id + 1) % NUM_EXPERTS, weight) {
                        cache.prefetch_admit((expert_id + 1) % NUM_EXPERTS);
                    }
                }
            }

            black_box(cache.stats().hit_rate())
        });
    });

    group.finish();
}

// ============================================================================
// Paging Simulation Benchmark
// ============================================================================

/// Benchmark: Expert paging latency simulation
fn bench_paging_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("paging_simulation");
    group.measurement_time(Duration::from_secs(3));

    // Simulate paging overhead based on expert memory footprint
    let intermediate_size = 11008;
    let hidden_size = 4096;
    let block_size = 256;
    let expert_size = expert_memory_footprint(intermediate_size, hidden_size, block_size) * 3;

    // Memory throughput assumptions (GB/s)
    // DDR5-4800: ~38 GB/s
    // Apple M4 unified memory: ~120 GB/s
    let memory_bandwidth_gbps = 120.0; // M4 assumption

    fn simulate_page_in(expert_size: usize, bandwidth_gbps: f64) -> Duration {
        let bytes = expert_size as f64;
        let gb = bytes / 1e9;
        let seconds = gb / bandwidth_gbps;
        Duration::from_secs_f64(seconds)
    }

    // Single expert page-in
    group.bench_function("single_expert_page_in", |b| {
        b.iter(|| {
            let latency = simulate_page_in(expert_size, memory_bandwidth_gbps);
            black_box(latency)
        });
    });

    // Batch expert page-in (amortized cost)
    for batch_size in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("batch_page_in", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let total_size = expert_size * size;
                    let latency = simulate_page_in(total_size, memory_bandwidth_gbps);
                    let per_expert = latency / size as u32;
                    black_box(per_expert)
                });
            },
        );
    }

    // Cache management overhead during paging
    group.bench_function("paging_with_cache_update", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        let mut expert_id = 0;
        b.iter(|| {
            expert_id = (expert_id + 1) % NUM_EXPERTS;

            // Check if page-in needed
            let needs_page_in = !cache.is_hot(expert_id);

            // Simulate page-in latency calculation
            let latency = if needs_page_in {
                simulate_page_in(expert_size, memory_bandwidth_gbps)
            } else {
                Duration::ZERO
            };

            // Update cache
            cache.access(expert_id);

            black_box(latency)
        });
    });

    group.finish();
}

// ============================================================================
// Batch Scheduler Benchmark
// ============================================================================

/// Benchmark: MoE batch scheduling performance
fn bench_batch_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scheduler");

    for batch_size in [1, 8, 32, 128, 512] {
        let routing_decisions: Vec<(usize, Vec<(usize, f32)>)> = (0..batch_size)
            .map(|token_idx| {
                let expert1 = (token_idx * 3) % NUM_EXPERTS;
                let expert2 = (expert1 + 1 + token_idx % 2) % NUM_EXPERTS;
                (token_idx, vec![(expert1, 0.6), (expert2, 0.4)])
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("schedule", batch_size),
            &routing_decisions,
            |b, routing| {
                b.iter(|| {
                    let batches = MoeBatchScheduler::schedule(routing);
                    black_box(batches)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Footprint Calculation Benchmark
// ============================================================================

/// Benchmark: Memory footprint calculations
fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");

    // Various model sizes
    let model_configs = [
        ("small", 4096, 2048, 256),
        ("medium", 8192, 4096, 256),
        ("mixtral", 11008, 4096, 256),
        ("large", 14336, 8192, 256),
    ];

    for (name, intermediate, hidden, block) in model_configs {
        group.bench_with_input(
            BenchmarkId::new("calculate", name),
            &(intermediate, hidden, block),
            |b, &(i, h, bs)| {
                b.iter(|| {
                    let gate = expert_memory_footprint(i, h, bs);
                    let up = expert_memory_footprint(i, h, bs);
                    let down = expert_memory_footprint(h, i, bs);
                    black_box(gate + up + down)
                });
            },
        );
    }

    // Cache line alignment
    group.bench_function("cache_line_align", |b| {
        let mut offset = 0usize;
        b.iter(|| {
            offset = (offset + 137) % 10000; // Varied offsets
            black_box(align_to_cache_line(offset))
        });
    });

    group.finish();
}

// ============================================================================
// Prefetch Decision Benchmark
// ============================================================================

/// Benchmark: Prefetch decision making
fn bench_prefetch_decision(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_decision");

    group.bench_function("should_prefetch", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Warm up
        for i in 0..HOT_SET_SIZE {
            cache.access(i);
        }

        let weights: Vec<f32> = (0..NUM_EXPERTS).map(|i| 0.05 + (i as f32) * 0.03).collect();

        let mut idx = 0;
        b.iter(|| {
            idx = (idx + 1) % NUM_EXPERTS;
            black_box(cache.should_prefetch(idx, weights[idx]))
        });
    });

    group.bench_function("prefetch_admit", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        let mut idx = 0;
        b.iter(|| {
            idx = (idx + 1) % NUM_EXPERTS;
            cache.prefetch_admit(idx);
            black_box(())
        });
    });

    group.bench_function("full_prefetch_cycle", |b| {
        let config = ExpertCacheConfig {
            max_hot_experts: HOT_SET_SIZE,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Adaptive,
        };
        let mut cache = ExpertCache::new(NUM_EXPERTS, config);

        // Simulate router weights
        let weights: Vec<f32> = vec![0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02];

        let mut token = 0;
        b.iter(|| {
            token += 1;

            // Current routing decision
            let top_2: Vec<usize> = (0..NUM_EXPERTS)
                .filter(|&i| weights[i] > 0.1)
                .take(2)
                .collect();

            // Access current experts
            for &expert_id in &top_2 {
                cache.access(expert_id);
            }

            // Check prefetch candidates
            for (expert_id, &weight) in weights.iter().enumerate() {
                if cache.should_prefetch(expert_id, weight) {
                    cache.prefetch_admit(expert_id);
                }
            }

            black_box(top_2)
        });
    });

    group.finish();
}

// ============================================================================
// Eviction Policy Comparison Benchmark
// ============================================================================

/// Benchmark: Compare eviction policies
fn bench_eviction_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction_policies");
    group.measurement_time(Duration::from_secs(3));

    let policies = [
        ("lru", EvictionPolicy::Lru),
        ("lfu", EvictionPolicy::Lfu),
        ("adaptive", EvictionPolicy::Adaptive),
    ];

    // Generate access pattern with locality
    let access_pattern: Vec<usize> = (0..1000)
        .map(|i| {
            // 70% local, 30% random
            if i % 10 < 7 {
                i % 3 // Local accesses to experts 0, 1, 2
            } else {
                (i * 7) % NUM_EXPERTS // Pseudo-random
            }
        })
        .collect();

    for (name, policy) in policies {
        group.bench_with_input(
            BenchmarkId::new("access_pattern", name),
            &policy,
            |b, &policy| {
                b.iter(|| {
                    let config = ExpertCacheConfig {
                        max_hot_experts: HOT_SET_SIZE,
                        prefetch_threshold: 0.1,
                        eviction_policy: policy,
                    };
                    let mut cache = ExpertCache::new(NUM_EXPERTS, config);

                    for &expert_id in &access_pattern {
                        cache.access(expert_id);
                    }

                    black_box(cache.stats().hit_rate())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory-Aware Router Benchmarks (P1-P4 Optimizations)
// ============================================================================

/// Benchmark: MemoryAwareRouter performance (ADR-092)
fn bench_memory_aware_router(c: &mut Criterion) {
    use ruvllm::moe::{AffinityConfig, ExpertAffinity, MemoryAwareRouter, RouterConfig};

    let mut group = c.benchmark_group("memory_aware_router");
    group.measurement_time(Duration::from_secs(5));

    // Test various expert counts
    for num_experts in [8, 16, 32, 64] {
        // P4: Top-2 unrolled optimization
        group.bench_with_input(
            BenchmarkId::new("route_top2", num_experts),
            &num_experts,
            |b, &n| {
                let config = RouterConfig::new(n, 2).with_cache_bonus(0.15);
                let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

                // Set half experts as resident
                let resident: Vec<usize> = (0..n / 2).collect();
                router.update_cache_state(&resident);

                // Generate gate logits
                let gate_logits: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.01).collect();

                b.iter(|| {
                    let (selected, _paging) = router.route(black_box(&gate_logits));
                    black_box(selected)
                });
            },
        );

        // P2: Batch routing optimization
        group.bench_with_input(
            BenchmarkId::new("route_batch_8", num_experts),
            &num_experts,
            |b, &n| {
                let config = RouterConfig::new(n, 2).with_cache_bonus(0.15);
                let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

                let resident: Vec<usize> = (0..n / 2).collect();
                router.update_cache_state(&resident);

                // Generate batch of 8 tokens
                let batch_logits: Vec<Vec<f32>> = (0..8)
                    .map(|t| (0..n).map(|i| 0.1 + (i as f32 + t as f32) * 0.01).collect())
                    .collect();
                let batch_refs: Vec<&[f32]> = batch_logits.iter().map(|v| v.as_slice()).collect();

                b.iter(|| {
                    let results = router.route_batch(black_box(&batch_refs));
                    black_box(results)
                });
            },
        );
    }

    // P1: Bitmask cache check overhead
    group.bench_function("cache_mask_check_64", |b| {
        let config = RouterConfig::new(64, 2);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        // Set alternating experts as resident
        let resident: Vec<usize> = (0..64).step_by(2).collect();
        router.update_cache_state(&resident);

        let mut id = 0usize;
        b.iter(|| {
            id = (id + 1) % 64;
            black_box(router.is_resident(id))
        });
    });

    // P1: Large expert count (>64, uses extended bitmask)
    group.bench_function("cache_mask_check_128", |b| {
        let config = RouterConfig::new(128, 4);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        let resident: Vec<usize> = (0..128).step_by(2).collect();
        router.update_cache_state(&resident);

        let mut id = 0usize;
        b.iter(|| {
            id = (id + 1) % 128;
            black_box(router.is_resident(id))
        });
    });

    // Compare top-2 vs top-4 selection
    group.bench_function("select_top2_vs_sort", |b| {
        let config = RouterConfig::new(64, 2).with_cache_bonus(0.15);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        let gate_logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();

        b.iter(|| black_box(router.select_top_k(black_box(&gate_logits))));
    });

    group.bench_function("select_top4_partial_sort", |b| {
        let config = RouterConfig::new(64, 4).with_cache_bonus(0.15);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        let gate_logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();

        b.iter(|| black_box(router.select_top_k(black_box(&gate_logits))));
    });

    group.finish();
}

/// Benchmark: SIMD affinity decay (P1 optimization)
fn bench_simd_affinity_decay(c: &mut Criterion) {
    use ruvllm::moe::{AffinityConfig, ExpertAffinity};

    let mut group = c.benchmark_group("simd_affinity_decay");

    for num_experts in [8, 16, 32, 64, 128, 256] {
        group.throughput(Throughput::Elements(num_experts as u64));

        group.bench_with_input(
            BenchmarkId::new("decay_all", num_experts),
            &num_experts,
            |b, &n| {
                let config = AffinityConfig::with_num_experts(n).with_decay(0.95);
                let mut affinity = ExpertAffinity::new(config);

                // Activate all experts initially
                let all: Vec<usize> = (0..n).collect();
                affinity.update(&all);

                b.iter(|| {
                    affinity.update(&[]); // Decay-only update
                    black_box(affinity.score(0))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("update_with_activation", num_experts),
            &num_experts,
            |b, &n| {
                let config = AffinityConfig::with_num_experts(n).with_decay(0.95);
                let mut affinity = ExpertAffinity::new(config);

                let activated = vec![0, 1]; // Activate 2 experts per call

                b.iter(|| {
                    affinity.update(&activated);
                    black_box(affinity.score(0))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_routing_overhead,
    bench_affinity_update,
    bench_precision_allocation,
    bench_cache_hit_rate_simulation,
    bench_paging_simulation,
    bench_batch_scheduler,
    bench_memory_footprint,
    bench_prefetch_decision,
    bench_eviction_policies,
    bench_memory_aware_router,
    bench_simd_affinity_decay,
);

criterion_main!(benches);
