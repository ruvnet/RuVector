//! MoE (Mixture of Experts) Module
//!
//! This module provides components for efficient Mixture of Experts inference,
//! including routing metrics tracking, expert affinity tracking, and performance
//! monitoring.
//!
//! ## Overview
//!
//! MoE architectures use sparse expert activation to achieve high model capacity
//! while keeping compute costs manageable. Key challenges include:
//!
//! - **Expert Cache Management**: Keeping frequently-used experts in memory
//! - **Routing Efficiency**: Minimizing overhead from expert selection
//! - **Paging Overhead**: Managing memory transfers for cold experts
//! - **Affinity Tracking**: Understanding expert co-activation patterns
//!
//! ## Key Components
//!
//! - [`MemoryAwareRouter`]: Memory-aware expert routing with cache residency bonus (ADR-092)
//! - [`RouterConfig`]: Configuration for router behavior and cache bonus parameters
//! - [`ExpertAffinity`]: EMA-based affinity tracking for memory-aware routing (ADR-092)
//! - [`AffinityConfig`]: Configuration for affinity tracking parameters
//! - [`MoeMetrics`]: Real-time tracking of cache hits, misses, paging, and routing
//! - [`MoeMetricsSummary`]: Aggregated performance summary statistics
//! - [`PagingRequest`]: Request to page experts in/out of cache
//!
//! ## ADR-092 Compliance
//!
//! This module implements memory-aware expert routing as specified in ADR-092:
//!
//! - **INV-2: Affinity Monotonicity**: EMA-based affinity scores decrease monotonically
//!   without new activations, ensuring predictable eviction behavior.
//! - **INV-6: Router Determinism**: Same input + cache state always produces same result
//! - Cache hit rates and prefetch effectiveness tracking
//! - Paging latency distribution
//! - Routing decision throughput
//! - Target: >=70% cache hit rate (vs 34% baseline)
//!
//! ## Example: Affinity Tracking
//!
//! ```rust
//! use ruvllm::moe::{ExpertAffinity, AffinityConfig};
//!
//! let config = AffinityConfig::with_num_experts(8).with_decay(0.95);
//! let mut affinity = ExpertAffinity::new(config);
//!
//! // Experts 2 and 5 were selected this round
//! affinity.update(&[2, 5]);
//!
//! // Get current affinity scores
//! let scores = affinity.scores();
//! assert!(scores[2] > scores[0]); // Expert 2 was activated
//!
//! // Get top-3 experts for prefetching
//! let top3 = affinity.top_k_by_affinity(3);
//! ```
//!
//! ## Example: Metrics Tracking
//!
//! ```rust,ignore
//! use ruvllm::moe::{MoeMetrics, MoeMetricsSummary};
//!
//! let mut metrics = MoeMetrics::new();
//!
//! // Track cache operations
//! metrics.record_cache_hit();
//! metrics.record_cache_miss();
//! metrics.record_page_in(Duration::from_micros(150));
//!
//! // Get summary statistics
//! let summary = metrics.summary();
//! println!("Hit rate: {:.2}%", summary.hit_rate * 100.0);
//! println!("Avg paging latency: {:.1}us", summary.avg_paging_latency_us);
//! ```

pub mod affinity;
pub mod metrics;
pub mod precision_allocator;
pub mod router;
pub mod sram_mapper;

/// Expert identifier type (matches bitnet/expert_cache.rs convention).
///
/// Expert IDs are zero-indexed integers. For a model with `num_experts=8`,
/// valid IDs are `0..8`.
pub type ExpertId = usize;

pub use affinity::{AffinityConfig, ExpertAffinity};
pub use metrics::{MoeMetrics, MoeMetricsSummary};
pub use precision_allocator::{ExpertPrecision, PrecisionAllocator, PrecisionConfig};
pub use router::{MemoryAwareRouter, PagingDirection, PagingPriority, PagingRequest, RouterConfig};
pub use sram_mapper::{HardwareConfig, HardwarePreset, MemoryTier, SramExpertAffinity, SramMapper};
