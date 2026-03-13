//! MoE Metrics Collection (ADR-092)
//!
//! Tracks cache hit rate, paging latency, and routing performance for
//! memory-aware expert routing.

use std::time::{Duration, Instant};

/// MoE routing and caching metrics.
///
/// Tracks cache hits, misses, paging operations, and timing information
/// to enable tuning of routing parameters.
#[derive(Debug, Clone, Default)]
pub struct MoeMetrics {
    /// Number of routing decisions where selected experts were cache-resident
    pub cache_hits: u64,
    /// Number of routing decisions requiring expert paging
    pub cache_misses: u64,
    /// Total experts paged in
    pub experts_paged_in: u64,
    /// Total experts paged out (evicted)
    pub experts_paged_out: u64,
    /// Total routing decisions made
    pub routing_decisions: u64,
    /// Cumulative routing latency in microseconds
    pub routing_latency_us: u64,
    /// Maximum routing latency in microseconds
    pub max_routing_latency_us: u64,
    /// Cumulative paging latency in microseconds
    pub paging_latency_us: u64,
    /// Maximum paging latency in microseconds
    pub max_paging_latency_us: u64,
    /// Number of prefetch operations
    pub prefetch_operations: u64,
    /// Successful prefetch hits (prefetched expert was subsequently used)
    pub prefetch_hits: u64,
    /// Affinity-based evictions (vs random/LRU)
    pub affinity_evictions: u64,
}

impl MoeMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Record multiple cache hits (P2 batch optimization)
    #[inline]
    pub fn record_cache_hits(&mut self, count: usize) {
        self.cache_hits += count as u64;
    }

    /// Record multiple cache misses (P2 batch optimization)
    #[inline]
    pub fn record_cache_misses(&mut self, count: usize) {
        self.cache_misses += count as u64;
    }

    /// Record expert paged in
    pub fn record_page_in(&mut self, latency: Duration) {
        self.experts_paged_in += 1;
        let latency_us = latency.as_micros() as u64;
        self.paging_latency_us += latency_us;
        self.max_paging_latency_us = self.max_paging_latency_us.max(latency_us);
    }

    /// Record expert paged out (evicted)
    pub fn record_page_out(&mut self) {
        self.experts_paged_out += 1;
    }

    /// Record a routing decision with latency
    pub fn record_routing(&mut self, latency: Duration) {
        self.routing_decisions += 1;
        let latency_us = latency.as_micros() as u64;
        self.routing_latency_us += latency_us;
        self.max_routing_latency_us = self.max_routing_latency_us.max(latency_us);
    }

    /// Record a prefetch operation
    pub fn record_prefetch(&mut self) {
        self.prefetch_operations += 1;
    }

    /// Record a successful prefetch hit
    pub fn record_prefetch_hit(&mut self) {
        self.prefetch_hits += 1;
    }

    /// Record an affinity-based eviction
    pub fn record_affinity_eviction(&mut self) {
        self.affinity_evictions += 1;
    }

    /// Compute the cache hit rate (0.0 - 1.0)
    ///
    /// Returns 0.0 if no routing decisions have been made.
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total as f32
    }

    /// Compute average routing latency in microseconds
    ///
    /// Returns 0.0 if no routing decisions have been made.
    pub fn avg_routing_latency_us(&self) -> f64 {
        if self.routing_decisions == 0 {
            return 0.0;
        }
        self.routing_latency_us as f64 / self.routing_decisions as f64
    }

    /// Compute average paging latency in microseconds
    ///
    /// Returns 0.0 if no paging operations have been recorded.
    pub fn avg_paging_latency_us(&self) -> f64 {
        if self.experts_paged_in == 0 {
            return 0.0;
        }
        self.paging_latency_us as f64 / self.experts_paged_in as f64
    }

    /// Compute prefetch accuracy (0.0 - 1.0)
    ///
    /// Returns 0.0 if no prefetch operations have been recorded.
    pub fn prefetch_accuracy(&self) -> f32 {
        if self.prefetch_operations == 0 {
            return 0.0;
        }
        self.prefetch_hits as f32 / self.prefetch_operations as f32
    }

    /// Generate a summary of current metrics
    pub fn summary(&self) -> MoeMetricsSummary {
        MoeMetricsSummary {
            hit_rate: self.hit_rate(),
            avg_routing_latency_us: self.avg_routing_latency_us(),
            max_routing_latency_us: self.max_routing_latency_us,
            avg_paging_latency_us: self.avg_paging_latency_us(),
            max_paging_latency_us: self.max_paging_latency_us,
            prefetch_accuracy: self.prefetch_accuracy(),
            total_routing_decisions: self.routing_decisions,
            total_page_operations: self.experts_paged_in + self.experts_paged_out,
        }
    }

    /// Reset all metrics to zero
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Summary of MoE metrics for reporting
#[derive(Debug, Clone)]
pub struct MoeMetricsSummary {
    /// Cache hit rate (0.0 - 1.0)
    pub hit_rate: f32,
    /// Average routing latency in microseconds
    pub avg_routing_latency_us: f64,
    /// Maximum routing latency in microseconds
    pub max_routing_latency_us: u64,
    /// Average paging latency in microseconds
    pub avg_paging_latency_us: f64,
    /// Maximum paging latency in microseconds
    pub max_paging_latency_us: u64,
    /// Prefetch accuracy (0.0 - 1.0)
    pub prefetch_accuracy: f32,
    /// Total number of routing decisions
    pub total_routing_decisions: u64,
    /// Total paging operations (in + out)
    pub total_page_operations: u64,
}

impl MoeMetricsSummary {
    /// Check if metrics meet ADR-092 targets
    ///
    /// Returns true if:
    /// - Cache hit rate >= 70%
    /// - Max routing latency <= 15 us (10 us target with some margin)
    pub fn meets_targets(&self) -> bool {
        self.hit_rate >= 0.70 && self.max_routing_latency_us <= 15
    }
}

/// Timer for measuring operation durations
pub struct MetricsTimer {
    start: Instant,
}

impl MetricsTimer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = MoeMetrics::new();
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut metrics = MoeMetrics::new();

        // 3 hits, 1 miss = 75% hit rate
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert!((metrics.hit_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_routing_latency() {
        let mut metrics = MoeMetrics::new();

        metrics.record_routing(Duration::from_micros(10));
        metrics.record_routing(Duration::from_micros(20));

        assert_eq!(metrics.routing_decisions, 2);
        assert!((metrics.avg_routing_latency_us() - 15.0).abs() < 1e-6);
        assert_eq!(metrics.max_routing_latency_us, 20);
    }

    #[test]
    fn test_prefetch_accuracy() {
        let mut metrics = MoeMetrics::new();

        metrics.record_prefetch();
        metrics.record_prefetch();
        metrics.record_prefetch();
        metrics.record_prefetch_hit();
        metrics.record_prefetch_hit();

        // 2 hits out of 3 prefetches = 66.67%
        assert!((metrics.prefetch_accuracy() - 0.6666667).abs() < 1e-6);
    }

    #[test]
    fn test_summary_meets_targets() {
        let summary = MoeMetricsSummary {
            hit_rate: 0.75,
            avg_routing_latency_us: 8.0,
            max_routing_latency_us: 12,
            avg_paging_latency_us: 100.0,
            max_paging_latency_us: 200,
            prefetch_accuracy: 0.6,
            total_routing_decisions: 100,
            total_page_operations: 20,
        };

        assert!(summary.meets_targets());
    }

    #[test]
    fn test_summary_fails_targets() {
        let summary = MoeMetricsSummary {
            hit_rate: 0.50, // Below 70%
            avg_routing_latency_us: 8.0,
            max_routing_latency_us: 12,
            avg_paging_latency_us: 100.0,
            max_paging_latency_us: 200,
            prefetch_accuracy: 0.6,
            total_routing_decisions: 100,
            total_page_operations: 20,
        };

        assert!(!summary.meets_targets());
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = MoeMetrics::new();
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.record_routing(Duration::from_micros(10));

        metrics.reset();

        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.routing_decisions, 0);
    }

    #[test]
    fn test_metrics_timer() {
        let timer = MetricsTimer::start();
        // Just verify it doesn't panic
        let _elapsed = timer.elapsed();
    }

    #[test]
    fn test_bulk_cache_recording() {
        let mut metrics = MoeMetrics::new();

        // P2 optimization: bulk recording
        metrics.record_cache_hits(5);
        metrics.record_cache_misses(2);

        assert_eq!(metrics.cache_hits, 5);
        assert_eq!(metrics.cache_misses, 2);

        // Mix with single recording
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert_eq!(metrics.cache_hits, 6);
        assert_eq!(metrics.cache_misses, 3);

        // Hit rate should be 6/9 = 66.67%
        assert!((metrics.hit_rate() - 0.6666667).abs() < 1e-5);
    }
}
