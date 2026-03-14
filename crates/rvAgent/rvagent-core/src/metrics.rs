//! Performance metrics collection (ADR-103 A9).
//!
//! Lightweight, lock-free counters for tracking tool calls, model calls,
//! token usage, and cumulative latencies. Designed for always-on use in
//! production without measurable overhead.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lightweight metrics collector for tracking performance.
///
/// All fields use `AtomicU64` with relaxed ordering for maximum throughput.
/// A consistent [`MetricsSnapshot`] can be obtained via [`snapshot`](Metrics::snapshot).
pub struct Metrics {
    /// Total number of tool invocations.
    pub tool_calls: AtomicU64,
    /// Total number of LLM model calls.
    pub model_calls: AtomicU64,
    /// Cumulative token count across all model calls.
    pub total_tokens: AtomicU64,
    /// Cumulative middleware pipeline time in nanoseconds.
    pub middleware_ns: AtomicU64,
    /// Cumulative tool execution time in nanoseconds.
    pub tool_ns: AtomicU64,
    /// Cumulative model call time in nanoseconds (for avg calculation).
    model_ns: AtomicU64,
}

impl Metrics {
    /// Create a zeroed metrics collector.
    pub fn new() -> Self {
        Self {
            tool_calls: AtomicU64::new(0),
            model_calls: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            middleware_ns: AtomicU64::new(0),
            tool_ns: AtomicU64::new(0),
            model_ns: AtomicU64::new(0),
        }
    }

    /// Record a completed tool call with its duration in nanoseconds.
    pub fn record_tool_call(&self, duration_ns: u64) {
        self.tool_calls.fetch_add(1, Ordering::Relaxed);
        self.tool_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record a completed model call with token count and duration in nanoseconds.
    pub fn record_model_call(&self, tokens: u64, duration_ns: u64) {
        self.model_calls.fetch_add(1, Ordering::Relaxed);
        self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.model_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record middleware pipeline processing time in nanoseconds.
    pub fn record_middleware(&self, duration_ns: u64) {
        self.middleware_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Take a point-in-time snapshot of all metrics.
    ///
    /// The snapshot is not strictly consistent across fields (no global lock),
    /// but each individual field is accurate at the moment it is read.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let tool_calls = self.tool_calls.load(Ordering::Relaxed);
        let model_calls = self.model_calls.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);
        let middleware_ns = self.middleware_ns.load(Ordering::Relaxed);
        let tool_ns = self.tool_ns.load(Ordering::Relaxed);

        let avg_middleware_us = if tool_calls + model_calls > 0 {
            middleware_ns as f64 / (tool_calls + model_calls) as f64 / 1000.0
        } else {
            0.0
        };

        let avg_tool_us = if tool_calls > 0 {
            tool_ns as f64 / tool_calls as f64 / 1000.0
        } else {
            0.0
        };

        MetricsSnapshot {
            tool_calls,
            model_calls,
            total_tokens,
            avg_middleware_us,
            avg_tool_us,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of performance metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total tool calls recorded.
    pub tool_calls: u64,
    /// Total model calls recorded.
    pub model_calls: u64,
    /// Total tokens consumed.
    pub total_tokens: u64,
    /// Average middleware pipeline latency in microseconds.
    pub avg_middleware_us: f64,
    /// Average tool execution latency in microseconds.
    pub avg_tool_us: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zeroed() {
        let m = Metrics::new();
        let s = m.snapshot();
        assert_eq!(s.tool_calls, 0);
        assert_eq!(s.model_calls, 0);
        assert_eq!(s.total_tokens, 0);
        assert_eq!(s.avg_middleware_us, 0.0);
        assert_eq!(s.avg_tool_us, 0.0);
    }

    #[test]
    fn test_record_tool() {
        let m = Metrics::new();
        m.record_tool_call(1_000_000); // 1ms
        m.record_tool_call(3_000_000); // 3ms
        let s = m.snapshot();
        assert_eq!(s.tool_calls, 2);
        assert_eq!(s.avg_tool_us, 2_000.0); // 2ms average
    }
}
