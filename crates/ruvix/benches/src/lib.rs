//! RuVix Benchmark Suite - Comprehensive benchmarks comparing RuVix Cognition Kernel
//! against Linux syscalls.
//!
//! This crate provides benchmarks for all 12 RuVix syscalls defined in ADR-087,
//! comparing their performance against equivalent Linux operations.
//!
//! # Target Latencies (ADR-087 Section 3.2)
//!
//! | Syscall | Target | Notes |
//! |---------|--------|-------|
//! | task_spawn | 10us | RVF component spawning |
//! | cap_grant | 500ns | O(1) capability lookup |
//! | region_map | 5us | Region-based memory |
//! | queue_send | 200ns | Zero-copy IPC |
//! | queue_recv | 200ns | Zero-copy IPC |
//! | timer_wait | 100ns | Timer scheduling |
//! | rvf_mount | 1ms | RVF package verification |
//! | attest_emit | 500ns | 82-byte attestation |
//! | vector_get | 100ns | Vector retrieval |
//! | vector_put_proved | 500ns | Reflex tier proof |
//! | graph_apply_proved | 1us | Standard tier proof |
//! | sensor_subscribe | 5us | Sensor registration |
//!
//! # Linux Comparison Points
//!
//! | RuVix Syscall | Linux Equivalent | Expected Speedup |
//! |---------------|------------------|------------------|
//! | cap_grant | setuid/capabilities | 16x |
//! | queue_send/recv | pipe/socket | 6x |
//! | region_map | mmap | 3x |
//! | task_spawn | fork/clone | 4x |
//! | vector_put_proved | write+fsync | Variable |

#![deny(unsafe_op_in_unsafe_fn)]

use std::time::Duration;

pub mod comparison;
pub mod linux;
pub mod report;
pub mod ruvix;
pub mod stats;
pub mod targets;

/// ADR-087 target latencies for each syscall.
pub const TARGETS: &[(&str, Duration)] = &[
    ("task_spawn", Duration::from_micros(10)),
    ("cap_grant", Duration::from_nanos(500)), // O(1) capability lookup
    ("region_map", Duration::from_micros(5)),
    ("queue_send", Duration::from_nanos(200)), // Zero-copy target
    ("queue_recv", Duration::from_nanos(200)),
    ("timer_wait", Duration::from_nanos(100)),
    ("rvf_mount", Duration::from_millis(1)),
    ("attest_emit", Duration::from_nanos(500)), // 82-byte attestation
    ("vector_get", Duration::from_nanos(100)),
    ("vector_put_proved", Duration::from_nanos(500)), // Reflex tier
    ("graph_apply_proved", Duration::from_micros(1)), // Standard tier
    ("sensor_subscribe", Duration::from_micros(5)),
];

/// Returns the target latency for a given syscall name.
pub fn target_for(syscall: &str) -> Option<Duration> {
    TARGETS
        .iter()
        .find(|(name, _)| *name == syscall)
        .map(|(_, target)| *target)
}

/// Benchmark result for a single operation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name.
    pub operation: String,
    /// Number of iterations.
    pub iterations: u64,
    /// Mean latency in nanoseconds.
    pub mean_ns: f64,
    /// Median latency (p50) in nanoseconds.
    pub p50_ns: f64,
    /// 95th percentile latency in nanoseconds.
    pub p95_ns: f64,
    /// 99th percentile latency in nanoseconds.
    pub p99_ns: f64,
    /// Minimum latency in nanoseconds.
    pub min_ns: f64,
    /// Maximum latency in nanoseconds.
    pub max_ns: f64,
    /// Standard deviation in nanoseconds.
    pub std_dev_ns: f64,
    /// Target latency (if applicable) in nanoseconds.
    pub target_ns: Option<f64>,
    /// Whether the benchmark meets its target.
    pub meets_target: bool,
}

impl BenchmarkResult {
    /// Creates a new benchmark result from raw measurements.
    pub fn from_measurements(
        operation: &str,
        measurements_ns: &[f64],
        target: Option<Duration>,
    ) -> Self {
        let n = measurements_ns.len() as f64;
        let mean_ns = measurements_ns.iter().sum::<f64>() / n;

        let mut sorted: Vec<f64> = measurements_ns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_idx = (measurements_ns.len() as f64 * 0.50) as usize;
        let p95_idx = (measurements_ns.len() as f64 * 0.95) as usize;
        let p99_idx = (measurements_ns.len() as f64 * 0.99) as usize;

        let p50_ns = sorted.get(p50_idx).copied().unwrap_or(mean_ns);
        let p95_ns = sorted.get(p95_idx).copied().unwrap_or(mean_ns);
        let p99_ns = sorted.get(p99_idx).copied().unwrap_or(mean_ns);

        let min_ns = sorted.first().copied().unwrap_or(0.0);
        let max_ns = sorted.last().copied().unwrap_or(0.0);

        let variance: f64 = measurements_ns
            .iter()
            .map(|x| (x - mean_ns).powi(2))
            .sum::<f64>()
            / n;
        let std_dev_ns = variance.sqrt();

        let target_ns = target.map(|t| t.as_nanos() as f64);
        let meets_target = target_ns.map(|t| p95_ns <= t).unwrap_or(true);

        Self {
            operation: operation.to_string(),
            iterations: measurements_ns.len() as u64,
            mean_ns,
            p50_ns,
            p95_ns,
            p99_ns,
            min_ns,
            max_ns,
            std_dev_ns,
            target_ns,
            meets_target,
        }
    }

    /// Returns the speedup over another result.
    pub fn speedup_over(&self, other: &BenchmarkResult) -> f64 {
        other.mean_ns / self.mean_ns
    }
}

/// Comparison result between RuVix and Linux operations.
#[derive(Debug, Clone)]
pub struct Comparison {
    /// Operation name.
    pub operation: String,
    /// RuVix syscall name.
    pub ruvix_syscall: String,
    /// Linux equivalent operation.
    pub linux_equivalent: String,
    /// RuVix benchmark result.
    pub ruvix_result: BenchmarkResult,
    /// Linux benchmark result.
    pub linux_result: BenchmarkResult,
    /// Speedup (Linux time / RuVix time).
    pub speedup: f64,
    /// Notes about the comparison.
    pub notes: String,
}

impl Comparison {
    /// Creates a new comparison.
    pub fn new(
        operation: &str,
        ruvix_syscall: &str,
        linux_equivalent: &str,
        ruvix_result: BenchmarkResult,
        linux_result: BenchmarkResult,
        notes: &str,
    ) -> Self {
        let speedup = ruvix_result.speedup_over(&linux_result);
        Self {
            operation: operation.to_string(),
            ruvix_syscall: ruvix_syscall.to_string(),
            linux_equivalent: linux_equivalent.to_string(),
            ruvix_result,
            linux_result,
            speedup,
            notes: notes.to_string(),
        }
    }
}

/// Memory overhead comparison.
#[derive(Debug, Clone)]
pub struct MemoryComparison {
    /// Operation name.
    pub operation: String,
    /// RuVix memory usage in bytes.
    pub ruvix_bytes: usize,
    /// Linux memory usage in bytes.
    pub linux_bytes: usize,
    /// Memory reduction (1 - ruvix/linux).
    pub reduction: f64,
    /// Notes about the comparison.
    pub notes: String,
}

impl MemoryComparison {
    /// Creates a new memory comparison.
    pub fn new(operation: &str, ruvix_bytes: usize, linux_bytes: usize, notes: &str) -> Self {
        let reduction = if linux_bytes > 0 {
            1.0 - (ruvix_bytes as f64 / linux_bytes as f64)
        } else {
            0.0
        };
        Self {
            operation: operation.to_string(),
            ruvix_bytes,
            linux_bytes,
            reduction,
            notes: notes.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_lookup() {
        assert_eq!(target_for("cap_grant"), Some(Duration::from_nanos(500)));
        assert_eq!(target_for("task_spawn"), Some(Duration::from_micros(10)));
        assert_eq!(target_for("nonexistent"), None);
    }

    #[test]
    fn test_benchmark_result_from_measurements() {
        let measurements = vec![100.0, 110.0, 105.0, 95.0, 102.0];
        let result = BenchmarkResult::from_measurements(
            "test_op",
            &measurements,
            Some(Duration::from_nanos(120)),
        );

        assert_eq!(result.iterations, 5);
        assert!((result.mean_ns - 102.4).abs() < 0.1);
        assert!(result.meets_target);
    }

    #[test]
    fn test_speedup_calculation() {
        let ruvix = BenchmarkResult::from_measurements("ruvix", &[100.0, 100.0], None);
        let linux = BenchmarkResult::from_measurements("linux", &[600.0, 600.0], None);

        let speedup = ruvix.speedup_over(&linux);
        assert!((speedup - 6.0).abs() < 0.1);
    }
}
