//! Performance tracking and metrics collection

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

/// Performance metrics collector
pub struct Metrics {
    enabled: bool,
    task_durations: RwLock<Vec<Duration>>,
    task_count: RwLock<usize>,
    success_count: RwLock<usize>,
    failure_count: RwLock<usize>,
    resource_usage: RwLock<ResourceUsage>,
    custom_metrics: RwLock<HashMap<String, f64>>,
    started_at: DateTime<Utc>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Total memory allocated in bytes
    pub memory_bytes: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Total CPU time in milliseconds
    pub cpu_time_ms: u64,
    /// Number of threads used
    pub thread_count: usize,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_bytes: 0,
            peak_memory_bytes: 0,
            cpu_time_ms: 0,
            thread_count: 0,
        }
    }
}

/// Latency histogram buckets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// < 10ms
    pub p10ms: usize,
    /// 10-50ms
    pub p50ms: usize,
    /// 50-100ms
    pub p100ms: usize,
    /// 100-500ms
    pub p500ms: usize,
    /// 500ms-1s
    pub p1s: usize,
    /// 1s-5s
    pub p5s: usize,
    /// > 5s
    pub p5s_plus: usize,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            p10ms: 0,
            p50ms: 0,
            p100ms: 0,
            p500ms: 0,
            p1s: 0,
            p5s: 0,
            p5s_plus: 0,
        }
    }
}

impl LatencyHistogram {
    fn record(&mut self, duration: Duration) {
        let ms = duration.as_millis();
        match ms {
            0..=10 => self.p10ms += 1,
            11..=50 => self.p50ms += 1,
            51..=100 => self.p100ms += 1,
            101..=500 => self.p500ms += 1,
            501..=1000 => self.p1s += 1,
            1001..=5000 => self.p5s += 1,
            _ => self.p5s_plus += 1,
        }
    }
}

/// Metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Total tasks executed
    pub total_tasks: usize,
    /// Successful tasks
    pub success_count: usize,
    /// Failed tasks
    pub failure_count: usize,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average task duration in milliseconds
    pub avg_duration_ms: f64,
    /// Median task duration in milliseconds
    pub median_duration_ms: f64,
    /// P95 task duration in milliseconds
    pub p95_duration_ms: f64,
    /// P99 task duration in milliseconds
    pub p99_duration_ms: f64,
    /// Minimum task duration in milliseconds
    pub min_duration_ms: f64,
    /// Maximum task duration in milliseconds
    pub max_duration_ms: f64,
    /// Throughput (tasks per second)
    pub throughput: f64,
    /// Latency histogram
    pub latency_histogram: LatencyHistogram,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Metrics collection start time
    pub started_at: DateTime<Utc>,
    /// Total elapsed time in seconds
    pub elapsed_secs: f64,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            task_durations: RwLock::new(Vec::new()),
            task_count: RwLock::new(0),
            success_count: RwLock::new(0),
            failure_count: RwLock::new(0),
            resource_usage: RwLock::new(ResourceUsage::default()),
            custom_metrics: RwLock::new(HashMap::new()),
            started_at: Utc::now(),
        }
    }

    /// Record a task duration
    pub fn record_task_duration(&self, duration: Duration) {
        if !self.enabled {
            return;
        }

        let mut durations = self.task_durations.write().unwrap();
        durations.push(duration);

        let mut count = self.task_count.write().unwrap();
        *count += 1;
    }

    /// Record a successful task
    pub fn record_success(&self) {
        if !self.enabled {
            return;
        }

        let mut count = self.success_count.write().unwrap();
        *count += 1;
    }

    /// Record a failed task
    pub fn record_failure(&self) {
        if !self.enabled {
            return;
        }

        let mut count = self.failure_count.write().unwrap();
        *count += 1;
    }

    /// Update resource usage
    pub fn update_resource_usage(&self, memory_bytes: u64, cpu_time_ms: u64) {
        if !self.enabled {
            return;
        }

        let mut usage = self.resource_usage.write().unwrap();
        usage.memory_bytes = memory_bytes;
        usage.peak_memory_bytes = usage.peak_memory_bytes.max(memory_bytes);
        usage.cpu_time_ms += cpu_time_ms;
    }

    /// Record a custom metric
    pub fn record_custom(&self, key: impl Into<String>, value: f64) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.custom_metrics.write().unwrap();
        metrics.insert(key.into(), value);
    }

    /// Calculate percentile from sorted durations
    fn percentile(durations: &[Duration], p: f64) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }

        let index = ((durations.len() as f64 * p) as usize).min(durations.len() - 1);
        durations[index].as_millis() as f64
    }

    /// Generate metrics summary
    pub fn summary(&self) -> MetricsSummary {
        let durations = self.task_durations.read().unwrap();
        let task_count = *self.task_count.read().unwrap();
        let success_count = *self.success_count.read().unwrap();
        let failure_count = *self.failure_count.read().unwrap();
        let resource_usage = self.resource_usage.read().unwrap().clone();
        let custom_metrics = self.custom_metrics.read().unwrap().clone();

        let mut sorted_durations = durations.clone();
        sorted_durations.sort();

        let avg_duration_ms = if !sorted_durations.is_empty() {
            sorted_durations.iter().map(|d| d.as_millis() as f64).sum::<f64>()
                / sorted_durations.len() as f64
        } else {
            0.0
        };

        let median_duration_ms = Self::percentile(&sorted_durations, 0.5);
        let p95_duration_ms = Self::percentile(&sorted_durations, 0.95);
        let p99_duration_ms = Self::percentile(&sorted_durations, 0.99);

        let min_duration_ms = sorted_durations
            .first()
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0);

        let max_duration_ms = sorted_durations
            .last()
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0);

        let elapsed_secs = (Utc::now() - self.started_at).num_milliseconds() as f64 / 1000.0;
        let throughput = if elapsed_secs > 0.0 {
            task_count as f64 / elapsed_secs
        } else {
            0.0
        };

        let success_rate = if task_count > 0 {
            success_count as f64 / task_count as f64
        } else {
            0.0
        };

        // Build latency histogram
        let mut histogram = LatencyHistogram::default();
        for duration in sorted_durations.iter() {
            histogram.record(*duration);
        }

        MetricsSummary {
            total_tasks: task_count,
            success_count,
            failure_count,
            success_rate,
            avg_duration_ms,
            median_duration_ms,
            p95_duration_ms,
            p99_duration_ms,
            min_duration_ms,
            max_duration_ms,
            throughput,
            latency_histogram: histogram,
            resource_usage,
            custom_metrics,
            started_at: self.started_at,
            elapsed_secs,
        }
    }

    /// Export metrics to JSON
    pub fn export(&self) -> serde_json::Value {
        serde_json::to_value(self.summary()).unwrap_or(serde_json::json!({}))
    }

    /// Reset all metrics
    pub fn reset(&self) {
        let mut durations = self.task_durations.write().unwrap();
        durations.clear();

        let mut task_count = self.task_count.write().unwrap();
        *task_count = 0;

        let mut success_count = self.success_count.write().unwrap();
        *success_count = 0;

        let mut failure_count = self.failure_count.write().unwrap();
        *failure_count = 0;

        let mut usage = self.resource_usage.write().unwrap();
        *usage = ResourceUsage::default();

        let mut custom = self.custom_metrics.write().unwrap();
        custom.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_recording() {
        let metrics = Metrics::new(true);

        metrics.record_task_duration(Duration::from_millis(100));
        metrics.record_task_duration(Duration::from_millis(200));
        metrics.record_task_duration(Duration::from_millis(150));

        metrics.record_success();
        metrics.record_success();
        metrics.record_failure();

        let summary = metrics.summary();
        assert_eq!(summary.total_tasks, 3);
        assert_eq!(summary.success_count, 2);
        assert_eq!(summary.failure_count, 1);
        assert_eq!(summary.success_rate, 2.0 / 3.0);
        assert!(summary.avg_duration_ms > 0.0);
    }

    #[test]
    fn test_latency_histogram() {
        let metrics = Metrics::new(true);

        metrics.record_task_duration(Duration::from_millis(5));
        metrics.record_task_duration(Duration::from_millis(25));
        metrics.record_task_duration(Duration::from_millis(75));
        metrics.record_task_duration(Duration::from_millis(250));
        metrics.record_task_duration(Duration::from_millis(750));

        let summary = metrics.summary();
        assert_eq!(summary.latency_histogram.p10ms, 1);
        assert_eq!(summary.latency_histogram.p50ms, 1);
        assert_eq!(summary.latency_histogram.p100ms, 1);
        assert_eq!(summary.latency_histogram.p500ms, 1);
        assert_eq!(summary.latency_histogram.p1s, 1);
    }

    #[test]
    fn test_custom_metrics() {
        let metrics = Metrics::new(true);

        metrics.record_custom("cache_hits", 42.0);
        metrics.record_custom("cache_misses", 8.0);

        let summary = metrics.summary();
        assert_eq!(summary.custom_metrics.get("cache_hits"), Some(&42.0));
        assert_eq!(summary.custom_metrics.get("cache_misses"), Some(&8.0));
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new(true);

        metrics.record_task_duration(Duration::from_millis(100));
        metrics.record_success();

        metrics.reset();

        let summary = metrics.summary();
        assert_eq!(summary.total_tasks, 0);
        assert_eq!(summary.success_count, 0);
    }
}
