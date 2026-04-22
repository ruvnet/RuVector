//! Statistics utilities for benchmark analysis.
//!
//! Provides statistical functions for analyzing benchmark results.

use hdrhistogram::Histogram;

/// Statistical summary of measurements.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Number of samples.
    pub count: u64,
    /// Mean value.
    pub mean: f64,
    /// Median (p50).
    pub median: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// 99.9th percentile.
    pub p999: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Coefficient of variation (std_dev / mean).
    pub cv: f64,
}

impl Stats {
    /// Computes statistics from a slice of f64 measurements (in nanoseconds).
    pub fn from_measurements(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::empty();
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        Self {
            count: data.len() as u64,
            mean,
            median: percentile(50.0),
            p90: percentile(90.0),
            p95: percentile(95.0),
            p99: percentile(99.0),
            p999: percentile(99.9),
            min: *sorted.first().unwrap(),
            max: *sorted.last().unwrap(),
            std_dev,
            cv: if mean > 0.0 { std_dev / mean } else { 0.0 },
        }
    }

    /// Creates empty statistics.
    pub fn empty() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p999: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
            cv: 0.0,
        }
    }

    /// Formats the stats as a human-readable string.
    pub fn format(&self) -> String {
        format!(
            "n={} mean={:.1}ns median={:.1}ns p95={:.1}ns p99={:.1}ns min={:.1}ns max={:.1}ns stddev={:.1}ns cv={:.2}%",
            self.count, self.mean, self.median, self.p95, self.p99, self.min, self.max, self.std_dev, self.cv * 100.0
        )
    }
}

/// HDR histogram wrapper for high-precision latency tracking.
pub struct LatencyHistogram {
    histogram: Histogram<u64>,
}

impl LatencyHistogram {
    /// Creates a new histogram with default settings.
    ///
    /// Tracks latencies from 1ns to 1 second with 3 significant digits.
    pub fn new() -> Self {
        let histogram =
            Histogram::new_with_bounds(1, 1_000_000_000, 3).expect("Failed to create histogram");
        Self { histogram }
    }

    /// Records a latency value in nanoseconds.
    pub fn record(&mut self, nanos: u64) {
        let _ = self.histogram.record(nanos);
    }

    /// Records multiple samples from a slice.
    pub fn record_all(&mut self, samples: &[f64]) {
        for &sample in samples {
            self.record(sample as u64);
        }
    }

    /// Returns the count of recorded values.
    pub fn count(&self) -> u64 {
        self.histogram.len()
    }

    /// Returns the minimum recorded value.
    pub fn min(&self) -> u64 {
        self.histogram.min()
    }

    /// Returns the maximum recorded value.
    pub fn max(&self) -> u64 {
        self.histogram.max()
    }

    /// Returns the mean of all recorded values.
    pub fn mean(&self) -> f64 {
        self.histogram.mean()
    }

    /// Returns the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.histogram.stdev()
    }

    /// Returns a percentile value.
    pub fn percentile(&self, p: f64) -> u64 {
        self.histogram.value_at_percentile(p)
    }

    /// Returns comprehensive statistics.
    pub fn stats(&self) -> HistogramStats {
        HistogramStats {
            count: self.count(),
            min: self.min(),
            max: self.max(),
            mean: self.mean(),
            std_dev: self.std_dev(),
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
            p999: self.percentile(99.9),
        }
    }

    /// Merges another histogram into this one.
    pub fn merge(&mut self, other: &LatencyHistogram) {
        self.histogram.add(&other.histogram).ok();
    }

    /// Resets the histogram.
    pub fn reset(&mut self) {
        self.histogram.reset();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from HDR histogram.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Total samples.
    pub count: u64,
    /// Minimum value.
    pub min: u64,
    /// Maximum value.
    pub max: u64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// 50th percentile (median).
    pub p50: u64,
    /// 90th percentile.
    pub p90: u64,
    /// 95th percentile.
    pub p95: u64,
    /// 99th percentile.
    pub p99: u64,
    /// 99.9th percentile.
    pub p999: u64,
}

impl HistogramStats {
    /// Formats as human-readable string.
    pub fn format(&self) -> String {
        format!(
            "n={} mean={:.0}ns p50={}ns p95={}ns p99={}ns p99.9={}ns min={}ns max={}ns",
            self.count, self.mean, self.p50, self.p95, self.p99, self.p999, self.min, self.max
        )
    }

    /// Formats as compact latency string.
    pub fn format_latency(&self) -> String {
        let format_time = |nanos: u64| -> String {
            if nanos >= 1_000_000 {
                format!("{:.2}ms", nanos as f64 / 1_000_000.0)
            } else if nanos >= 1_000 {
                format!("{:.2}us", nanos as f64 / 1_000.0)
            } else {
                format!("{}ns", nanos)
            }
        };

        format!(
            "p50={} p95={} p99={} max={}",
            format_time(self.p50),
            format_time(self.p95),
            format_time(self.p99),
            format_time(self.max)
        )
    }
}

/// Throughput calculator.
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    /// Total operations.
    pub operations: u64,
    /// Total duration in nanoseconds.
    pub duration_ns: u64,
    /// Operations per second.
    pub ops_per_sec: f64,
    /// Bytes processed (if applicable).
    pub bytes: u64,
    /// Bytes per second (if applicable).
    pub bytes_per_sec: f64,
}

impl ThroughputStats {
    /// Creates throughput stats from operations and duration.
    pub fn from_ops(operations: u64, duration_ns: u64) -> Self {
        let duration_sec = duration_ns as f64 / 1_000_000_000.0;
        let ops_per_sec = if duration_sec > 0.0 {
            operations as f64 / duration_sec
        } else {
            0.0
        };

        Self {
            operations,
            duration_ns,
            ops_per_sec,
            bytes: 0,
            bytes_per_sec: 0.0,
        }
    }

    /// Creates throughput stats including byte count.
    pub fn from_ops_and_bytes(operations: u64, bytes: u64, duration_ns: u64) -> Self {
        let duration_sec = duration_ns as f64 / 1_000_000_000.0;
        let ops_per_sec = if duration_sec > 0.0 {
            operations as f64 / duration_sec
        } else {
            0.0
        };
        let bytes_per_sec = if duration_sec > 0.0 {
            bytes as f64 / duration_sec
        } else {
            0.0
        };

        Self {
            operations,
            duration_ns,
            ops_per_sec,
            bytes,
            bytes_per_sec,
        }
    }

    /// Formats as human-readable string.
    pub fn format(&self) -> String {
        let format_rate = |rate: f64| -> String {
            if rate >= 1_000_000.0 {
                format!("{:.2}M op/s", rate / 1_000_000.0)
            } else if rate >= 1_000.0 {
                format!("{:.2}K op/s", rate / 1_000.0)
            } else {
                format!("{:.0} op/s", rate)
            }
        };

        if self.bytes > 0 {
            let bw = if self.bytes_per_sec >= 1_000_000_000.0 {
                format!("{:.2} GB/s", self.bytes_per_sec / 1_000_000_000.0)
            } else if self.bytes_per_sec >= 1_000_000.0 {
                format!("{:.2} MB/s", self.bytes_per_sec / 1_000_000.0)
            } else {
                format!("{:.2} KB/s", self.bytes_per_sec / 1_000.0)
            };
            format!("{} ({})", format_rate(self.ops_per_sec), bw)
        } else {
            format_rate(self.ops_per_sec)
        }
    }
}

/// Compares two sets of measurements and returns speedup.
pub fn calculate_speedup(baseline: &[f64], optimized: &[f64]) -> f64 {
    let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
    let optimized_mean = optimized.iter().sum::<f64>() / optimized.len() as f64;

    if optimized_mean > 0.0 {
        baseline_mean / optimized_mean
    } else {
        1.0
    }
}

/// Statistical significance test (simple t-test approximation).
pub fn is_significant(a: &[f64], b: &[f64], confidence: f64) -> bool {
    if a.len() < 30 || b.len() < 30 {
        return false;
    }

    let mean_a: f64 = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / b.len() as f64;

    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / a.len() as f64;
    let var_b: f64 = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / b.len() as f64;

    let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();

    if se == 0.0 {
        return false;
    }

    let t = (mean_a - mean_b).abs() / se;

    // Approximate critical values for common confidence levels
    let critical = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        _ => 1.0,
    };

    t > critical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_from_measurements() {
        let data = vec![100.0, 110.0, 90.0, 105.0, 95.0];
        let stats = Stats::from_measurements(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 100.0).abs() < 0.1);
        assert!(stats.min >= 90.0);
        assert!(stats.max <= 110.0);
    }

    #[test]
    fn test_histogram() {
        let mut hist = LatencyHistogram::new();

        for i in 0..1000 {
            hist.record(100 + (i % 100));
        }

        assert_eq!(hist.count(), 1000);
        assert!(hist.min() >= 100);
        assert!(hist.max() < 200);
    }

    #[test]
    fn test_throughput_stats() {
        let stats = ThroughputStats::from_ops(1_000_000, 1_000_000_000); // 1M ops in 1 second
        assert!((stats.ops_per_sec - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_speedup_calculation() {
        let baseline = vec![100.0, 100.0, 100.0];
        let optimized = vec![10.0, 10.0, 10.0];
        let speedup = calculate_speedup(&baseline, &optimized);
        assert!((speedup - 10.0).abs() < 0.1);
    }
}
