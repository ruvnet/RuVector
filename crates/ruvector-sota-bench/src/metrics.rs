//! Benchmark metrics: recall, latency, memory, throughput.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMetrics {
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
}

impl LatencyMetrics {
    pub fn from_nanos(mut ns: Vec<u128>) -> Self {
        ns.sort_unstable();
        let n = ns.len();
        let p = |pct: f64| ns[(pct * (n - 1) as f64) as usize] as f64 / 1_000.0;
        Self {
            mean_us: ns.iter().sum::<u128>() as f64 / n as f64 / 1_000.0,
            p50_us: p(0.50),
            p95_us: p(0.95),
            p99_us: p(0.99),
            p999_us: p(0.999),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchScore {
    pub index: String,
    pub dataset: String,
    pub recall: RecallMetrics,
    pub latency: LatencyMetrics,
    pub qps: f64,
    pub build_secs: f64,
    pub memory_mb: f64,
    pub darwin_score: f64,
    pub sota: bool,
    pub params: std::collections::HashMap<String, String>,
}
