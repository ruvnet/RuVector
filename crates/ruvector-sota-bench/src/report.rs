//! Benchmark reporting — console tables, JSON, CSV, leaderboard comparison.
use crate::metrics::BenchScore;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardRow {
    pub rank: usize,
    pub system: String,
    pub dataset: String,
    pub recall_at_10: f64,
    pub qps: f64,
    pub memory_mb: f64,
    pub p99_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub generated_at: String,
    pub git_sha: String,
    pub scores: Vec<BenchScore>,
    pub leaderboard: Vec<LeaderboardRow>,
    pub sota_claims: Vec<String>,
}

impl BenchReport {
    pub fn new(scores: Vec<BenchScore>) -> Self {
        let git_sha = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let sota_claims: Vec<String> = scores
            .iter()
            .filter(|s| s.sota)
            .map(|s| {
                format!(
                    "{} on {}: recall@10={:.4} qps={:.0}",
                    s.index, s.dataset, s.recall.recall_at_10, s.qps
                )
            })
            .collect();

        // Sort into leaderboard by darwin_score descending
        let mut leaderboard: Vec<LeaderboardRow> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| LeaderboardRow {
                rank: i + 1,
                system: s.index.clone(),
                dataset: s.dataset.clone(),
                recall_at_10: s.recall.recall_at_10,
                qps: s.qps,
                memory_mb: s.memory_mb,
                p99_us: s.latency.p99_us,
            })
            .collect();
        leaderboard.sort_by(|a, b| b.recall_at_10.partial_cmp(&a.recall_at_10).unwrap());
        for (i, row) in leaderboard.iter_mut().enumerate() {
            row.rank = i + 1;
        }

        Self {
            generated_at: chrono::Utc::now().to_rfc3339(),
            git_sha,
            scores,
            leaderboard,
            sota_claims,
        }
    }

    pub fn print_table(&self) {
        println!("\n╔══ RuVector SOTA Benchmark Report ══════════════════════════════════╗");
        println!("  Generated: {}  SHA: {}", self.generated_at, self.git_sha);
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!(
            "  {:<24} {:<24} {:>10} {:>10} {:>9}",
            "Index", "Dataset", "Recall@10", "QPS", "p99 µs"
        );
        println!("  {}", "─".repeat(80));
        for s in &self.scores {
            let sota_mark = if s.sota { " ★SOTA" } else { "" };
            println!(
                "  {:<24} {:<24} {:>9.4} {:>10.0} {:>8.1}{}",
                s.index, s.dataset, s.recall.recall_at_10, s.qps, s.latency.p99_us, sota_mark
            );
        }
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        if self.sota_claims.is_empty() {
            println!("  No SOTA claims this run.");
        } else {
            println!("  SOTA claims (recall@10 ≥ 0.95 AND QPS ≥ 80% of HNSWlib):");
            for c in &self.sota_claims {
                println!("    ★ {c}");
            }
        }
        println!("╚═══════════════════════════════════════════════════════════════════╝\n");
    }

    pub fn save_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let f = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(f, self)?;
        Ok(())
    }
}
