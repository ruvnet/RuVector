/**
 * Darwin Mode scorePolicy for RuVector SOTA benchmarks (ADR-266).
 *
 * This policy drives autonomous ANN parameter evolution by scoring
 * each variant's benchmark output against the Darwin score function:
 *
 *   score = 0.40 × recall@10
 *         + 0.30 × log(QPS / baseline_QPS).clamp(0, 1)
 *         + 0.20 × (1 − memory_mb / baseline_mb).max(0)
 *         + 0.10 × (1 − p99_ms / baseline_ms).max(0)
 *
 * The policy reads the JSON report produced by `sota-all --json` and
 * returns the highest darwin_score found, normalized to [0, 1].
 *
 * Baselines (HNSWlib reference on SIFT-128, single thread, commodity HW):
 *   QPS: 500   memory: 200 MB   p99: 5 ms
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as child_process from "node:child_process";
import type { RunTrace } from "../src/types.js";

// ── Baselines (ADR-265 §4) ──────────────────────────────────────────────────
const BASELINE_QPS    = 500;
const BASELINE_MEM_MB = 200;
const BASELINE_P99_MS = 5;

// ── Minimum thresholds to claim SOTA ────────────────────────────────────────
const MIN_RECALL_FOR_SOTA = 0.95;
const MIN_QPS_RATIO       = 0.80;   // must be ≥ 80% of baseline QPS

interface BenchScore {
  index: string;
  dataset: string;
  recall: { recall_at_10: number };
  qps: number;
  memory_mb: number;
  latency: { p99_us: number };
  darwin_score: number;
  sota: boolean;
}

interface BenchReport {
  scores: BenchScore[];
  sota_claims: string[];
}

function darwinScore(
  recall10: number,
  qps: number,
  memMb: number,
  p99Us: number,
): number {
  const qpsTerm = Math.min(1, Math.max(0, Math.log(qps / BASELINE_QPS)));
  const memTerm = Math.max(0, 1 - memMb / BASELINE_MEM_MB);
  const latTerm = Math.max(0, 1 - (p99Us / 1000) / BASELINE_P99_MS);
  return 0.40 * recall10 + 0.30 * qpsTerm + 0.20 * memTerm + 0.10 * latTerm;
}

/**
 * Score a variant by running the SOTA benchmark suite.
 *
 * Called by Darwin Mode after each mutation. Returns a score in [0, 1].
 * Higher score → more fit variant → more likely to be selected for next gen.
 */
export async function scoreVariant(traces: RunTrace[]): Promise<number> {
  // Check if the benchmark binary exists
  const binPath = path.resolve(
    import.meta.dirname ?? ".",
    "../../../../target/release/sota-all",
  );

  const reportPath = `/tmp/ruvector-darwin-score-${Date.now()}.json`;

  try {
    // Run smoke benchmark (fast, deterministic)
    child_process.execSync(
      `${binPath} --smoke --no-hybrid --no-matryoshka --json ${reportPath} --ef-search 100`,
      { timeout: 60_000, stdio: "pipe" },
    );
  } catch {
    // Benchmark binary not built or failed — fall back to trace-based scoring
    return scoreFromTraces(traces);
  }

  try {
    const report: BenchReport = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    fs.rmSync(reportPath, { force: true });

    if (!report.scores?.length) return 0;

    // Return the maximum darwin_score across all benchmark runs
    const best = Math.max(...report.scores.map((s) => s.darwin_score));
    const sotaBonus = report.sota_claims.length > 0 ? 0.05 : 0;
    return Math.min(1, best + sotaBonus);
  } catch {
    return scoreFromTraces(traces);
  }
}

/**
 * Fallback: score from test traces when the benchmark binary isn't available.
 * Uses test pass rate × coverage heuristic as a proxy for ANN quality.
 */
function scoreFromTraces(traces: RunTrace[]): number {
  if (!traces.length) return 0;
  const passed = traces.filter((t) => t.exitCode === 0).length;
  const passRate = passed / traces.length;
  // Penalise slow traces (proxy for p99 latency degradation)
  const avgMs = traces.reduce((s, t) => s + (t.durationMs ?? 0), 0) / traces.length;
  const latencyPenalty = Math.min(0.3, avgMs / 300_000); // cap at 5 min
  return Math.max(0, passRate - latencyPenalty);
}

/**
 * Extract the best metric summary from the last benchmark run.
 * Used by Darwin Mode to populate the leaderboard in its archive.
 */
export function extractMetrics(reportPath: string): Record<string, number> {
  try {
    const report: BenchReport = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    const scores = report.scores ?? [];
    if (!scores.length) return {};
    const best = scores.reduce((a, b) => (a.darwin_score > b.darwin_score ? a : b));
    return {
      recall_at_10:  best.recall.recall_at_10,
      qps:           best.qps,
      memory_mb:     best.memory_mb,
      p99_us:        best.latency.p99_us,
      darwin_score:  best.darwin_score,
      sota_claims:   report.sota_claims.length,
    };
  } catch {
    return {};
  }
}
