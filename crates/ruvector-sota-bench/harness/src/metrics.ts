export interface BenchScore {
  index: string;
  dataset: string;
  recall: { recall_at_10: number };
  qps: number;
  memory_mb: number;
  latency: { p99_us: number };
  darwin_score?: number;
  sota?: boolean;
  build_secs?: number;
  params?: Record<string, string>;
}

export interface BenchReport {
  scores: BenchScore[];
  sota_claims?: string[];
  generated_at?: string;
  git_sha?: string;
  leaderboard?: unknown[];
}

export interface AggregateMetrics {
  primary: number;
  recallAt10: number;
  qps: number;
  memoryMb: number;
  p99Us: number;
  regressions: string[];
}

const BASELINE_QPS = 500;
const BASELINE_MEMORY_MB = 200;
const BASELINE_P99_MS = 5;

function finite(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`invalid non-negative metric: ${name}`);
  }
  return value;
}

export function darwinScore(score: BenchScore): number {
  const recall = finite(score.recall?.recall_at_10, "recall_at_10");
  const qps = finite(score.qps, "qps");
  const memory = finite(score.memory_mb, "memory_mb");
  const p99Us = finite(score.latency?.p99_us, "p99_us");
  const qpsTerm = Math.min(1, Math.max(0, Math.log(Math.max(qps, 1) / BASELINE_QPS)));
  const memoryTerm = Math.max(0, 1 - memory / BASELINE_MEMORY_MB);
  const latencyTerm = Math.max(0, 1 - p99Us / 1000 / BASELINE_P99_MS);
  return Math.max(0, Math.min(1,
    0.4 * recall + 0.3 * qpsTerm + 0.2 * memoryTerm + 0.1 * latencyTerm,
  ));
}

export function parseBenchReport(value: unknown): BenchReport {
  if (!value || typeof value !== "object" || !Array.isArray((value as BenchReport).scores)) {
    throw new Error("benchmark report must contain a scores array");
  }
  const report = value as BenchReport;
  for (const score of report.scores) {
    if (!score || typeof score !== "object") throw new Error("invalid benchmark score");
    darwinScore(score);
  }
  if (report.scores.length === 0) throw new Error("benchmark report has no scores");
  return report;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/** Aggregate every run, rather than selecting the best configuration. */
export function aggregateReports(reports: BenchReport[]): AggregateMetrics {
  const scores = reports.flatMap((report) => report.scores);
  if (scores.length === 0) throw new Error("cannot aggregate an empty report set");
  const recallAt10 = median(scores.map((score) => score.recall.recall_at_10));
  const qps = median(scores.map((score) => score.qps));
  const memoryMb = Math.max(...scores.map((score) => score.memory_mb));
  const p99Us = median(scores.map((score) => score.latency.p99_us));
  const primary = median(scores.map(darwinScore));
  const regressions: string[] = [];
  if (recallAt10 < 0.95) regressions.push("recall_below_0.95");
  if (qps < BASELINE_QPS * 0.8) regressions.push("qps_below_baseline_floor");
  return { primary, recallAt10, qps, memoryMb, p99Us, regressions };
}
