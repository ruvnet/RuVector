import { createHash } from "node:crypto";

export interface PairedDecision {
  meanDelta: number;
  lower95: number;
  upper95: number;
  outcome: "pass" | "fail" | "inconclusive";
  samples: number;
}

function rng(seed: string): () => number {
  let state = Number.parseInt(createHash("sha256").update(seed).digest("hex").slice(0, 8), 16) >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function quantile(sorted: number[], probability: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(probability * sorted.length))] ?? 0;
}

export function pairedBootstrapDecision(
  baseline: readonly number[],
  candidate: readonly number[],
  minimumEffect = 0.005,
  bootstrapSamples = 2_000,
): PairedDecision {
  if (baseline.length !== candidate.length || baseline.length < 2) {
    return { meanDelta: 0, lower95: -Infinity, upper95: Infinity, outcome: "inconclusive", samples: 0 };
  }
  const deltas = candidate.map((value, index) => value - baseline[index]!);
  const random = rng(JSON.stringify({ baseline, candidate, bootstrapSamples }));
  const means: number[] = [];
  for (let sample = 0; sample < bootstrapSamples; sample += 1) {
    let total = 0;
    for (let index = 0; index < deltas.length; index += 1) {
      total += deltas[Math.floor(random() * deltas.length)]!;
    }
    means.push(total / deltas.length);
  }
  means.sort((a, b) => a - b);
  const meanDelta = deltas.reduce((sum, value) => sum + value, 0) / deltas.length;
  const lower95 = quantile(means, 0.025);
  const upper95 = quantile(means, 0.975);
  const outcome = lower95 >= minimumEffect ? "pass" : upper95 <= 0 ? "fail" : "inconclusive";
  return { meanDelta, lower95, upper95, outcome, samples: bootstrapSamples };
}
