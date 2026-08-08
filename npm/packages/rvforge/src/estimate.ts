/**
 * Pre-submission estimate: build time, output size, and cost.
 *
 * The CLI shows this before uploading anything (requirements §5.6). Numbers
 * come from the published GitHub-hosted runner pricing and the performance
 * targets in ADR-283 §8, so they are planning figures, not a quote.
 */

import {
  TARGET_ARTIFACTS,
  TARGET_WORKER,
  type BuildTarget,
  type PackagingMode,
  type WorkerFamily,
} from './types';

/** USD per runner-minute (GitHub-hosted, ADR-283 §1). */
export const RUNNER_COST_PER_MINUTE: Readonly<Record<WorkerFamily, number>> = Object.freeze({
  linux: 0.006,
  windows: 0.01,
  macos: 0.062,
});

/** Build-time targets: cached under 3 min/platform, uncached under 10. */
export const CACHED_MINUTES = 3;
export const UNCACHED_MINUTES = 10;

/** Package overhead ceilings from ADR-283 §8, in bytes. */
export const SHARED_READER_OVERHEAD_BYTES = 30 * 1024 * 1024;
export const EMBEDDED_OVERHEAD_BYTES = 50 * 1024 * 1024;

export interface TargetEstimate {
  target: BuildTarget;
  worker: WorkerFamily;
  minutes: number;
  costUsd: number;
  artifacts: string[];
  /** Upper bound on the bytes this target's artifacts occupy in total. */
  outputBytes: number;
}

export interface BuildEstimate {
  cached: boolean;
  targets: TargetEstimate[];
  /** Wall-clock minutes assuming targets build concurrently (§7.3). */
  wallClockMinutes: number;
  totalRunnerMinutes: number;
  totalCostUsd: number;
  totalOutputBytes: number;
}

export interface EstimateInput {
  targets: readonly BuildTarget[];
  packagingMode: PackagingMode;
  rvfSize: number;
  /** Whether dependency caches are expected to be warm. */
  cached?: boolean;
}

/** Per-target overhead added on top of the RVF, by packaging mode. */
function overheadBytes(mode: PackagingMode): number {
  return mode === 'shared-reader' ? SHARED_READER_OVERHEAD_BYTES : EMBEDDED_OVERHEAD_BYTES;
}

/** Bytes the RVF itself contributes to one artifact. */
function payloadBytes(mode: PackagingMode, rvfSize: number): number {
  // Thin and shared-reader installers ship a locator, not the model.
  return mode === 'embedded' ? rvfSize : 0;
}

/** Estimate build time, output size, and cost. Pure and deterministic. */
export function estimateBuild(input: EstimateInput): BuildEstimate {
  const cached = input.cached ?? false;
  const minutes = cached ? CACHED_MINUTES : UNCACHED_MINUTES;
  const perArtifact = overheadBytes(input.packagingMode) + payloadBytes(input.packagingMode, input.rvfSize);

  const targets: TargetEstimate[] = [...input.targets].sort().map((target) => {
    const worker = TARGET_WORKER[target];
    const artifacts = [...TARGET_ARTIFACTS[target]];
    return {
      target,
      worker,
      minutes,
      costUsd: round(minutes * RUNNER_COST_PER_MINUTE[worker], 4),
      artifacts,
      outputBytes: perArtifact * artifacts.length,
    };
  });

  const totalRunnerMinutes = targets.reduce((sum, t) => sum + t.minutes, 0);
  return {
    cached,
    targets,
    wallClockMinutes: targets.length > 0 ? minutes : 0,
    totalRunnerMinutes,
    totalCostUsd: round(
      targets.reduce((sum, t) => sum + t.costUsd, 0),
      4,
    ),
    totalOutputBytes: targets.reduce((sum, t) => sum + t.outputBytes, 0),
  };
}

const round = (value: number, places: number): number => {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
};

/** Human-readable byte size, e.g. `1.4 GB`. */
export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}
