/**
 * Boundary helpers: argument validation before crossing the WASM ABI, and
 * mapping of the module's snake_case JSON onto the camelCase public types.
 *
 * The raw snake_case keys are kept alongside the camelCase ones so callers
 * that read them from earlier versions keep working.
 */
import type {
  TrainResult,
  AcceptanceManifest,
  AcceptanceModeResult,
  CycleMetrics,
  PolicyState,
  SkipModeStats,
} from './types';

const I32_MAX = 2 ** 31 - 1;

/** Throw a RangeError unless `value` is an integer in [min, I32_MAX]. */
export function checkInt(name: string, value: number, min: number, max = I32_MAX): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer in [${min}, ${max}], got ${String(value)}`);
  }
  return value;
}

/** Validate a seed and split it into (lo, hi) 32-bit halves. */
export function splitSeed(seed?: bigint | number): [number, number] {
  let s: bigint;
  if (seed === undefined) {
    s = BigInt(Math.floor(Math.random() * 2 ** 32)) | (BigInt(Math.floor(Math.random() * 2 ** 32)) << 32n);
  } else if (typeof seed === 'bigint') {
    s = seed;
  } else {
    if (!Number.isInteger(seed)) {
      throw new RangeError(`seed must be an integer or a bigint, got ${String(seed)}`);
    }
    s = BigInt(seed);
  }
  return [Number(s & 0xFFFFFFFFn), Number((s >> 32n) & 0xFFFFFFFFn)];
}

type Raw = Record<string, any>;

export function toTrainResult(raw: Raw): TrainResult {
  return {
    ...raw,
    trained: raw.trained,
    correct: raw.correct,
    accuracy: raw.accuracy,
    patternsLearned: raw.patterns_learned,
  };
}

function toCycle(raw: Raw): CycleMetrics {
  return {
    ...raw,
    cycle: raw.cycle,
    accuracy: raw.accuracy,
    costPerSolve: raw.cost_per_solve,
    noiseAccuracy: raw.noise_accuracy,
    violations: raw.violations,
    patternsLearned: raw.patterns_learned,
  };
}

function toMode(raw: Raw): AcceptanceModeResult {
  const cycles: CycleMetrics[] = (raw.cycles ?? []).map(toCycle);
  return {
    ...raw,
    passed: raw.passed,
    finalAccuracy: cycles.length > 0 ? cycles[cycles.length - 1].accuracy : 0,
    cycles,
    accuracyMaintained: raw.accuracy_maintained,
    costImproved: raw.cost_improved,
    robustnessImproved: raw.robustness_improved,
    zeroViolations: raw.zero_violations,
    dimensionsImproved: raw.dimensions_improved,
  };
}

export function toManifest(raw: Raw): AcceptanceManifest {
  return {
    version: raw.version,
    modeA: toMode(raw.mode_a),
    modeB: toMode(raw.mode_b),
    modeC: toMode(raw.mode_c),
    allPassed: raw.all_passed,
    witnessEntries: raw.witness_entries,
    witnessChainBytes: raw.witness_chain_bytes,
  };
}

function toArmStats(raw: Raw): SkipModeStats {
  return {
    ...raw,
    attempts: raw.attempts,
    successes: raw.successes,
    totalSteps: raw.total_steps,
    alphaSafety: raw.alpha_safety,
    betaSafety: raw.beta_safety,
    costEma: raw.cost_ema,
    earlyCommitWrongs: raw.early_commit_wrongs,
  };
}

export function toPolicyState(raw: Raw): PolicyState {
  // Bucket and arm names are map keys, not fields: keep them as-is.
  const contextStats: PolicyState['contextStats'] = {};
  for (const [bucket, arms] of Object.entries<Raw>(raw.context_stats ?? {})) {
    contextStats[bucket] = {};
    for (const [arm, stats] of Object.entries<Raw>(arms)) {
      contextStats[bucket][arm] = toArmStats(stats);
    }
  }
  return {
    ...raw,
    contextStats,
    earlyCommitPenalties: raw.early_commit_penalties,
    earlyCommitsTotal: raw.early_commits_total,
    earlyCommitsWrong: raw.early_commits_wrong,
    prepass: raw.prepass,
    speculativeAttempts: raw.speculative_attempts,
    speculativeArm2Wins: raw.speculative_arm2_wins,
  };
}
