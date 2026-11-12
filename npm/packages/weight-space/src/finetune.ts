/**
 * Outcome-driven fine-tuning: turn recorded trajectories into training rows, train with ruv-fann,
 * and keep the result only if a held-out check improves.
 */
import { FannRuntime, loadFann, meanSquaredError, trainWithinBudget } from './fann.js';
import { WeightSpaceError } from './model.js';
import type { SnapshotRecord, WeightRegistry } from './registry.js';

export interface Trajectory {
  /** Model input that was used. */
  input: ArrayLike<number>;
  /** Output the model (or the agent acting on it) produced. */
  output: ArrayLike<number>;
  /** Outcome score; at or above `successThreshold` counts as a success. */
  score: number;
  /** Corrected output for a failure. Failures without one carry no regression target and are skipped. */
  correction?: ArrayLike<number>;
}

export interface LearnOptions {
  /** Default 0.5. */
  successThreshold?: number;
  /** Fraction of usable rows held out for the accept/reject check (default 0.25, at least 1 row). */
  holdoutFraction?: number;
  /** Required relative MSE reduction on the holdout to accept (default 0.01 = 1%). */
  minImprovement?: number;
  epochs?: number;
  learningRate?: number;
  optimizer?: 'sgd' | 'adamw';
  seed?: number;
  runtime?: FannRuntime;
  note?: string;
}

export interface LearnResult {
  accepted: boolean;
  reason: string;
  parent: string;
  /** The saved child snapshot (only when accepted). */
  child?: SnapshotRecord;
  rows: { train: number; holdout: number; successes: number; corrections: number; skipped: number };
  holdoutMse: { before: number; after: number };
  trainLoss: { initial: number; final: number };
}

function vec(name: string, v: ArrayLike<number>, size: number): ArrayLike<number> {
  if (!v || v.length !== size) throw new WeightSpaceError(`${name} must have length ${size}`, 'ESHAPE');
  for (let i = 0; i < size; i++) {
    if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) throw new WeightSpaceError(`${name}[${i}] must be finite`, 'EINVALID');
  }
  return v;
}

/**
 * Fine-tune snapshot `id` on `trajectories`. Successful trajectories train toward the produced
 * output; failed ones train toward `correction` when given. A deterministic holdout split decides
 * acceptance: the child is saved (linked to `id`) only if holdout MSE drops by `minImprovement`.
 */
export async function learnFromOutcomes(
  registry: WeightRegistry,
  id: string,
  trajectories: Trajectory[],
  options: LearnOptions = {},
): Promise<LearnResult> {
  const parent = registry.get(id);
  if (!parent) throw new WeightSpaceError(`unknown snapshot '${id}'`, 'ENOTFOUND');
  if (!Array.isArray(trajectories)) throw new WeightSpaceError('trajectories must be an array', 'EINVALID');
  const threshold = options.successThreshold ?? 0.5;
  const inSize = parent.topology.sizes[0];
  const outSize = parent.topology.sizes[parent.topology.sizes.length - 1];

  const xs: ArrayLike<number>[] = [];
  const ys: ArrayLike<number>[] = [];
  let successes = 0, corrections = 0, skipped = 0;
  trajectories.forEach((t, i) => {
    if (!t || typeof t.score !== 'number' || !Number.isFinite(t.score)) throw new WeightSpaceError(`trajectory ${i} score must be finite`, 'EINVALID');
    const x = vec(`trajectory ${i} input`, t.input, inSize);
    if (t.score >= threshold) {
      xs.push(x); ys.push(vec(`trajectory ${i} output`, t.output, outSize)); successes++;
    } else if (t.correction) {
      xs.push(x); ys.push(vec(`trajectory ${i} correction`, t.correction, outSize)); corrections++;
    } else {
      skipped++;
    }
  });

  const base = { parent: id, rows: { train: 0, holdout: 0, successes, corrections, skipped }, holdoutMse: { before: NaN, after: NaN }, trainLoss: { initial: NaN, final: NaN } };
  if (xs.length < 2) return { ...base, accepted: false, reason: `need at least 2 usable trajectories, got ${xs.length}` };

  // Deterministic split: every k-th row (after a seeded shuffle) goes to the holdout.
  const order = xs.map((_, i) => i);
  let s = (options.seed ?? 7) >>> 0 || 1;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const nHold = Math.max(1, Math.round(order.length * (options.holdoutFraction ?? 0.25)));
  const holdIdx = order.slice(0, nHold);
  const trainIdx = order.slice(nHold);
  const pack = (idx: number[]) => {
    const inputs = new Float64Array(idx.length * inSize);
    const targets = new Float64Array(idx.length * outSize);
    idx.forEach((r, k) => {
      inputs.set(Array.from(xs[r]), k * inSize);
      targets.set(Array.from(ys[r]), k * outSize);
    });
    return { inputs, targets, rows: idx.length };
  };
  const train = pack(trainIdx);
  const hold = pack(holdIdx);

  const runtime = options.runtime ?? await loadFann();
  const before = meanSquaredError(runtime, parent.model, hold.inputs, hold.targets, hold.rows);
  const fit = trainWithinBudget(runtime, parent.model, train.inputs, train.targets, train.rows, {
    epochs: options.epochs ?? 50,
    learningRate: options.learningRate ?? 0.05,
    optimizer: options.optimizer ?? 'sgd',
    seed: options.seed,
  });
  const after = meanSquaredError(runtime, fit.model, hold.inputs, hold.targets, hold.rows);
  const minImprovement = options.minImprovement ?? 0.01;
  const result = {
    ...base,
    rows: { ...base.rows, train: train.rows, holdout: hold.rows },
    holdoutMse: { before, after },
    trainLoss: { initial: fit.initialLoss, final: fit.finalLoss },
  };
  if (!(after < before * (1 - minImprovement))) {
    return { ...result, accepted: false, reason: `holdout MSE ${after.toExponential(3)} did not improve on ${before.toExponential(3)} by ${minImprovement * 100}%` };
  }
  const child = await registry.commit(id, fit.model, {
    note: options.note,
    outcomes: [],
    detail: { holdoutMse: { before, after }, rows: result.rows, epochs: options.epochs ?? 50 },
  }, 'finetune');
  return { ...result, accepted: true, reason: 'holdout MSE improved', child };
}
