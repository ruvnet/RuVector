/**
 * Lazy loader for the ruv-fann WASM runtime, plus budget-aware training.
 *
 * ruv-fann is ESM-only. tsc's CommonJS output rewrites `import()` into `require()`, which
 * cannot load it, so the dynamic import is built at runtime (same pattern as ruvector's
 * neural-q module).
 */
import { DenseModel, LIMITS, WeightSpaceError, parameterCount, validateModel } from './model.js';

const dynamicImport = new Function('specifier', 'return import(specifier)') as (s: string) => Promise<any>;

export interface PreparedModel {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly maxRows: number;
  predict(inputs: Float64Array, rows?: number): Float64Array;
  predictInto(inputs: Float64Array, outputs: Float64Array, rows?: number): void;
  dispose(): void;
}

export interface FannTrainOptions {
  inputs: Float64Array;
  targets: Float64Array;
  rows: number;
  epochs: number;
  learningRate: number;
  optimizer?: 'sgd' | 'adamw';
}

export interface FannRuntime {
  readonly backend: 'wasm' | 'wasm-simd';
  prepare(model: DenseModel | string, options?: { maxRows?: number }): PreparedModel;
  train(model: DenseModel | string, options: FannTrainOptions): {
    model: DenseModel; initialLoss: number; finalLoss: number; epochs: number; parameterCount: number;
  };
}

let cached: Promise<FannRuntime> | null = null;

/**
 * Load ruv-fann once per process. Rejects with code `ENOFANN` if it cannot be imported, and
 * `ELIMITS` if the installed version reports different LIMITS than this package validates against.
 * `importer` lets tests simulate a missing package.
 */
export function loadFann(importer?: (specifier: string) => Promise<any>): Promise<FannRuntime> {
  if (importer) return load(importer);
  if (!cached) {
    cached = load(dynamicImport);
    cached.catch(() => { cached = null; });
  }
  return cached;
}

async function load(importer: (specifier: string) => Promise<any>): Promise<FannRuntime> {
  let mod: any;
  try {
    mod = await importer('ruv-fann');
  } catch (err) {
    throw new WeightSpaceError(`ruv-fann could not be loaded (${(err as Error).message}); run: npm i ruv-fann@0.1.0`, 'ENOFANN');
  }
  const limits = mod.LIMITS ?? {};
  for (const key of Object.keys(LIMITS) as (keyof typeof LIMITS)[]) {
    if (limits[key] !== LIMITS[key]) {
      throw new WeightSpaceError(`ruv-fann LIMITS.${key}=${limits[key]} differs from the expected ${LIMITS[key]}`, 'ELIMITS');
    }
  }
  return (await mod.loadRuntime({ simd: 'auto' })) as FannRuntime;
}

/** ruv-fann's training work estimate (ruv-fann 0.1.0 src/index.js). */
export function trainingWork(model: DenseModel, rows: number, epochs: number): number {
  const params = parameterCount(model);
  const inSize = model.layers[0].inputs;
  const outSize = model.layers[model.layers.length - 1].outputs;
  const backward = rows * (32 * params + 8 * (inSize + outSize) + 1);
  return backward * (epochs + 1) + 64 * params * epochs;
}

export interface TrainLoopOptions {
  epochs: number;
  learningRate: number;
  optimizer?: 'sgd' | 'adamw';
  /** Shuffle seed for mini-batches when the full set does not fit one call. */
  seed?: number;
  /** Fraction of ruv-fann's per-call work budget to use (default 0.9). */
  budgetFraction?: number;
}

export interface TrainLoopResult {
  model: DenseModel;
  initialLoss: number;
  finalLoss: number;
  calls: number;
  batchRows: number;
}

/**
 * Train for `epochs` passes over all rows, splitting into as many ruv-fann `train()` calls as its
 * per-call work budget requires. When the full set fits, several epochs run per call; otherwise
 * each call covers one mini-batch for one epoch. Note: each call starts a fresh optimiser, so with
 * `adamw` the moment estimates reset between calls; `sgd` (default) has no such state.
 */
export function trainWithinBudget(
  runtime: FannRuntime,
  start: DenseModel,
  inputs: Float64Array,
  targets: Float64Array,
  rows: number,
  options: TrainLoopOptions,
): TrainLoopResult {
  const model0 = validateModel(start);
  const inSize = model0.layers[0].inputs;
  const outSize = model0.layers[model0.layers.length - 1].outputs;
  if (!Number.isSafeInteger(rows) || rows < 1) throw new WeightSpaceError('rows must be a positive integer', 'ERANGE');
  if (inputs.length !== rows * inSize) throw new WeightSpaceError(`inputs must have length rows*${inSize}`, 'ESHAPE');
  if (targets.length !== rows * outSize) throw new WeightSpaceError(`targets must have length rows*${outSize}`, 'ESHAPE');
  if (!Number.isSafeInteger(options.epochs) || options.epochs < 1) throw new WeightSpaceError('epochs must be >= 1', 'ERANGE');
  const optimizer = options.optimizer ?? 'sgd';
  const budget = LIMITS.maxTrainingWork * (options.budgetFraction ?? 0.9);

  // Largest batch that fits one epoch in one call.
  let batchRows = Math.min(rows, LIMITS.maxRows);
  while (batchRows > 1 && trainingWork(model0, batchRows, 1) > budget) batchRows = Math.floor(batchRows / 2);
  if (trainingWork(model0, batchRows, 1) > budget) {
    throw new WeightSpaceError('model is too large to train one row within the ruv-fann work budget', 'EBUDGET');
  }

  let model = model0;
  let calls = 0;
  let initialLoss = NaN;
  let finalLoss = NaN;

  if (batchRows === rows) {
    // Whole set per call: pack as many epochs per call as the budget allows.
    let perCall = 1;
    while (perCall < LIMITS.maxEpochs && trainingWork(model0, rows, perCall + 1) <= budget) perCall++;
    let remaining = options.epochs;
    while (remaining > 0) {
      const epochs = Math.min(perCall, remaining);
      const r = runtime.train(model, { inputs, targets, rows, epochs, learningRate: options.learningRate, optimizer });
      if (calls === 0) initialLoss = r.initialLoss;
      finalLoss = r.finalLoss;
      model = r.model;
      remaining -= epochs;
      calls++;
    }
    return { model, initialLoss, finalLoss, calls, batchRows };
  }

  // Mini-batches: one epoch per call over a shuffled slice.
  const order = Array.from({ length: rows }, (_, i) => i);
  let s = (options.seed ?? 1) >>> 0 || 1;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const bIn = new Float64Array(batchRows * inSize);
  const bOut = new Float64Array(batchRows * outSize);
  let lossSum = 0;
  let lossRows = 0;
  for (let epoch = 0; epoch < options.epochs; epoch++) {
    for (let i = rows - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    lossSum = 0;
    lossRows = 0;
    for (let startRow = 0; startRow < rows; startRow += batchRows) {
      const n = Math.min(batchRows, rows - startRow);
      const xin = n === batchRows ? bIn : new Float64Array(n * inSize);
      const xout = n === batchRows ? bOut : new Float64Array(n * outSize);
      for (let r = 0; r < n; r++) {
        const src = order[startRow + r];
        xin.set(inputs.subarray(src * inSize, (src + 1) * inSize), r * inSize);
        xout.set(targets.subarray(src * outSize, (src + 1) * outSize), r * outSize);
      }
      const res = runtime.train(model, { inputs: xin, targets: xout, rows: n, epochs: 1, learningRate: options.learningRate, optimizer });
      if (calls === 0) initialLoss = res.initialLoss;
      lossSum += res.finalLoss * n;
      lossRows += n;
      model = res.model;
      calls++;
    }
    finalLoss = lossSum / lossRows;
  }
  return { model, initialLoss, finalLoss, calls, batchRows };
}

/** Mean squared error of `model` over `rows` examples, using one prepared network and `predictInto`. */
export function meanSquaredError(
  runtime: FannRuntime,
  model: DenseModel,
  inputs: Float64Array,
  targets: Float64Array,
  rows: number,
): number {
  const prepared = runtime.prepare(model, { maxRows: Math.min(Math.max(rows, 1), LIMITS.maxRows) });
  try {
    const inSize = prepared.inputSize;
    const outSize = prepared.outputSize;
    // One reusable output buffer; inputs are passed as zero-copy subarrays.
    let sum = 0;
    const scratch = new Float64Array(Math.min(prepared.maxRows, rows) * outSize);
    for (let start = 0; start < rows; start += prepared.maxRows) {
      const n = Math.min(prepared.maxRows, rows - start);
      const dst = n * outSize === scratch.length ? scratch : new Float64Array(n * outSize);
      prepared.predictInto(inputs.subarray(start * inSize, (start + n) * inSize), dst, n);
      for (let i = 0; i < n * outSize; i++) {
        const d = dst[i] - targets[start * outSize + i];
        sum += d * d;
      }
    }
    return sum / Math.max(1, rows * outSize);
  } finally {
    prepared.dispose();
  }
}
