/**
 * Neural Q-function for the learning engine ('neural-q').
 *
 * Tabular Q-learning stores one value per (state, action) string pair, so a state it has
 * never seen scores every action 0. This module fits a small dense network
 * (hashed state features -> one Q-value per action) on the experience buffer, which lets
 * routing generalise to unseen states that share tokens with seen ones (for example
 * `refactor:.tsx` after `edit:.tsx` and `refactor:.ts`).
 *
 * The network runs on the optional peer dependency `ruv-fann` (WASM, ESM-only). It is
 * loaded lazily with a real dynamic import; when it is not installed the engine keeps
 * using the tabular Q-table, and the buffer is still recorded so a later install can train.
 */

// tsc (module: commonjs) rewrites `import()` into `require()`, which cannot load an ESM-only
// package on Node 20. Build the dynamic import at runtime so it stays a real import().
const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<any>;

/** Number of hashed input features. */
export const FEATURE_DIM = 64;
const HIDDEN = 32;
/** Max (state, action) pairs kept per task; the oldest are evicted first. */
const MAX_PAIRS = 512;
/** Retrain after this many new observations (when the runtime is loaded). */
const TRAIN_EVERY = 16;
/** ruv-fann caps one synchronous train() call at 100M work units; stay under it. */
const WORK_BUDGET = 90_000_000;
const MAX_EPOCHS = 200;
const LEARNING_RATE = 0.01;

interface DenseLayer {
  inputs: number;
  outputs: number;
  weights: number[];
  bias: number[];
  activation: 'Tanh' | 'Linear';
  steepness: number;
}
export interface DenseModel {
  schema: 'ruv-fann.dense-layers/1';
  layers: DenseLayer[];
}
interface PreparedModel {
  predict(inputs: Float64Array, rows?: number): Float64Array;
  dispose(): void;
}
interface FannRuntime {
  readonly backend: string;
  prepare(model: DenseModel, options?: { maxRows?: number }): PreparedModel;
  train(model: DenseModel, options: {
    inputs: Float64Array; targets: Float64Array; rows: number;
    epochs: number; learningRate: number; optimizer?: 'sgd' | 'adamw';
  }): { model: DenseModel; initialLoss: number; finalLoss: number; epochs: number };
}

/** Serialised per-task state (stored inside LearningEngine.export()). */
export interface NeuralTaskState {
  vocab: string[];
  /** [state, action, meanTarget, count], oldest first. */
  pairs: [string, string, number, number][];
  model: DenseModel | null;
  pending: number;
  lastLoss: number | null;
}

let runtimePromise: Promise<FannRuntime | null> | null = null;

/**
 * Load ruv-fann once per process. Resolves to null when it is not installed or cannot load.
 * `importer` exists for tests that need to simulate a missing package.
 */
export function loadFannRuntime(
  importer: (specifier: string) => Promise<any> = dynamicImport,
): Promise<FannRuntime | null> {
  if (importer !== dynamicImport) return tryLoad(importer);
  if (!runtimePromise) runtimePromise = tryLoad(importer);
  return runtimePromise;
}

async function tryLoad(importer: (specifier: string) => Promise<any>): Promise<FannRuntime | null> {
  try {
    const mod = await importer('ruv-fann');
    return (await mod.loadRuntime({ simd: 'auto' })) as FannRuntime;
  } catch {
    return null;
  }
}

function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function addFeature(out: Float64Array, key: string, weight: number): void {
  const h = fnv1a(key);
  out[h % FEATURE_DIM] += (h & 0x80000000) ? -weight : weight;
}

/**
 * Signed feature hashing over tokens, dotted sub-tokens and character trigrams.
 * `fix:.ts` and `src/core/foo.test.ts` both produce useful overlap with related states.
 */
export function featurize(state: string, out: Float64Array = new Float64Array(FEATURE_DIM), offset = 0): Float64Array {
  const view = offset === 0 && out.length === FEATURE_DIM ? out : out.subarray(offset, offset + FEATURE_DIM);
  view.fill(0);
  const tokens = state.toLowerCase().split(/[\s/\\:|,;=]+/).filter(Boolean);
  for (const token of tokens) {
    addFeature(view, `t:${token}`, 1);
    const parts = token.split(/(?=\.)/).filter(Boolean);
    if (parts.length > 1) for (const p of parts) addFeature(view, `p:${p}`, 1);
    const padded = `^${token}$`;
    for (let i = 0; i + 3 <= padded.length; i++) addFeature(view, `g:${padded.slice(i, i + 3)}`, 0.5);
  }
  let norm = 0;
  for (let i = 0; i < FEATURE_DIM; i++) norm += view[i] * view[i];
  if (norm > 0) {
    const inv = 1 / Math.sqrt(norm);
    for (let i = 0; i < FEATURE_DIM; i++) view[i] *= inv;
  }
  return out;
}

/** Deterministic Xavier-uniform initialisation (seeded from the vocabulary). */
function initModel(outputs: number, seed: number): DenseModel {
  let s = seed >>> 0 || 1;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const layer = (inputs: number, outs: number, activation: 'Tanh' | 'Linear'): DenseLayer => {
    const limit = Math.sqrt(6 / (inputs + outs));
    return {
      inputs, outputs: outs, activation, steepness: 1,
      weights: Array.from({ length: inputs * outs }, () => (rand() * 2 - 1) * limit),
      bias: new Array(outs).fill(0),
    };
  };
  return { schema: 'ruv-fann.dense-layers/1', layers: [layer(FEATURE_DIM, HIDDEN, 'Tanh'), layer(HIDDEN, outputs, 'Linear')] };
}

function parameterCount(model: DenseModel): number {
  return model.layers.reduce((n, l) => n + l.inputs * l.outputs + l.outputs, 0);
}

/** Largest epoch count whose ruv-fann training work stays under WORK_BUDGET. */
function epochBudget(model: DenseModel, rows: number): number {
  const params = parameterCount(model);
  const outputs = model.layers[model.layers.length - 1].outputs;
  const perEpoch = rows * (32 * params + 8 * (FEATURE_DIM + outputs) + 1) + 64 * params;
  const epochs = Math.floor(WORK_BUDGET / perEpoch) - 1;
  return Math.max(1, Math.min(MAX_EPOCHS, epochs));
}

function emptyTask(): NeuralTaskState {
  return { vocab: [], pairs: [], model: null, pending: 0, lastLoss: null };
}

export class NeuralQ {
  private tasks = new Map<string, NeuralTaskState>();
  private prepared = new Map<string, PreparedModel>();
  private runtime: FannRuntime | null = null;

  /** Attach a loaded runtime; returns whether the neural path is now active. */
  attach(runtime: FannRuntime | null): boolean {
    this.runtime = runtime;
    return runtime !== null;
  }

  get backend(): string | null {
    return this.runtime ? this.runtime.backend : null;
  }

  /** Record a (state, action) -> target observation; retrains every TRAIN_EVERY observations. */
  observe(task: string, state: string, action: string, target: number): void {
    const t = this.tasks.get(task) ?? emptyTask();
    this.tasks.set(task, t);
    const idx = t.pairs.findIndex(p => p[0] === state && p[1] === action);
    if (idx >= 0) {
      const [, , mean, count] = t.pairs[idx];
      t.pairs.splice(idx, 1);
      t.pairs.push([state, action, mean + (target - mean) / (count + 1), count + 1]);
    } else {
      t.pairs.push([state, action, target, 1]);
      if (t.pairs.length > MAX_PAIRS) t.pairs.shift();
    }
    if (!t.vocab.includes(action)) t.vocab.push(action);
    t.pending++;
    if (this.runtime && t.pending >= TRAIN_EVERY) this.train(task);
  }

  /** Fit the network on the task's buffer. Returns the final loss, or null if not trainable. */
  train(task: string, rounds = 1): number | null {
    const t = this.tasks.get(task);
    if (!this.runtime || !t || t.pairs.length === 0) return null;
    const outputs = t.vocab.length;
    const last = t.model?.layers[t.model.layers.length - 1];
    if (!t.model || !last || last.outputs !== outputs) t.model = initModel(outputs, fnv1a(t.vocab.join('\u0000')));

    const rows = t.pairs.length;
    const inputs = new Float64Array(rows * FEATURE_DIM);
    t.pairs.forEach(([state], r) => featurize(state, inputs, r * FEATURE_DIM));
    const actionIndex = new Map(t.vocab.map((a, i) => [a, i] as const));

    for (let round = 0; round < rounds; round++) {
      // Targets: the network's own output with only the taken action's entry replaced,
      // so untaken actions contribute zero error.
      const prepared = this.runtime.prepare(t.model, { maxRows: rows });
      let targets: Float64Array;
      try {
        targets = prepared.predict(inputs, rows);
      } finally {
        prepared.dispose();
      }
      t.pairs.forEach(([, action, mean], r) => { targets[r * outputs + (actionIndex.get(action) as number)] = mean; });
      const result = this.runtime.train(t.model, {
        inputs, targets, rows, epochs: epochBudget(t.model, rows), learningRate: LEARNING_RATE, optimizer: 'adamw',
      });
      t.model = result.model;
      t.lastLoss = result.finalLoss;
    }
    t.pending = 0;
    this.prepared.get(task)?.dispose();
    this.prepared.delete(task);
    return t.lastLoss;
  }

  /** Q-values for the task's known actions, or null when no trained model/runtime is available. */
  predict(task: string, state: string): Map<string, number> | null {
    const t = this.tasks.get(task);
    if (!this.runtime || !t || t.pairs.length === 0) return null;
    // Train on first use (e.g. a buffer recorded before ruv-fann was installed) or when the
    // action vocabulary outgrew the output layer; otherwise observe() keeps the model fresh.
    const outputs = t.model?.layers[t.model.layers.length - 1].outputs;
    if (outputs !== t.vocab.length) this.train(task);
    if (!t.model) return null;
    let prepared = this.prepared.get(task);
    if (!prepared) {
      prepared = this.runtime.prepare(t.model as DenseModel, { maxRows: 1 });
      this.prepared.set(task, prepared);
    }
    const q = prepared.predict(featurize(state), 1);
    return new Map(t.vocab.map((a, i) => [a, q[i]] as const));
  }

  hasModel(task: string): boolean {
    return !!this.tasks.get(task)?.model;
  }

  info(task: string): { pairs: number; actions: number; trained: boolean; lastLoss: number | null } {
    const t = this.tasks.get(task) ?? emptyTask();
    return { pairs: t.pairs.length, actions: t.vocab.length, trained: !!t.model, lastLoss: t.lastLoss };
  }

  export(): Record<string, NeuralTaskState> {
    return Object.fromEntries(this.tasks);
  }

  import(data: Record<string, NeuralTaskState> | undefined): void {
    this.clear();
    for (const [task, s] of Object.entries(data || {})) {
      if (!s || !Array.isArray(s.pairs) || !Array.isArray(s.vocab)) continue;
      this.tasks.set(task, {
        vocab: s.vocab.filter(a => typeof a === 'string'),
        pairs: s.pairs.filter(p => Array.isArray(p) && p.length === 4 && Number.isFinite(p[2])).slice(-MAX_PAIRS),
        model: s.model && s.model.schema === 'ruv-fann.dense-layers/1' ? s.model : null,
        pending: Number.isFinite(s.pending) ? s.pending : 0,
        lastLoss: Number.isFinite(s.lastLoss as number) ? s.lastLoss : null,
      });
    }
  }

  clear(): void {
    for (const p of this.prepared.values()) p.dispose();
    this.prepared.clear();
    this.tasks.clear();
  }
}
