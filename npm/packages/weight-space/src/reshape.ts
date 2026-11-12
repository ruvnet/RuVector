/**
 * Learned query reshaping. EXPERIMENTAL and OFF: exported only from `@ruvector/weight-space/experimental`,
 * never used by the main API or by ruvector, because it failed its retrieval gate (see README).
 *
 * A residual bottleneck MLP, `q' = normalise(q + W2 * tanh(W1 * q))`, trained with ruv-fann on
 * feedback (query, relevant, irrelevant[]) so that q' moves toward the relevant item.
 * The output layer starts at zero, so an untrained reshaper is the identity.
 */
import { FannRuntime, PreparedModel, loadFann, trainWithinBudget } from './fann.js';
import { DenseModel, WeightSpaceError, createMlp, validateModel } from './model.js';
import { checkVector } from './store.js';

export interface FeedbackPair {
  query: ArrayLike<number>;
  relevant: ArrayLike<number>;
  irrelevant?: ArrayLike<number>[];
}

export interface ReshapeTrainOptions {
  epochs?: number;
  learningRate?: number;
  /** How far to push away from the mean irrelevant vector (default 0.5). */
  negativeWeight?: number;
  seed?: number;
}

/** Anything with a cosine `search(vector, k)` (e.g. a VectorStore). */
export interface SearchableIndex {
  search(vector: ArrayLike<number>, k: number): Promise<{ id: string; similarity: number }[]>;
}

function unit(v: ArrayLike<number>, out: Float64Array, off: number, d: number): void {
  let n = 0;
  for (let i = 0; i < d; i++) n += v[i] * v[i];
  const inv = n > 0 ? 1 / Math.sqrt(n) : 0;
  for (let i = 0; i < d; i++) out[off + i] = v[i] * inv;
}

export class QueryReshaper {
  private prepared: PreparedModel | null = null;
  private readonly inBuf: Float64Array;
  private readonly outBuf: Float64Array;

  private constructor(private runtime: FannRuntime, private current: DenseModel, readonly dimensions: number) {
    this.inBuf = new Float64Array(dimensions);
    this.outBuf = new Float64Array(dimensions);
  }

  /** New identity reshaper for `dimensions`-d embeddings with a `hidden`-unit bottleneck. */
  static async create(dimensions: number, options: { hidden?: number; seed?: number; runtime?: FannRuntime } = {}): Promise<QueryReshaper> {
    const runtime = options.runtime ?? await loadFann();
    const base = createMlp([dimensions, options.hidden ?? 16, dimensions], { hidden: 'Tanh', output: 'Linear', seed: options.seed ?? 1 });
    const layers = base.layers.map((l, i) => (i === base.layers.length - 1 ? { ...l, weights: new Array(l.weights.length).fill(0) } : l));
    return new QueryReshaper(runtime, validateModel({ schema: base.schema, layers }), dimensions);
  }

  /** Restore a reshaper from its residual model (e.g. one saved in a WeightRegistry). */
  static async fromModel(model: DenseModel, options: { runtime?: FannRuntime } = {}): Promise<QueryReshaper> {
    const m = validateModel(model);
    const d = m.layers[0].inputs;
    if (m.layers[m.layers.length - 1].outputs !== d) throw new WeightSpaceError('reshaper model must map d -> d', 'ESHAPE');
    return new QueryReshaper(options.runtime ?? await loadFann(), m, d);
  }

  /** The residual model (input -> delta). */
  get model(): DenseModel {
    return this.current;
  }

  /** Train on feedback pairs. Returns the ruv-fann loss before and after. */
  train(pairs: FeedbackPair[], options: ReshapeTrainOptions = {}): { initialLoss: number; finalLoss: number; calls: number } {
    const d = this.dimensions;
    if (!Array.isArray(pairs) || pairs.length === 0) throw new WeightSpaceError('pairs must be a non-empty array', 'EINVALID');
    const neg = options.negativeWeight ?? 0.5;
    const inputs = new Float64Array(pairs.length * d);
    const targets = new Float64Array(pairs.length * d);
    const rel = new Float64Array(d);
    const irr = new Float64Array(d);
    const tmp = new Float64Array(d);
    pairs.forEach((p, r) => {
      checkVector(p.query, d, `pairs[${r}].query`);
      checkVector(p.relevant, d, `pairs[${r}].relevant`);
      unit(p.query, inputs, r * d, d);
      unit(p.relevant, rel, 0, d);
      irr.fill(0);
      const negs = p.irrelevant ?? [];
      for (const n of negs) {
        checkVector(n, d, `pairs[${r}].irrelevant`);
        unit(n, tmp, 0, d);
        for (let i = 0; i < d; i++) irr[i] += tmp[i] / negs.length;
      }
      for (let i = 0; i < d; i++) {
        const goal = rel[i] + (negs.length ? neg * (rel[i] - irr[i]) : 0);
        targets[r * d + i] = goal - inputs[r * d + i];
      }
    });
    const fit = trainWithinBudget(this.runtime, this.current, inputs, targets, pairs.length, {
      epochs: options.epochs ?? 30,
      learningRate: options.learningRate ?? 0.05,
      optimizer: 'sgd',
      seed: options.seed,
    });
    this.current = fit.model;
    this.dispose();
    return { initialLoss: fit.initialLoss, finalLoss: fit.finalLoss, calls: fit.calls };
  }

  /** Reshape one query. Returns a new unit-length Float64Array. */
  reshape(query: ArrayLike<number>): Float64Array {
    const out = new Float64Array(this.dimensions);
    this.reshapeInto(query, out);
    return out;
  }

  /** Allocation-free reshape into `out` (length = dimensions). */
  reshapeInto(query: ArrayLike<number>, out: Float64Array): void {
    const d = this.dimensions;
    checkVector(query, d, 'query');
    if (!(out instanceof Float64Array) || out.length !== d) throw new WeightSpaceError(`out must be a Float64Array of length ${d}`, 'ESHAPE');
    if (!this.prepared) this.prepared = this.runtime.prepare(this.current, { maxRows: 1 });
    unit(query, this.inBuf, 0, d);
    this.prepared.predictInto(this.inBuf, this.outBuf, 1);
    let n = 0;
    for (let i = 0; i < d; i++) {
      out[i] = this.inBuf[i] + this.outBuf[i];
      n += out[i] * out[i];
    }
    const inv = n > 0 ? 1 / Math.sqrt(n) : 0;
    for (let i = 0; i < d; i++) out[i] *= inv;
  }

  /** Search `index` with the reshaped query. */
  async searchWithReshape(index: SearchableIndex, query: ArrayLike<number>, k: number) {
    return index.search(this.reshape(query), k);
  }

  /** Release the prepared WASM network (it is re-created lazily on the next reshape). */
  dispose(): void {
    if (this.prepared) this.prepared.dispose();
    this.prepared = null;
  }
}

/** Functional form of `reshaper.reshape(queryVec)`. */
export function reshape(reshaper: QueryReshaper, queryVec: ArrayLike<number>): Float64Array {
  return reshaper.reshape(queryVec);
}

/** Functional form of `reshaper.searchWithReshape(index, q, k)`. */
export function searchWithReshape(reshaper: QueryReshaper, index: SearchableIndex, q: ArrayLike<number>, k: number) {
  return reshaper.searchWithReshape(index, q, k);
}
