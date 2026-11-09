/**
 * Attention Fallbacks - Safe wrapper around @ruvector/attention with automatic array conversion
 *
 * This wrapper handles the array type conversion automatically, allowing users
 * to pass either regular arrays or Float32Arrays.
 *
 * @ruvector/attention requires Float32Array inputs.
 * This wrapper handles the conversion automatically.
 */

// Lazy load to avoid import errors if not installed
let attentionModule: any = null;
let loadError: Error | null = null;

function getAttentionModule() {
  if (attentionModule) return attentionModule;
  if (loadError) throw loadError;

  try {
    attentionModule = require('@ruvector/attention');
    return attentionModule;
  } catch (e: any) {
    loadError = new Error(
      `@ruvector/attention is not installed or failed to load: ${e.message}\n` +
      `Install with: npm install @ruvector/attention`
    );
    throw loadError;
  }
}

/**
 * Convert any array-like input to Float32Array
 */
function toFloat32Array(input: number[] | Float32Array | Float64Array): Float32Array {
  if (input instanceof Float32Array) {
    return input;
  }
  return new Float32Array(input);
}

/**
 * Convert nested arrays to Float32Arrays
 */
function toFloat32Arrays(inputs: (number[] | Float32Array | Float64Array)[]): Float32Array[] {
  return inputs.map(arr => toFloat32Array(arr));
}

/**
 * Convert Float32Array result back to regular array if needed
 */
function fromFloat32Array(input: Float32Array): number[] {
  return Array.from(input);
}

/**
 * Attention output interface
 */
export interface AttentionOutput {
  /** Output vector as regular array */
  values: number[];
  /** Output as Float32Array for performance-critical code */
  raw: Float32Array;
}

/**
 * Multi-head attention mechanism
 *
 * This wrapper automatically converts array inputs to Float32Array.
 */
export class MultiHeadAttention {
  private inner: any;
  public readonly dim: number;
  public readonly numHeads: number;

  /**
   * Create a new multi-head attention instance
   *
   * @param dim - Embedding dimension (must be divisible by numHeads)
   * @param numHeads - Number of attention heads
   */
  constructor(dim: number, numHeads: number) {
    const attention = getAttentionModule();
    this.inner = new attention.MultiHeadAttention(dim, numHeads);
    this.dim = dim;
    this.numHeads = numHeads;
  }

  /**
   * Compute multi-head attention
   *
   * @param query - Query vector
   * @param keys - Array of key vectors
   * @param values - Array of value vectors
   * @returns Attention output
   *
   * @example
   * ```typescript
   * const mha = new MultiHeadAttention(64, 4);
   *
   * // Works with regular arrays
   * const result1 = mha.compute([...64 values], [[...64], [...64]], [[...64], [...64]]);
   *
   * // Also works with Float32Array
   * const q = new Float32Array(64);
   * const k = [new Float32Array(64)];
   * const v = [new Float32Array(64)];
   * const result2 = mha.compute(q, k, v);
   * ```
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  /**
   * Compute and return raw Float32Array (faster, no conversion)
   */
  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }

  get headDim(): number {
    return this.dim / this.numHeads;
  }
}

/**
 * Flash attention with tiled computation
 */
export class FlashAttention {
  private inner: any;
  public readonly dim: number;
  public readonly blockSize: number;

  /**
   * Create a new flash attention instance
   *
   * @param dim - Embedding dimension
   * @param blockSize - Block size for tiled computation (default: 512)
   */
  constructor(dim: number, blockSize: number = 512) {
    const attention = getAttentionModule();
    this.inner = new attention.FlashAttention(dim, blockSize);
    this.dim = dim;
    this.blockSize = blockSize;
  }

  /**
   * Compute flash attention
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }
}

/**
 * Hyperbolic attention in Poincare ball model
 */
export class HyperbolicAttention {
  private inner: any;
  public readonly dim: number;
  public readonly curvature: number;

  /**
   * Create a new hyperbolic attention instance
   *
   * @param dim - Embedding dimension
   * @param curvature - Hyperbolic curvature (typically 1.0)
   */
  constructor(dim: number, curvature: number = 1.0) {
    const attention = getAttentionModule();
    this.inner = new attention.HyperbolicAttention(dim, curvature);
    this.dim = dim;
    this.curvature = curvature;
  }

  /**
   * Compute hyperbolic attention
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }
}

/**
 * Linear attention (Performer-style) with O(n) complexity
 */
export class LinearAttention {
  private inner: any;
  public readonly dim: number;
  public readonly numFeatures: number;

  /**
   * Create a new linear attention instance
   *
   * @param dim - Embedding dimension
   * @param numFeatures - Number of random features
   */
  constructor(dim: number, numFeatures: number) {
    const attention = getAttentionModule();
    this.inner = new attention.LinearAttention(dim, numFeatures);
    this.dim = dim;
    this.numFeatures = numFeatures;
  }

  /**
   * Compute linear attention
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }
}

/**
 * Local-global attention (Longformer-style)
 */
export class LocalGlobalAttention {
  private inner: any;
  public readonly dim: number;
  public readonly localWindow: number;
  public readonly globalTokens: number;

  /**
   * Create a new local-global attention instance
   *
   * @param dim - Embedding dimension
   * @param localWindow - Size of local attention window
   * @param globalTokens - Number of global attention tokens
   */
  constructor(dim: number, localWindow: number, globalTokens: number) {
    const attention = getAttentionModule();
    this.inner = new attention.LocalGlobalAttention(dim, localWindow, globalTokens);
    this.dim = dim;
    this.localWindow = localWindow;
    this.globalTokens = globalTokens;
  }

  /**
   * Compute local-global attention
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }
}

/**
 * MoE configuration
 */
export interface MoEConfig {
  dim: number;
  numExperts: number;
  topK: number;
  expertCapacity?: number;
}

/**
 * Mixture of Experts attention
 */
export class MoEAttention {
  private inner: any;
  public readonly config: MoEConfig;

  /**
   * Create a new MoE attention instance
   *
   * @param config - MoE configuration
   */
  constructor(config: MoEConfig) {
    const attention = getAttentionModule();
    this.inner = new attention.MoEAttention({
      dim: config.dim,
      numExperts: config.numExperts,
      topK: config.topK,
      expertCapacity: config.expertCapacity ?? 1.25,
    });
    this.config = config;
  }

  /**
   * Create with simple parameters
   */
  static simple(dim: number, numExperts: number, topK: number): MoEAttention {
    return new MoEAttention({ dim, numExperts, topK });
  }

  /**
   * Compute MoE attention
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return {
      values: fromFloat32Array(raw),
      raw
    };
  }

  computeRaw(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Float32Array {
    return this.inner.compute(query, keys, values);
  }
}

// Hyperbolic math utilities

/**
 * Project a vector into the Poincare ball
 */
export function projectToPoincareBall(
  vector: number[] | Float32Array,
  curvature: number = 1.0
): number[] {
  const attention = getAttentionModule();
  const result = attention.projectToPoincareBall(toFloat32Array(vector), curvature);
  return fromFloat32Array(result);
}

/**
 * Compute hyperbolic (Poincare) distance between two points
 */
export function poincareDistance(
  a: number[] | Float32Array,
  b: number[] | Float32Array,
  curvature: number = 1.0
): number {
  const attention = getAttentionModule();
  return attention.poincareDistance(toFloat32Array(a), toFloat32Array(b), curvature);
}

/**
 * Mobius addition in hyperbolic space
 */
export function mobiusAddition(
  a: number[] | Float32Array,
  b: number[] | Float32Array,
  curvature: number = 1.0
): number[] {
  const attention = getAttentionModule();
  const result = attention.mobiusAddition(toFloat32Array(a), toFloat32Array(b), curvature);
  return fromFloat32Array(result);
}

/**
 * Exponential map from tangent space to hyperbolic space
 */
export function expMap(
  base: number[] | Float32Array,
  tangent: number[] | Float32Array,
  curvature: number = 1.0
): number[] {
  const attention = getAttentionModule();
  const result = attention.expMap(toFloat32Array(base), toFloat32Array(tangent), curvature);
  return fromFloat32Array(result);
}

/**
 * Logarithmic map from hyperbolic space to tangent space
 */
export function logMap(
  base: number[] | Float32Array,
  point: number[] | Float32Array,
  curvature: number = 1.0
): number[] {
  const attention = getAttentionModule();
  const result = attention.logMap(toFloat32Array(base), toFloat32Array(point), curvature);
  return fromFloat32Array(result);
}

/**
 * Check if attention module is available
 */
export function isAttentionAvailable(): boolean {
  try {
    getAttentionModule();
    return true;
  } catch {
    return false;
  }
}

/**
 * Get attention module version
 */
export function getAttentionVersion(): string | null {
  try {
    const attention = getAttentionModule();
    return attention.version?.() ?? null;
  } catch {
    return null;
  }
}

// ============================================================================
// Graph-based Attention (for code structure)
// ============================================================================

/**
 * Graph attention with Rotary Position Embeddings
 * Excellent for code AST and dependency graphs
 */
export class GraphRoPeAttention {
  private inner: any;
  public readonly dim: number;
  public readonly numHeads: number;
  public readonly maxSeqLen: number;

  /**
   * @param numHeads - kept for API compatibility; the native kernel is single-head
   * @param maxSeqLen - maximum graph position (native `maxPosition`)
   */
  constructor(dim: number, numHeads: number = 4, maxSeqLen: number = 4096) {
    const attention = getAttentionModule();
    this.inner = new attention.GraphRoPeAttention({ dim, maxPosition: maxSeqLen });
    this.dim = dim;
    this.numHeads = numHeads;
    this.maxSeqLen = maxSeqLen;
  }

  /**
   * @param keyPositions - graph positions of the keys (e.g. hop distances);
   *   when given, rotary embeddings are applied relative to `queryPosition`
   */
  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[],
    keyPositions?: number[],
    queryPosition: number = 0
  ): AttentionOutput {
    const q = toFloat32Array(query);
    const k = toFloat32Arrays(keys);
    const v = toFloat32Arrays(values);
    const raw = keyPositions
      ? this.inner.computeWithPositions(q, k, v, queryPosition, keyPositions)
      : this.inner.compute(q, k, v);
    return { values: fromFloat32Array(raw), raw };
  }
}

/**
 * Edge-featured attention for graphs with edge attributes
 * Useful for weighted dependency graphs
 */
export class EdgeFeaturedAttention {
  private inner: any;
  public readonly dim: number;
  public readonly edgeDim: number;

  constructor(dim: number, edgeDim: number = 16) {
    const attention = getAttentionModule();
    this.inner = new attention.EdgeFeaturedAttention({ nodeDim: dim, edgeDim, numHeads: 1 });
    this.dim = dim;
    this.edgeDim = edgeDim;
  }

  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[],
    edgeFeatures?: (number[] | Float32Array)[]
  ): AttentionOutput {
    const q = toFloat32Array(query);
    const k = toFloat32Arrays(keys);
    const v = toFloat32Arrays(values);
    const raw = edgeFeatures
      ? this.inner.computeWithEdges(q, k, v, toFloat32Arrays(edgeFeatures))
      : this.inner.compute(q, k, v);
    return { values: fromFloat32Array(raw), raw };
  }
}

/**
 * Dual-space attention (Euclidean + Hyperbolic)
 * Best of both worlds for hierarchical + semantic similarity
 */
export class DualSpaceAttention {
  private inner: any;
  public readonly dim: number;
  public readonly curvature: number;
  public readonly alpha: number;

  /**
   * @param alpha - weight of the Euclidean space in [0, 1]; the hyperbolic
   *   space gets `1 - alpha`
   */
  constructor(dim: number, curvature: number = 1.0, alpha: number = 0.5) {
    const attention = getAttentionModule();
    this.inner = attention.DualSpaceAttention.withWeights(dim, curvature, alpha, 1 - alpha);
    this.dim = dim;
    this.curvature = curvature;
    this.alpha = alpha;
  }

  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return { values: fromFloat32Array(raw), raw };
  }
}

/**
 * Basic dot-product attention
 */
export class DotProductAttention {
  private inner: any;
  public readonly dim: number;

  constructor(dim: number) {
    const attention = getAttentionModule();
    this.inner = new attention.DotProductAttention(dim);
    this.dim = dim;
  }

  compute(
    query: number[] | Float32Array,
    keys: (number[] | Float32Array)[],
    values: (number[] | Float32Array)[]
  ): AttentionOutput {
    const raw = this.inner.compute(
      toFloat32Array(query),
      toFloat32Arrays(keys),
      toFloat32Arrays(values)
    );
    return { values: fromFloat32Array(raw), raw };
  }
}

// ============================================================================
// Parallel/Batch Attention Compute
// ============================================================================

/** Attention kernels the native batch/parallel API can run. */
export type BatchAttentionType =
  | 'dot' | 'dot_product' | 'scaled-dot-product'
  | 'flash' | 'linear' | 'hyperbolic' | 'local-global';

/** Optional kernel parameters for parallel/batch compute. */
export interface BatchAttentionOptions {
  blockSize?: number;
  numFeatures?: number;
  localWindow?: number;
  globalTokens?: number;
  curvature?: number;
}

const NATIVE_ATTENTION_TYPES: Record<string, string> = {
  'dot': 'ScaledDotProduct',
  'dot_product': 'ScaledDotProduct',
  'dot-product': 'ScaledDotProduct',
  'scaled-dot-product': 'ScaledDotProduct',
  'flash': 'Flash',
  'linear': 'Linear',
  'hyperbolic': 'Hyperbolic',
  'local-global': 'LocalGlobal',
  'local_global': 'LocalGlobal',
};

function toNativeAttentionType(type: string): string {
  const mapped = NATIVE_ATTENTION_TYPES[String(type).toLowerCase()];
  if (!mapped) {
    throw new Error(
      `Attention type '${type}' has no native batch kernel; ` +
      `use one of: dot, flash, linear, hyperbolic, local-global`
    );
  }
  return mapped;
}

function isNestedVectors(x: any): boolean {
  return Array.isArray(x) && x.length > 0 && (Array.isArray(x[0]) || x[0] instanceof Float32Array) &&
    x[0].length > 0 && (Array.isArray(x[0][0]) || x[0][0] instanceof Float32Array);
}

/**
 * Compute attention in parallel across multiple queries.
 *
 * `keys`/`values` are either one shared set used for every query, or one set
 * per query (`keys[i]` belongs to `queries[i]`).
 */
export async function parallelAttentionCompute(
  queries: (number[] | Float32Array)[],
  keys: (number[] | Float32Array)[] | (number[] | Float32Array)[][],
  values: (number[] | Float32Array)[] | (number[] | Float32Array)[][],
  attentionType: BatchAttentionType = 'dot',
  options: BatchAttentionOptions = {}
): Promise<number[][]> {
  const attention = getAttentionModule();
  const nativeType = toNativeAttentionType(attentionType);
  const q = toFloat32Arrays(queries);
  if (q.length === 0) return [];
  const perQuery = (set: any): Float32Array[][] => isNestedVectors(set)
    ? (set as (number[] | Float32Array)[][]).map(toFloat32Arrays)
    : q.map(() => toFloat32Arrays(set as (number[] | Float32Array)[]));
  const k = perQuery(keys);
  const v = perQuery(values);
  if (k.length !== q.length || v.length !== q.length) {
    throw new Error(`parallelAttentionCompute: got ${q.length} queries but ${k.length} key sets and ${v.length} value sets`);
  }
  const config = { attentionType: nativeType, dim: q[0].length, ...options };
  const result = await attention.parallelAttentionCompute(config, q, k, v);
  return result.outputs.map((r: Float32Array) => fromFloat32Array(r));
}

/**
 * Batch attention compute for multiple query-key-value sets
 */
export async function batchAttentionCompute(
  batches: Array<{
    query: number[] | Float32Array;
    keys: (number[] | Float32Array)[];
    values: (number[] | Float32Array)[];
  }>,
  attentionType: BatchAttentionType = 'dot',
  options: BatchAttentionOptions = {}
): Promise<number[][]> {
  if (batches.length === 0) return [];
  return parallelAttentionCompute(
    batches.map(b => b.query),
    batches.map(b => b.keys),
    batches.map(b => b.values),
    attentionType,
    options
  );
}

/**
 * Async flash attention
 */
export async function computeFlashAttentionAsync(
  query: number[] | Float32Array,
  keys: (number[] | Float32Array)[],
  values: (number[] | Float32Array)[],
  blockSize: number = 64
): Promise<number[]> {
  const attention = getAttentionModule();
  const q = toFloat32Array(query);
  const result = await attention.computeFlashAttentionAsync(
    q, toFloat32Arrays(keys), toFloat32Arrays(values), q.length, blockSize
  );
  return fromFloat32Array(result);
}

/**
 * Async hyperbolic attention
 */
export async function computeHyperbolicAttentionAsync(
  query: number[] | Float32Array,
  keys: (number[] | Float32Array)[],
  values: (number[] | Float32Array)[],
  curvature: number = 1.0
): Promise<number[]> {
  const attention = getAttentionModule();
  const q = toFloat32Array(query);
  const result = await attention.computeHyperbolicAttentionAsync(
    q, toFloat32Arrays(keys), toFloat32Arrays(values), q.length, curvature
  );
  return fromFloat32Array(result);
}

// ============================================================================
// Training Utilities (for SONA integration)
// ============================================================================

/**
 * Adam optimizer for attention training.
 *
 * The native optimizer is sized to the parameter vector, so it is created on
 * the first `step()` and every later step must pass the same length.
 */
export class AdamOptimizer {
  private inner: any = null;
  private paramCount = 0;

  constructor(
    public readonly learningRate: number = 0.001,
    public readonly beta1: number = 0.9,
    public readonly beta2: number = 0.999
  ) {
    getAttentionModule();
  }

  /** Returns the updated parameters; `params` itself is not modified. */
  step(gradients: number[] | Float32Array, params: number[] | Float32Array): number[] {
    if (gradients.length !== params.length) {
      throw new Error(`AdamOptimizer.step: ${gradients.length} gradients for ${params.length} params`);
    }
    if (!this.inner) {
      this.inner = getAttentionModule().AdamOptimizer.withBetas(params.length, this.learningRate, this.beta1, this.beta2);
      this.paramCount = params.length;
    } else if (params.length !== this.paramCount) {
      throw new Error(`AdamOptimizer.step: expected ${this.paramCount} params, got ${params.length}`);
    }
    // The native step consumes its params buffer, so always hand it a copy.
    const result = this.inner.step(new Float32Array(params), toFloat32Array(gradients));
    return fromFloat32Array(result);
  }
}

/**
 * InfoNCE contrastive loss
 */
export function infoNceLoss(
  anchor: number[] | Float32Array,
  positive: number[] | Float32Array,
  negatives: (number[] | Float32Array)[],
  temperature: number = 0.07
): number {
  const attention = getAttentionModule();
  return new attention.InfoNceLoss(temperature).compute(
    toFloat32Array(anchor),
    toFloat32Array(positive),
    toFloat32Arrays(negatives)
  );
}

/** Options for {@link mineHardNegatives}. */
export interface MineHardNegativesOptions {
  /** Positive example; defaults to the anchor. */
  positive?: number[] | Float32Array;
  strategy?: 'Random' | 'HardNegative' | 'SemiHard' | 'DistanceWeighted';
}

/**
 * Hard negative mining for contrastive learning. Returns the selected
 * candidate vectors, hardest first.
 */
export function mineHardNegatives(
  anchor: number[] | Float32Array,
  candidates: (number[] | Float32Array)[],
  topK: number = 5,
  options: MineHardNegativesOptions = {}
): number[][] {
  const attention = getAttentionModule();
  const miner = new attention.HardNegativeMiner(options.strategy ?? 'HardNegative');
  const a = toFloat32Array(anchor);
  const pool = toFloat32Arrays(candidates);
  const indices: number[] = miner.mine(
    a,
    options.positive ? toFloat32Array(options.positive) : a,
    pool,
    Math.min(topK, pool.length)
  );
  return indices.map(i => fromFloat32Array(pool[i]));
}

// ============================================================================
// Benchmarking
// ============================================================================

/**
 * Benchmark the native attention kernels, keyed by kernel name.
 */
export async function benchmarkAttention(
  dim: number,
  seqLen: number,
  iterations: number = 100
): Promise<Record<string, { avgMs: number; minMs: number; maxMs: number }>> {
  const attention = getAttentionModule();
  const out: Record<string, { avgMs: number; minMs: number; maxMs: number }> = {};
  for (const type of ['ScaledDotProduct', 'Flash', 'Linear', 'Hyperbolic']) {
    const r = await attention.benchmarkAttention(type, dim, seqLen, iterations);
    out[r.name || type] = { avgMs: r.avgMs, minMs: r.minMs, maxMs: r.maxMs };
  }
  return out;
}

export default {
  // Core attention types
  DotProductAttention,
  MultiHeadAttention,
  FlashAttention,
  HyperbolicAttention,
  LinearAttention,
  LocalGlobalAttention,
  MoEAttention,

  // Graph attention types
  GraphRoPeAttention,
  EdgeFeaturedAttention,
  DualSpaceAttention,

  // Parallel/batch compute
  parallelAttentionCompute,
  batchAttentionCompute,
  computeFlashAttentionAsync,
  computeHyperbolicAttentionAsync,

  // Training utilities
  AdamOptimizer,
  infoNceLoss,
  mineHardNegatives,

  // Hyperbolic math
  projectToPoincareBall,
  poincareDistance,
  mobiusAddition,
  expMap,
  logMap,

  // Utilities
  isAttentionAvailable,
  getAttentionVersion,
  benchmarkAttention,
};
