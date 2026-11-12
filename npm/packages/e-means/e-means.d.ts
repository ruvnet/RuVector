export interface EMeansOptions {
  clusters: number;
  /** Positive integer, default 3. */
  epochs?: number;
  /** Positive integer; defaults to the paper's batch scaling law, capped at the row count. */
  batchSize?: number;
  /** Peak EMA step in (0, 1]; defaults to the paper's step scaling law. */
  peakStep?: number;
  /** Non-negative safe integer; default 42. */
  seed?: number;
  finalization?: 'lastEpochMean' | 'ema';
}

export interface EMeansModel {
  /** Flat row-major clusters × dimensions codebook. */
  centroids: Float32Array;
  /** Nearest final centroid for every input row. */
  assignments: Uint32Array;
  dimensions: number;
  /** Mean squared Euclidean distance per vector, not per coordinate. */
  mse: number;
  batchSize: number;
  peakStep: number;
  updates: number;
  revivedClusters: number;
}

/** Load the bundled WebAssembly runtime. In browsers the default .wasm URL must be served. */
export function init(input?: BufferSource | WebAssembly.Module): Promise<void>;
/** Requires init(). Inputs are not modified. */
export function fit(data: Float32Array, dimensions: number, options: EMeansOptions): EMeansModel;
/** Requires init(). An empty input returns an empty assignment array. */
export function predict(data: Float32Array, dimensions: number, centroids: Float32Array): Uint32Array;
