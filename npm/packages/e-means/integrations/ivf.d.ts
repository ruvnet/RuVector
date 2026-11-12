import type { EMeansModel, EMeansOptions } from '../e-means.js';

/** Caller-owned structured-clone-compatible records; postingList validates on every operation. */
export interface IVFPartition {
  schema: 'ruvector-e-means-ivf-v1';
  model: EMeansModel;
  /** Length clusters+1. Cluster j occupies indices[offsets[j]..offsets[j+1]). */
  offsets: Uint32Array;
  /** Original input indices, in ascending order within every cluster. */
  indices: Uint32Array;
}

/** Requires init() from the root package. Trains centroids and returns owned posting arrays. */
export function partition(data: Float32Array, dimensions: number, options: EMeansOptions): IVFPartition;
/** Validates shape, assignment consistency and stable unique indices; returns an owned list. */
export function postingList(value: IVFPartition, cluster: number): Uint32Array;
