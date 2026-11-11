import type { EMeansOptions } from '../e-means.js';

/** Caller-owned mutable records. Every codec operation validates their shape and finite coordinates. */
export interface ProductQuantizer {
  schema: 'ruvector-e-means-pq-v1';
  dimensions: number;
  subspaces: number;
  /** 1..256, shared by every subspace. */
  clusters: number;
  /** Flat subspace-major codebooks: subspaces × clusters × (dimensions/subspaces). */
  centroids: Float32Array;
}

export interface ProductQuantizerJSON {
  schema: 'ruvector-e-means-pq-v1';
  dimensions: number;
  subspaces: number;
  clusters: number;
  centroids: number[];
}

/** Requires init() from the root package. Input data is borrowed; returned centroids are owned. */
export function trainProductQuantizer(data: Float32Array, dimensions: number, subspaces: number, options: EMeansOptions): ProductQuantizer;
/** Requires init(). Codes are row-major, rows × subspaces bytes. Empty input is supported. */
export function encodeProductQuantizer(model: ProductQuantizer, data: Float32Array): Uint8Array;
/** No runtime initialization required. Returns an owned reconstruction; empty codes are supported. */
export function decodeProductQuantizer(model: ProductQuantizer, codes: Uint8Array): Float32Array;
export function productQuantizerToJSON(model: ProductQuantizer): ProductQuantizerJSON;
export function productQuantizerFromJSON(value: unknown): ProductQuantizer;
