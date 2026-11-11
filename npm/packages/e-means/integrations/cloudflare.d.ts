export interface WorkerOptions {
  /** Statically import @ruvector/e-means/wasm in a Worker. */
  wasmModule: WebAssembly.Module;
  bearerToken?: string;
  /** Explicit local/trusted opt-in; false by default. */
  allowUnauthenticated?: boolean;
  maxRequestBytes?: number;
  maxWork?: number;
  bodyTimeoutMs?: number;
}
/** One row-major set of full-dimensional centroids, not subspace-major PQ books. */
export interface CodebookInput {
  dimensions: number;
  centroids: Float32Array;
  schema?: 'ruvector.e-means-codebook/1';
}
export interface StoredCodebook extends CodebookInput {
  schema: 'ruvector.e-means-codebook/1';
  clusters: number;
}
export interface R2Binding {
  put(key: string, body: string, options?: { httpMetadata?: { contentType?: string } }): Promise<unknown>;
  get(key: string): Promise<{ size: number; body: ReadableStream<Uint8Array> } | null>;
}
export interface VectorizeVector {
  id: string;
  values: number[];
  metadata: { cluster: number };
  namespace?: string;
}
export interface VectorizeBinding {
  upsert(vectors: VectorizeVector[]): Promise<{ mutationId: string }>;
}
export interface ExportOptions {
  /** Use a versioned prefix for a new codebook; old centroid IDs are not deleted. */
  prefix: string;
  namespace?: string;
  batchSize?: number;
}
export interface ExportReceipt { submitted: number; mutationIds: string[] }
/** Does not persist request data. Operations fail closed unless authentication is configured. */
export function createClusteringWorker(options: WorkerOptions): { fetch(request: Request): Promise<Response> };
export function persistCodebook(bucket: Pick<R2Binding, 'put'>, key: string, model: CodebookInput): Promise<{
  key: string; dimensions: number; clusters: number; bytes: number;
}>;
export function loadCodebook(bucket: Pick<R2Binding, 'get'>, key: string): Promise<StoredCodebook | null>;
/** Batches upserts; acknowledgement does not establish query visibility or atomicity. */
export function exportCentroids(index: VectorizeBinding, model: CodebookInput, options: ExportOptions): Promise<ExportReceipt>;
export class CentroidExportError extends Error {
  constructor(submitted: number, mutationIds: readonly string[], cause: unknown);
  /** Rows in previously acknowledged batches; the failing batch may also have applied. */
  submitted: number;
  mutationIds: string[];
}
export const WORKER_LIMITS: Readonly<{
  maxRequestBytes: number; maxWork: number; maxRows: number; maxDimensions: number;
  maxClusters: number; maxEpochs: number; maxElements: number; maxCodebookBytes: number;
}>;
