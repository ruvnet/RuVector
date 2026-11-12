import type { EMeansModel } from '@ruvector/e-means';
import {
  createClusteringWorker, persistCodebook, loadCodebook, exportCentroids,
  type CodebookInput, type R2Binding, type VectorizeBinding,
} from '@ruvector/e-means/cloudflare';

declare const wasm: WebAssembly.Module;
declare const model: EMeansModel;
declare const request: Request;
const worker = createClusteringWorker({ wasmModule: wasm, maxWork: 1_000_000, bodyTimeoutMs: 1000 });
const response: Promise<Response> = worker.fetch(request);
const bucket: R2Binding = { put: async () => ({}), get: async () => null };
const index: VectorizeBinding = { upsert: async () => ({ mutationId: 'acknowledgement' }) };
const input: CodebookInput = model;
void response;
void persistCodebook(bucket, 'model', input);
void loadCodebook(bucket, 'model');
void exportCentroids(index, model, { prefix: 'model', namespace: 'owned' });
// @ts-expect-error Worker configuration numbers cannot be strings.
createClusteringWorker({ wasmModule: wasm, maxWork: 'unbounded' });
// @ts-expect-error Codebook storage requires owned float32 centroid arrays.
persistCodebook(bucket, 'model', { dimensions: 1, centroids: [0, 1] });
// @ts-expect-error Vectorize returns an acknowledgement identity.
const invalidIndex: VectorizeBinding = { upsert: async () => ({}) };
void invalidIndex;
