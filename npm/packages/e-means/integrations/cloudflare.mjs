import { init, fit, predict } from '../runtime.mjs';
import { workEstimate } from './clustering-validation.mjs';
import { HTTPError, integer, LIMITS, plain, floats, fitOptions } from './cloudflare/validation.mjs';
import { readJSON, response, tokenMatches } from './cloudflare/http.mjs';

export { persistCodebook, loadCodebook, exportCentroids, CentroidExportError } from './cloudflare/storage.mjs';
export { LIMITS as WORKER_LIMITS } from './cloudflare/validation.mjs';

/** Private, stateless Worker handler. Pass a statically imported WASM module. */
export function createClusteringWorker(options) {
  const config = plain(options, ['wasmModule', 'bearerToken', 'allowUnauthenticated',
    'maxRequestBytes', 'maxWork', 'bodyTimeoutMs'], ['wasmModule'], 'Worker options');
  if (!(config.wasmModule instanceof WebAssembly.Module)) throw new TypeError('wasmModule must be a precompiled WebAssembly.Module');
  if (config.bearerToken !== undefined && (typeof config.bearerToken !== 'string' ||
      config.bearerToken.length < 24 || config.bearerToken.length > 1024 || !/^[\x21-\x7e]+$/.test(config.bearerToken))) {
    throw new TypeError('bearerToken must be 24..1024 printable non-space ASCII characters');
  }
  if (config.allowUnauthenticated !== undefined && typeof config.allowUnauthenticated !== 'boolean') {
    throw new TypeError('allowUnauthenticated must be boolean');
  }
  const maxBytes = integer(config.maxRequestBytes ?? 512 * 1024, 1, LIMITS.maxRequestBytes, 'maxRequestBytes');
  const maxWork = integer(config.maxWork ?? 10_000_000, 1, LIMITS.maxWork, 'maxWork');
  const timeoutMs = integer(config.bodyTimeoutMs ?? 5000, 1, 30_000, 'bodyTimeoutMs');
  const token = config.bearerToken;
  return Object.freeze({ async fetch(request) {
    try {
      if (!(request instanceof Request)) throw new TypeError('request must be a Request');
      const url = new URL(request.url);
      if (url.search) throw new HTTPError(400, 'invalid_query', 'Query parameters are unsupported');
      if (url.pathname === '/health') {
        if (request.method !== 'GET') return response({ error: 'method_not_allowed' }, 405, { Allow: 'GET' });
        return response({ schema: 'ruvector.e-means-worker/1', stateless: true,
          authenticatedOperations: Boolean(token), maxRequestBytes: maxBytes, maxWork });
      }
      if (!['/fit', '/predict'].includes(url.pathname)) return response({ error: 'not_found' }, 404);
      if (request.method !== 'POST') return response({ error: 'method_not_allowed' }, 405, { Allow: 'POST' });
      if (!token && config.allowUnauthenticated !== true) {
        throw new HTTPError(503, 'authentication_unconfigured', 'Configure E_MEANS_BEARER_TOKEN before using operations');
      }
      if (token && !tokenMatches(request.headers.get('authorization'), `Bearer ${token}`)) {
        throw new HTTPError(401, 'unauthorized', 'A valid bearer token is required');
      }
      const isFit = url.pathname === '/fit';
      const fields = isFit ? ['data', 'dimensions', 'options'] : ['data', 'dimensions', 'centroids'];
      const body = plain(await readJSON(request, maxBytes, timeoutMs), fields, fields, 'request');
      const dimensions = integer(body.dimensions, 1, LIMITS.maxDimensions, 'dimensions');
      const data = floats(body.data, dimensions, 'data', !isFit);
      if (isFit) {
        const options = fitOptions(body.options, data.length / dimensions);
        // Shared coordinate-scale admission estimate, including centroid
        // statistics and O(k²) donor revival; not a hard CPU-clock limit.
        const epochs = options.epochs ?? 3;
        const work = workEstimate({ ...options, epochs }, data.length / dimensions, 0, dimensions);
        if (work > maxWork) throw new HTTPError(413, 'work_budget_exceeded', 'Fit exceeds configured work budget');
        await init(config.wasmModule);
        const model = fit(data, dimensions, options);
        return response({ ...model, centroids: Array.from(model.centroids), assignments: Array.from(model.assignments) });
      }
      const centroids = floats(body.centroids, dimensions, 'centroids');
      const clusters = integer(centroids.length / dimensions, 1, LIMITS.maxClusters, 'clusters');
      if (data.length + centroids.length > LIMITS.maxElements || data.length * clusters > maxWork) {
        throw new HTTPError(413, 'work_budget_exceeded', 'Prediction exceeds configured work or element budget');
      }
      await init(config.wasmModule);
      return response({ assignments: Array.from(predict(data, dimensions, centroids)) });
    } catch (error) {
      if (error instanceof HTTPError) return response({ error: error.code, message: error.message }, error.status);
      if (error instanceof TypeError || error instanceof RangeError) return response({ error: 'invalid_input', message: error.message }, 422);
      return response({ error: 'execution_failed', message: 'Clustering execution failed' }, 500);
    }
  } });
}
