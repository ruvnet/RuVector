import wasmModule from '@ruvector/e-means/wasm';
import { createClusteringWorker } from '@ruvector/e-means/cloudflare';

export default {
  fetch(request, env) {
    // The secret is supplied by the deployment's binding configuration.
    return createClusteringWorker({ wasmModule, bearerToken: env.E_MEANS_BEARER_TOKEN }).fetch(request);
  },
};
