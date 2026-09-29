// prepublishOnly guard: refuse to publish this TypeScript wrapper while model
// loading and generation are unimplemented. The registry package is the
// wasm-pack build of crates/ruvllm-wasm; publishing this folder would replace
// it with a facade whose loadModel/generate cannot run a model.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const src = readFileSync(fileURLToPath(new URL('../src/index.ts', import.meta.url)), 'utf8');
if (/export const INFERENCE_AVAILABLE = false;/.test(src) && process.env.RUVLLM_WASM_ALLOW_FACADE_PUBLISH !== '1') {
  console.error(
    'refusing to publish @ruvector/ruvllm-wasm from npm/packages/ruvllm-wasm: ' +
      'INFERENCE_AVAILABLE is false, so loadModel/generate/chat only throw. ' +
      'Publish the wasm-pack build of crates/ruvllm-wasm instead, or implement inference first.'
  );
  process.exit(1);
}
