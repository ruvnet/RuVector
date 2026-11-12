import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { init, fit } from '../e-means.mjs';

const bytes = await readFile(new URL('../wasm/ruvector_e_means_bg.wasm', import.meta.url));
assert.deepEqual([...bytes.subarray(0, 4)], [0, 97, 115, 109], 'missing or invalid WebAssembly artifact');
await init(bytes);
const model = fit(new Float32Array([0, 2]), 1, { clusters: 1, epochs: 1 });
assert.equal(model.centroids[0], 1, 'bundled WASM must run the Rust mean finalizer');
assert.equal(model.mse, 1);
console.error('Bundled e-means WASM verified');
