// Regression test for issue #323 — ONNX embedder failed on Node 22 LTS
// because `pkg/ruvector_onnx_embeddings_wasm.js` (the wasm-bindgen
// `--target bundler` output) does `import * as wasm from "./*.wasm"`,
// which Node 22 rejects without `--experimental-wasm-modules`.
//
// This test loads the new Node-friendly entry that uses
// `fs.readFileSync` + `WebAssembly.instantiate` and asserts the bindings
// surface looks healthy. Run with:
//
//   node test/onnx-node22-loader.test.mjs

import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, '..');
const nodeLoader = join(repoRoot, 'src/core/onnx/pkg/ruvector_onnx_embeddings_wasm_node.mjs');

let failures = 0;
function check(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg);
    failures++;
  } else {
    console.log('  ok:', msg);
  }
}

console.log('Node version:', process.version);
console.log('Loader path:', nodeLoader);

check(existsSync(nodeLoader), 'Node-friendly loader exists in src/core/onnx/pkg/');

// The actual smoke: dynamically import the loader. If Node still rejected
// the .wasm static import this would throw `Unknown file extension ".wasm"`
// and the test would fail.
const mod = await import(nodeLoader);

check(typeof mod === 'object', 'loader produced a module namespace');

// wasm-bindgen `--target bundler` re-exports everything via _bg.js, so the
// loader should re-export the same surface. Pick a couple of well-known
// wasm-bindgen runtime exports as a sanity check.
const expectedSymbols = ['__wbg_set_wasm', 'WasmEmbedder', 'WasmEmbedderConfig'];
for (const sym of expectedSymbols) {
  check(typeof mod[sym] !== 'undefined', `re-exports ${sym}`);
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log(`\nNode ${process.version} ONNX loader smoke OK`);
