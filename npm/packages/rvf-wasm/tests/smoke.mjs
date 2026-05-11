// Smoke test: regression guard for issue #415.
// Verifies that @ruvector/rvf-wasm can be loaded as ESM without
// `SyntaxError: Cannot use 'import.meta' outside a module` and that
// init() returns a usable WASM exports object.

import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(__dirname, '..');

let failures = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg);
    failures++;
  } else {
    console.log('  ok:', msg);
  }
}

// 1. ESM import via package.json `default` (the published surface).
const mod = await import(join(pkgRoot, 'pkg/rvf_wasm.mjs'));
assert(typeof mod.default === 'function', 'mjs default export is a function');

// 2. Initialize and confirm we got real WASM exports back.
const exports_ = await mod.default();
assert(exports_ != null && typeof exports_ === 'object', 'init() returned exports object');
assert(exports_.memory instanceof WebAssembly.Memory, 'exports.memory is WebAssembly.Memory');

// 3. Idempotent re-init returns the cached instance.
const exports2 = await mod.default();
assert(exports2 === exports_, 'init() is idempotent');

if (failures > 0) {
  console.error(`\n${failures} smoke check(s) failed`);
  process.exit(1);
}
console.log('\nrvf-wasm smoke OK');
