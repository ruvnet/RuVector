#!/usr/bin/env node
/**
 * copy-onnx.js — copy the bundled ONNX WASM runtime assets into `dist/`.
 *
 * `tsc` only emits `.js`/`.d.ts` from TypeScript sources, so the prebuilt
 * wasm-bindgen artifacts under `src/core/onnx/` (`loader.js`, the `pkg/*.js`
 * glue, and `pkg/*.wasm`) never reach `dist/core/onnx/`. Without them the
 * ONNX embedder throws "ONNX WASM files not bundled" on a clean install
 * (issue #354). This recursively mirrors `src/core/onnx/` → `dist/core/onnx/`,
 * skipping editor/tooling junk (`.claude-flow`, etc.).
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');
const srcDir = path.join(pkgRoot, 'src', 'core', 'onnx');
const destDir = path.join(pkgRoot, 'dist', 'core', 'onnx');

const SKIP = new Set(['.claude-flow', '.DS_Store', 'node_modules']);

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src)) {
      if (SKIP.has(entry)) continue;
      copyRecursive(path.join(src, entry), path.join(dest, entry));
    }
  } else {
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
  }
}

if (!fs.existsSync(srcDir)) {
  console.error(`copy-onnx: source directory not found: ${srcDir}`);
  process.exit(1);
}

// Replace any stale copy so removed files don't linger.
fs.rmSync(destDir, { recursive: true, force: true });
copyRecursive(srcDir, destDir);

const required = [
  'loader.js',
  path.join('pkg', 'loader.js'),
  path.join('pkg', 'ruvector_onnx_embeddings_wasm.js'),
  path.join('pkg', 'ruvector_onnx_embeddings_wasm_bg.js'),
  path.join('pkg', 'ruvector_onnx_embeddings_wasm_bg.wasm'),
  path.join('pkg', 'package.json'),
];
const missing = required.filter((rel) => !fs.existsSync(path.join(destDir, rel)));
if (missing.length > 0) {
  console.error(`copy-onnx: expected ONNX assets missing after copy: ${missing.join(', ')}`);
  process.exit(1);
}

console.log(`copy-onnx: mirrored ${required.length}+ ONNX assets to dist/core/onnx/`);
