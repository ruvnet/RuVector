#!/usr/bin/env node
/**
 * copy-onnx-assets.js — copy non-TypeScript ONNX runtime files into `dist/`.
 *
 * Why: `tsconfig.json` does not set `allowJs`, so `tsc` only emits compiled
 * `.ts` output. The wasm-bindgen artifacts under `src/core/onnx/pkg/`
 * (`*.wasm`, `*_bg.js`, `*.d.ts`, `package.json`, `LICENSE`) and the
 * sibling `src/core/onnx/loader.js` are required at runtime by
 * `dist/core/onnx-embedder.js` but were not being copied — published
 * tarballs were missing the WASM payload entirely (#354), making
 * `OptimizedOnnxEmbedder` unloadable on every clean install.
 *
 * Implemented as a Node script (no `cp -r`) so the build runs unchanged
 * on Windows.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');

function copyRecursive(src, dst) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dst, { recursive: true });
    for (const entry of fs.readdirSync(src)) {
      // Skip dotfiles (e.g. transient `.claude-flow/` agent metadata)
      // and node_modules — neither belongs in the published artifact.
      if (entry.startsWith('.') || entry === 'node_modules') continue;
      copyRecursive(path.join(src, entry), path.join(dst, entry));
    }
  } else {
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.copyFileSync(src, dst);
  }
}

// (sourceRel, destinationRel) — both relative to the package root.
const assets = [
  ['src/core/onnx/loader.js', 'dist/core/onnx/loader.js'],
  ['src/core/onnx/pkg', 'dist/core/onnx/pkg'],
];

let fileCount = 0;
function countFiles(p) {
  const stat = fs.statSync(p);
  if (stat.isDirectory()) {
    for (const entry of fs.readdirSync(p)) countFiles(path.join(p, entry));
  } else {
    fileCount++;
  }
}

for (const [srcRel, dstRel] of assets) {
  const src = path.join(pkgRoot, srcRel);
  const dst = path.join(pkgRoot, dstRel);
  if (!fs.existsSync(src)) {
    console.error(`copy-onnx-assets: missing source ${srcRel}`);
    process.exit(1);
  }
  copyRecursive(src, dst);
  countFiles(src);
}

// Sanity check the runtime payload landed where the embedder expects it.
const required = [
  'dist/core/onnx/loader.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.wasm',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm.js',
];
const missing = required.filter((rel) => !fs.existsSync(path.join(pkgRoot, rel)));
if (missing.length > 0) {
  console.error('copy-onnx-assets: required files missing after copy:');
  for (const m of missing) console.error(`  - ${m}`);
  process.exit(1);
}

console.log(`copy-onnx-assets: ${fileCount} ONNX runtime file(s) staged under dist/.`);
