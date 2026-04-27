#!/usr/bin/env node
/**
 * copy-onnx-assets.js — copy the bundled ONNX runtime files from src/core/onnx
 * into dist/core/onnx after `tsc`.
 *
 * Why: tsc only emits .js for .ts inputs. The ONNX subsystem ships a JS
 * loader, a wasm-bindgen JS bridge, and the .wasm binary itself — none of
 * which are TypeScript and so all of which are skipped by tsc. The previous
 * build script copied only `pkg/package.json`, which left embed text and
 * the embed-stream pipeline broken with:
 *   "Failed to initialize ONNX embedder: ONNX WASM files not bundled.
 *    The onnx/ directory is missing."
 *
 * This script does an explicit, cross-platform copy of every runtime asset
 * the embedder reads via `path.join(__dirname, 'onnx', ...)` in
 * src/core/onnx-embedder.ts. It is intentionally explicit (not a recursive
 * dir copy) so a new file gets noticed and its inclusion is a deliberate
 * decision, not an accident.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');
const srcDir = path.join(pkgRoot, 'src', 'core', 'onnx');
const dstDir = path.join(pkgRoot, 'dist', 'core', 'onnx');

const ASSETS = [
  // Outer loader (model fetch / cache layer)
  'loader.js',
  // wasm-bindgen generated bridge
  'pkg/ruvector_onnx_embeddings_wasm.js',
  'pkg/ruvector_onnx_embeddings_wasm.d.ts',
  'pkg/ruvector_onnx_embeddings_wasm_bg.js',
  'pkg/ruvector_onnx_embeddings_wasm_bg.wasm',
  'pkg/ruvector_onnx_embeddings_wasm_bg.wasm.d.ts',
  // Already-shipped metadata + license (kept for completeness)
  'pkg/package.json',
  'pkg/LICENSE',
];

let copied = 0;
let missing = 0;
for (const rel of ASSETS) {
  const src = path.join(srcDir, rel);
  const dst = path.join(dstDir, rel);
  if (!fs.existsSync(src)) {
    console.warn(`copy-onnx-assets: WARN source missing: ${rel}`);
    missing++;
    continue;
  }
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  copied++;
}

if (missing > 0) {
  console.error(
    `copy-onnx-assets: ${missing} expected source asset(s) missing — ONNX subsystem may not work at runtime.`,
  );
  process.exit(1);
}

console.log(`copy-onnx-assets: ${copied} asset(s) copied to dist/core/onnx/.`);
