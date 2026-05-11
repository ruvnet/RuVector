#!/usr/bin/env node
/**
 * verify-dist.js — pre-publish gate that fails the build if any file
 * `bin/cli.js` requires from `../dist/...` is missing, OR if any of the
 * runtime asset paths read at startup by dist/core/onnx-embedder.js are
 * missing.
 *
 * Why: 0.2.23 was published without a `dist/` directory at all (issue #399),
 * and 0.2.24/0.2.25 still shipped without the ONNX runtime assets (the
 * embedder reads them via `path.join(__dirname, 'onnx', ...)` rather than
 * `require()`, so the original CLI-only scan didn't notice them).
 *
 * This script makes the publish itself fail loudly when either category of
 * artifact is incomplete.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');

function fail(msg) {
  console.error(msg);
  process.exit(1);
}

// ────────────────────────────────────────────────────────────────────────
// 1. CLI `require('../dist/...js')` scan (original behavior)
// ────────────────────────────────────────────────────────────────────────
const cliPath = path.join(pkgRoot, 'bin', 'cli.js');
if (!fs.existsSync(cliPath)) {
  fail('verify-dist: bin/cli.js not found — package layout is broken.');
}

const cliSource = fs.readFileSync(cliPath, 'utf8');
const distRequires = Array.from(
  cliSource.matchAll(/require\(['"]\.\.\/(dist\/[^'"]+\.js)['"]\)/g),
  (m) => m[1],
);
const cliUnique = Array.from(new Set(distRequires)).sort();
const cliMissing = cliUnique.filter(
  (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
);

if (cliMissing.length > 0) {
  console.error(
    `verify-dist: ${cliMissing.length} dist file(s) referenced by bin/cli.js are missing:`,
  );
  for (const rel of cliMissing) console.error(`  - ${rel}`);
  console.error(
    "\nRun `npm run build` and confirm tsc emitted under dist/. If a path was renamed,",
  );
  fail('update bin/cli.js to match.');
}

// ────────────────────────────────────────────────────────────────────────
// 2. Runtime asset gate — ONNX subsystem reads non-JS files via
//    path.join(__dirname, 'onnx', ...). tsc never touches them, so a
//    missing build-script copy is invisible to the CLI-only scan above.
//    Each path here corresponds to a `path.join(__dirname, 'onnx', ...)`
//    site in dist/core/onnx-embedder.js.
// ────────────────────────────────────────────────────────────────────────
const RUNTIME_ASSETS = [
  'dist/core/onnx/loader.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.wasm',
];

const runtimeMissing = RUNTIME_ASSETS.filter(
  (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
);

if (runtimeMissing.length > 0) {
  console.error(
    `verify-dist: ${runtimeMissing.length} ONNX runtime asset(s) missing — embed text and embed-stream pipelines will fail at startup:`,
  );
  for (const rel of runtimeMissing) console.error(`  - ${rel}`);
  console.error(
    "\nThese live under src/core/onnx/ in source. Make sure `npm run build` runs the",
  );
  fail('copy-onnx-assets step (see scripts/copy-onnx-assets.js).');
}

console.log(
  `verify-dist: ${cliUnique.length} dist require path(s) + ${RUNTIME_ASSETS.length} runtime asset(s) present.`,
);
