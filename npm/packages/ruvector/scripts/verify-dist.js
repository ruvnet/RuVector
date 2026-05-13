#!/usr/bin/env node
/**
 * verify-dist.js — pre-publish gate that fails the build if the published
 * artifact would be incomplete.
 *
 * History:
 *  - 0.2.23 was published without a `dist/` directory at all (issue #399),
 *    which silently broke `ruvector doctor`, the entire `embed` subsystem,
 *    and `rvf` commands.
 *  - A later publish slipped without `dist/index.js` (issue #376).
 *  - Clean installs were missing the ONNX WASM runtime assets (issue #354).
 *
 * tsc was supposed to run via `prepublishOnly`, but the hook didn't always
 * fire (or the build failed silently). This script makes the publish itself
 * fail loudly when the artifact is incomplete. It runs AFTER `npm run build`.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');

function fail(lines) {
  for (const line of lines) console.error(line);
  process.exit(1);
}

// 1. Core entry points + ONNX runtime assets that MUST exist regardless of
//    what bin/cli.js references.
const REQUIRED = [
  'dist/index.js',
  'dist/index.d.ts',
  'bin/cli.js',
  // ONNX WASM runtime assets (issue #354) — clean installs need these to embed.
  'dist/core/onnx/loader.js',
  'dist/core/onnx/pkg/loader.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.js',
  'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.wasm',
  'dist/core/onnx/pkg/package.json',
];

const missingRequired = REQUIRED.filter(
  (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
);
if (missingRequired.length > 0) {
  fail([
    `verify-dist: ${missingRequired.length} required artifact(s) missing:`,
    ...missingRequired.map((rel) => `  - ${rel}`),
    '',
    'Run `npm run build` (tsc + scripts/copy-onnx.js) and re-check.',
  ]);
}

// 2. Every `require('../dist/...')` referenced by bin/cli.js must resolve.
const cliPath = path.join(pkgRoot, 'bin', 'cli.js');
const cliSource = fs.readFileSync(cliPath, 'utf8');
const distRequires = Array.from(
  cliSource.matchAll(/require\(['"]\.\.\/(dist\/[^'"]+\.js)['"]\)/g),
  (m) => m[1],
);
const uniqueDistRequires = Array.from(new Set(distRequires)).sort();
const missingDist = uniqueDistRequires.filter(
  (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
);
if (missingDist.length > 0) {
  fail([
    `verify-dist: ${missingDist.length} dist file(s) referenced by bin/cli.js are missing:`,
    ...missingDist.map((rel) => `  - ${rel}`),
    '',
    'Run `npm run build` and confirm tsc emitted under dist/. If a path was renamed,',
    'update bin/cli.js to match.',
  ]);
}

// 3. The npm tarball must actually carry `dist/` — i.e. it must be in `files`.
const pkgJson = require(path.join(pkgRoot, 'package.json'));
const files = Array.isArray(pkgJson.files) ? pkgJson.files : [];
const hasDist = files.some((f) => f === 'dist' || f === 'dist/' || f.startsWith('dist/'));
if (!hasDist) {
  fail([
    'verify-dist: package.json "files" does not include "dist/" — the built',
    'artifact would not be published. Add "dist/" to the files array.',
  ]);
}

console.log(
  `verify-dist: ${REQUIRED.length} required artifact(s) + ${uniqueDistRequires.length} cli dist path(s) present; "dist/" is published.`,
);
