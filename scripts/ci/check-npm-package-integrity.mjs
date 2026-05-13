#!/usr/bin/env node
// Regression guard for npm publish hygiene.
//
// For each package listed below, this script:
//   1. runs `npm pack` to produce the exact tarball that would be published
//   2. extracts it to a temp dir
//   3. asserts every path in `requiredFiles` is present in the tarball
//   4. runs `node --check` on every JS/MJS/CJS entry point reachable via
//      package.json `main` / `module` / `bin` / `exports`
//   5. attempts to actually load the main entry (require for CJS, import for
//      ESM) and FAILS on packaging errors (SyntaxError, ERR_REQUIRE_ESM,
//      ERR_PACKAGE_PATH_NOT_EXPORTED, MODULE_NOT_FOUND for a bundled file).
//      Runtime errors that are NOT packaging problems (e.g. a native .node
//      addon missing on this host, "fetch is not defined" in a browser path)
//      are tolerated.
//
// This exists because issues #354, #372, #376, #415, #417 all shipped to npm
// as "the published artifact is structurally broken" — every one of them would
// have been caught by steps 3–5 above. Add new packages / required files here
// whenever a publish-hygiene bug is fixed so it can never regress.

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

/**
 * @typedef {Object} PkgSpec
 * @property {string} dir            path to the package, relative to repo root
 * @property {string[]} requiredFiles paths (relative to package root) that MUST
 *                                    be present in the published tarball
 * @property {boolean} [loadMain]    if false, skip the load() step (e.g. the
 *                                    package's only entry is a native binary)
 */

/** @type {PkgSpec[]} */
const PACKAGES = [
  {
    dir: 'npm/packages/ruvector',
    requiredFiles: [
      'dist/index.js',
      'bin/cli.js',
      'bin/mcp-server.js',
      // ONNX runtime assets — regression guard for #354 / #323
      'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.wasm',
      'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm.js',
      'dist/core/onnx/pkg/ruvector_onnx_embeddings_wasm_bg.js',
      'dist/core/onnx/pkg/loader.js',
    ],
  },
  {
    dir: 'npm/packages/rvf-wasm',
    // After #415: ESM-only package; the .mjs must be present and must NOT
    // transitively re-import a CJS-classified .js that uses import.meta.
    requiredFiles: ['pkg/rvf_wasm.mjs', 'pkg/rvf_wasm_bg.wasm'],
  },
  // NOTE: @ruvector/pi-brain's #372 regression is guarded in the
  // `mcp-server-loads` workflow job (it must be loaded via `await import`, not
  // `require`) rather than here — pi-brain ships TS-only and only materialises
  // `dist/` via `tsc`, so packing it requires a build step that doesn't belong
  // in this lightweight script.
];

let failures = 0;
const fail = (pkg, msg) => {
  failures++;
  console.error(`  ✗ [${pkg}] ${msg}`);
};
const ok = (pkg, msg) => console.log(`  ✓ [${pkg}] ${msg}`);

// Errors that mean "this package is structurally broken for consumers".
const PACKAGING_ERROR_CODES = new Set([
  'ERR_REQUIRE_ESM',
  'ERR_PACKAGE_PATH_NOT_EXPORTED',
  'ERR_UNSUPPORTED_DIR_IMPORT',
  'ERR_MODULE_NOT_FOUND',
]);
function isPackagingError(err) {
  if (!err) return false;
  if (err instanceof SyntaxError) return true;
  if (err.code && PACKAGING_ERROR_CODES.has(err.code)) return true;
  // A bundled file that should have been in `files` is missing.
  if (err.code === 'MODULE_NOT_FOUND' && !/node_modules/.test(String(err.message))) return true;
  return false;
}

function packAndExtract(pkgAbsDir, scratch) {
  // `npm pack --json` prints [{ filename, files: [{path}], ... }]
  const out = execFileSync('npm', ['pack', '--json', '--pack-destination', scratch], {
    cwd: pkgAbsDir,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const meta = JSON.parse(out)[0];
  const tarball = join(scratch, meta.filename);
  const extractDir = join(scratch, 'extracted');
  execFileSync('mkdir', ['-p', extractDir]);
  execFileSync('tar', ['-xzf', tarball, '-C', extractDir, '--strip-components=1']);
  return { extractDir, fileList: meta.files.map((f) => f.path.replace(/^\.\//, '')) };
}

function entryPoints(pkgJson) {
  const eps = new Set();
  const add = (v) => {
    if (typeof v === 'string') eps.add(v);
    else if (v && typeof v === 'object') for (const sub of Object.values(v)) add(sub);
  };
  add(pkgJson.main);
  add(pkgJson.module);
  add(pkgJson.bin);
  add(pkgJson.exports);
  // types / .d.ts files are not executable
  return [...eps].filter((p) => /\.(c|m)?js$/.test(p));
}

async function tryLoadMain(extractDir, pkgJson) {
  const isEsm = pkgJson.type === 'module';
  const mainRel = (pkgJson.exports && pkgJson.exports['.'] &&
    (pkgJson.exports['.'].require || pkgJson.exports['.'].import || pkgJson.exports['.'].default)) ||
    pkgJson.main || pkgJson.module;
  if (!mainRel) return; // nothing declared
  const mainAbs = join(extractDir, mainRel);
  if (!existsSync(mainAbs)) {
    throw Object.assign(new Error(`declared entry "${mainRel}" missing from tarball`), { code: 'MODULE_NOT_FOUND' });
  }
  if (isEsm || /\.mjs$/.test(mainRel)) {
    await import(pathToFileURL(mainAbs).href);
  } else {
    const require = (await import('node:module')).createRequire(import.meta.url);
    require(mainAbs);
  }
}

for (const spec of PACKAGES) {
  const pkgAbsDir = join(REPO_ROOT, spec.dir);
  if (!existsSync(join(pkgAbsDir, 'package.json'))) {
    fail(spec.dir, 'package.json not found — did the path change?');
    continue;
  }
  const pkgJson = JSON.parse(readFileSync(join(pkgAbsDir, 'package.json'), 'utf8'));
  console.log(`\n── ${pkgJson.name}@${pkgJson.version}  (${spec.dir})`);

  const scratch = mkdtempSync(join(tmpdir(), 'pkgcheck-'));
  try {
    let extractDir, fileList;
    try {
      ({ extractDir, fileList } = packAndExtract(pkgAbsDir, scratch));
    } catch (e) {
      fail(pkgJson.name, `npm pack failed: ${e.message}`);
      continue;
    }
    const fileSet = new Set(fileList);

    for (const req of spec.requiredFiles) {
      if (fileSet.has(req) || existsSync(join(extractDir, req))) ok(pkgJson.name, `tarball contains ${req}`);
      else fail(pkgJson.name, `tarball is MISSING required file: ${req}`);
    }

    for (const ep of entryPoints(pkgJson)) {
      const epAbs = join(extractDir, ep);
      if (!existsSync(epAbs)) { fail(pkgJson.name, `entry point declared in package.json but not in tarball: ${ep}`); continue; }
      try {
        execFileSync(process.execPath, ['--check', epAbs], { stdio: 'pipe' });
        ok(pkgJson.name, `node --check ${ep}`);
      } catch (e) {
        fail(pkgJson.name, `node --check FAILED for ${ep}: ${String(e.stderr || e.message).trim().split('\n').slice(-3).join(' ')}`);
      }
    }

    if (spec.loadMain !== false) {
      try {
        await tryLoadMain(extractDir, pkgJson);
        ok(pkgJson.name, 'main entry loads without packaging errors');
      } catch (e) {
        if (isPackagingError(e)) fail(pkgJson.name, `loading main entry threw a PACKAGING error: ${e.code || e.name}: ${e.message}`);
        else ok(pkgJson.name, `main entry load threw a non-packaging runtime error (tolerated): ${e.code || e.name}`);
      }
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

console.log('');
if (failures > 0) {
  console.error(`✗ npm package integrity check FAILED with ${failures} problem(s).`);
  process.exit(1);
}
console.log('✓ npm package integrity check passed.');
