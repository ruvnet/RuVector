#!/usr/bin/env node
/**
 * verify-dist.js — pre-publish gate that fails the build if any file
 * `bin/cli.js` requires from `../dist/...` is missing.
 *
 * Why: 0.2.23 was published without a `dist/` directory at all (issue #399),
 * which silently broke `ruvector doctor`, the entire `embed` subsystem, and
 * `rvf` commands. tsc was supposed to run via `prepublishOnly`, but the
 * hook didn't fire (or the build failed silently). This script makes the
 * publish itself fail loudly when the artifact is incomplete.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');
const cliPath = path.join(pkgRoot, 'bin', 'cli.js');

if (!fs.existsSync(cliPath)) {
  console.error('verify-dist: bin/cli.js not found — package layout is broken.');
  process.exit(1);
}

const cliSource = fs.readFileSync(cliPath, 'utf8');

// Collect every `require('../dist/...')` referenced by the CLI.
const distRequires = Array.from(
  cliSource.matchAll(/require\(['"]\.\.\/(dist\/[^'"]+\.js)['"]\)/g),
  (m) => m[1],
);
const unique = Array.from(new Set(distRequires)).sort();

const missing = unique.filter(
  (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
);

if (missing.length > 0) {
  console.error(
    `verify-dist: ${missing.length} dist file(s) referenced by bin/cli.js are missing:`,
  );
  for (const rel of missing) {
    console.error(`  - ${rel}`);
  }
  console.error(
    "\nRun `npm run build` and confirm tsc emitted under dist/. If a path was renamed,",
  );
  console.error('update bin/cli.js to match.');
  process.exit(1);
}

console.log(`verify-dist: ${unique.length} dist path(s) present.`);
