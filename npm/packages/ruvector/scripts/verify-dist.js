#!/usr/bin/env node
/**
 * verify-dist.js — pre-publish gate that fails the build if the published
 * tarball would be unusable.
 *
 * History:
 *   - #399: 0.2.23 published without `dist/` at all — `ruvector doctor`,
 *           `embed`, and `rvf` commands all crashed because tsc didn't
 *           run via `prepublishOnly` (or failed silently).
 *   - #376: same release (0.2.23) — `main: dist/index.js` was declared but
 *           the tarball didn't contain it, so any consumer doing
 *           `require('ruvector')` or `await import('ruvector')` blew up
 *           on a fresh install.
 *
 * Both regressions are guarded here. The script asserts:
 *   1. Every `require('../dist/...')` in `bin/cli.js` resolves to a file.
 *   2. `package.json#main`, `types`, and every `bin` entry resolve to a
 *      file (this is what would have caught #376).
 *   3. Optional smoke check: `node -e "require('<main>')"` succeeds when
 *      run with the repo as cwd. Skipped under VERIFY_DIST_SKIP_SMOKE=1
 *      so a hot fix can publish even if a peer dep is unsatisfied.
 */

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const pkgRoot = path.resolve(__dirname, '..');
const pkgJsonPath = path.join(pkgRoot, 'package.json');
const cliPath = path.join(pkgRoot, 'bin', 'cli.js');

if (!fs.existsSync(pkgJsonPath)) {
  console.error('verify-dist: package.json not found — package layout is broken.');
  process.exit(1);
}
const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));

const errors = [];

// 1. cli.js dist requires (#399).
if (!fs.existsSync(cliPath)) {
  errors.push(`bin/cli.js not found at ${path.relative(pkgRoot, cliPath)}`);
} else {
  const cliSource = fs.readFileSync(cliPath, 'utf8');
  const distRequires = Array.from(
    cliSource.matchAll(/require\(['"]\.\.\/(dist\/[^'"]+\.js)['"]\)/g),
    (m) => m[1],
  );
  const unique = Array.from(new Set(distRequires)).sort();
  const missing = unique.filter(
    (rel) => !fs.existsSync(path.join(pkgRoot, rel)),
  );
  if (missing.length > 0) {
    errors.push(
      `${missing.length} dist file(s) referenced by bin/cli.js are missing:\n` +
        missing.map((r) => `    - ${r}`).join('\n'),
    );
  } else {
    console.log(`verify-dist: ${unique.length} dist path(s) referenced by bin/cli.js present.`);
  }
}

// 2. package.json entrypoints (#376).
const entrypointFields = [];
if (typeof pkg.main === 'string') entrypointFields.push(['main', pkg.main]);
if (typeof pkg.types === 'string') entrypointFields.push(['types', pkg.types]);
if (typeof pkg.module === 'string') entrypointFields.push(['module', pkg.module]);
if (pkg.bin && typeof pkg.bin === 'object') {
  for (const [name, rel] of Object.entries(pkg.bin)) {
    if (typeof rel === 'string') entrypointFields.push([`bin.${name}`, rel]);
  }
} else if (typeof pkg.bin === 'string') {
  entrypointFields.push(['bin', pkg.bin]);
}

const missingEntries = entrypointFields.filter(
  ([, rel]) => !fs.existsSync(path.join(pkgRoot, rel)),
);
if (missingEntries.length > 0) {
  errors.push(
    `${missingEntries.length} package.json entrypoint(s) point at missing files:\n` +
      missingEntries.map(([f, r]) => `    - ${f} → ${r}`).join('\n'),
  );
} else {
  console.log(`verify-dist: ${entrypointFields.length} package.json entrypoint(s) present.`);
}

// 3. Smoke: actually require the main entry (catches "file present but
//    syntactically broken" regressions). Skipped if main is missing —
//    error already reported above.
if (
  !process.env.VERIFY_DIST_SKIP_SMOKE &&
  typeof pkg.main === 'string' &&
  fs.existsSync(path.join(pkgRoot, pkg.main))
) {
  const mainAbs = path.join(pkgRoot, pkg.main);
  const result = spawnSync(
    process.execPath,
    ['-e', `require(${JSON.stringify(mainAbs)})`],
    { cwd: pkgRoot, encoding: 'utf8' },
  );
  if (result.status !== 0) {
    errors.push(
      `\`require('${pkg.main}')\` failed (exit ${result.status}):\n` +
        (result.stderr || result.stdout || '<no output>')
          .split('\n')
          .slice(0, 6)
          .map((l) => `    ${l}`)
          .join('\n'),
    );
  } else {
    console.log(`verify-dist: require('${pkg.main}') smoke OK.`);
  }
}

if (errors.length > 0) {
  console.error('\nverify-dist: package would publish broken:\n');
  for (const e of errors) console.error(`  - ${e}`);
  console.error(
    '\nRun `npm run build` and confirm tsc emitted under dist/. If a path was renamed,',
  );
  console.error('update package.json or bin/cli.js to match.');
  process.exit(1);
}
