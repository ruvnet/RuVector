#!/usr/bin/env node
/**
 * Post-build step for the ESM output (dist/esm).
 *
 * The package root is CommonJS (no "type" field), so Node treats every .js
 * file as CJS unless a nearer package.json says otherwise. tsc does not emit
 * one, which made `import('ruvbot')` fail. This script:
 *
 *   1. writes dist/esm/package.json with {"type":"module"}
 *   2. verifies that every relative specifier in dist/esm resolves to a real
 *      file (ESM forbids extensionless and directory imports) and fails the
 *      build if one does not, so the bug cannot silently come back.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const esmDir = path.resolve(__dirname, '..', 'dist', 'esm');

if (!fs.existsSync(esmDir)) {
  console.error(`fix-esm: ${esmDir} does not exist; run build:esm first`);
  process.exit(1);
}

fs.writeFileSync(
  path.join(esmDir, 'package.json'),
  JSON.stringify({ type: 'module' }, null, 2) + '\n'
);

const SPECIFIER_RE =
  /(?:\bfrom\s*|\bimport\s*\(\s*|^\s*import\s+)(['"])(\.{1,2}\/[^'"]*)\1/gm;

function walk(dir, out) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.name.endsWith('.js')) out.push(full);
  }
  return out;
}

const problems = [];
for (const file of walk(esmDir, [])) {
  const src = fs.readFileSync(file, 'utf8');
  for (const match of src.matchAll(SPECIFIER_RE)) {
    const spec = match[2];
    const target = path.resolve(path.dirname(file), spec);
    let ok = false;
    try {
      ok = fs.statSync(target).isFile();
    } catch {
      ok = false;
    }
    if (!ok) problems.push(`${path.relative(esmDir, file)}: '${spec}'`);
  }
}

if (problems.length > 0) {
  console.error('fix-esm: unresolvable relative ESM specifiers (use explicit ".js" / "/index.js"):');
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}

console.log('fix-esm: wrote dist/esm/package.json; all relative ESM specifiers resolve');
