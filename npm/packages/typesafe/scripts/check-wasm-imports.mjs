#!/usr/bin/env node
// Focused ADR-005 (c) check: the wasm32 build links no WASI and nothing named
// fs/net/socket/fetch in its import section. Run in CI right after the wasm
// build (`node scripts/check-wasm-imports.mjs <path.wasm>`). Reuses the wasm
// logic in check-security.mjs. Exits non-zero on any forbidden import.

import { existsSync, statSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkWasm } from './check-security.mjs';

// Default to the package's wasm/ output dir so the no-arg CI form
// (`node scripts/check-wasm-imports.mjs`) works alongside an explicit path.
const DEFAULT_WASM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'wasm');

function findWasm(arg) {
  const path = arg || DEFAULT_WASM_DIR;
  if (existsSync(path)) {
    if (statSync(path).isDirectory()) {
      const hit = readdirSync(path).find((f) => f.endsWith('.wasm'));
      return hit ? join(path, hit) : null;
    }
    return path;
  }
  return null;
}

const target = findWasm(process.argv[2]);
if (!target) {
  console.error(`no .wasm found at ${process.argv[2] ?? '(no path given)'}`);
  process.exit(2);
}
const res = checkWasm(target);
console.log(`wasm import check: ${target} — ${res.scanned} imports`);
if (res.violations.length) {
  console.error('FORBIDDEN wasm imports:');
  for (const v of res.violations) console.error(`  ${v.pattern}`);
  process.exit(1);
}
console.log('no WASI / fs / net / socket / fetch imports — clear');
