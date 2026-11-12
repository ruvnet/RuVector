import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Cloned in spirit from npm/packages/ruvector/test/mcp-command-security.js
 * (ADR-005): the decision path must never shell out, open a network client, or
 * log request/state text. Scans src/**\/*.ts and bin/cli.js only — never the
 * tests, which legitimately use fetch/execFileSync.
 */

const here = fileURLToPath(new URL('.', import.meta.url));
const root = join(here, '..');

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (full.endsWith('.ts')) out.push(full);
  }
  return out;
}

const files = [...walk(join(root, 'src')), join(root, 'bin', 'cli.js')];

const banned = [
  { re: /\brequire\(\s*['"]child_process['"]\s*\)/, why: 'no child_process require' },
  { re: /\bfrom\s+['"]child_process['"]/, why: 'no child_process import' },
  { re: /\bfrom\s+['"]node:child_process['"]/, why: 'no node:child_process import' },
  { re: /\bexecSync\s*\(/, why: 'no execSync' },
  { re: /\bexecFileSync\s*\(/, why: 'no execFileSync in the decision path' },
  { re: /\bexec\s*\(/, why: 'no exec()' },
  { re: /\bspawn\s*\(/, why: 'no spawn()' },
  { re: /\bshell\s*:\s*true\b/, why: 'no shell: true' },
  { re: /\bfetch\s*\(/, why: 'no fetch() in src/cli' },
  { re: /\bhttp\.request\s*\(/, why: 'no http client request' },
  { re: /\bhttps\b/, why: 'no https client use' },
  { re: /console\.log\([^)]*\b(state|request|body)\b/, why: 'never log state/request/body' },
];

for (const file of files) {
  const source = readFileSync(file, 'utf8');
  const rel = file.slice(root.length + 1);
  for (const { re, why } of banned) {
    test(`${rel}: ${why}`, () => {
      assert.doesNotMatch(source, re, `${rel} violates: ${why}`);
    });
  }
}

test('the server uses createServer, not an outbound client', () => {
  const serve = readFileSync(join(root, 'src', 'serve.ts'), 'utf8');
  assert.match(serve, /createServer\s*\(/, 'serve.ts should build an http server');
});

test('scan actually covered the source tree', () => {
  assert.ok(files.length >= 10, `expected to scan the src tree, saw ${files.length} files`);
});
