import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * ADR-005: the query path must never shell out, open a network client, or log
 * triple/label text. Scans src/**\/*.ts and bin/cli.js only — not the tests,
 * which legitimately use fetch/require. `serve.ts` builds an http *server* and
 * is exempt from the outbound-client patterns.
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

const isServer = (file) => file.endsWith('serve.ts');

const banned = [
  { re: /\brequire\(\s*['"]child_process['"]\s*\)/, why: 'no child_process require' },
  { re: /\bfrom\s+['"](?:node:)?child_process['"]/, why: 'no child_process import' },
  { re: /\bexecSync\s*\(/, why: 'no execSync' },
  { re: /\bexecFileSync\s*\(/, why: 'no execFileSync' },
  { re: /\bspawn\s*\(/, why: 'no spawn()' },
  { re: /\bshell\s*:\s*true\b/, why: 'no shell: true' },
  { re: /\bfetch\s*\(/, why: 'no fetch() in src/cli' },
  { re: /console\.log\([^)]*\b(triple|triples|label|state|body)\b/, why: 'never log triple/label/body text' },
];

// Outbound-network patterns: banned everywhere except the server (serve.ts).
const networkBanned = [
  { re: /\bhttp\.request\s*\(/, why: 'no http client request' },
  { re: /\bnet\.connect\s*\(/, why: 'no raw socket connect' },
];

for (const file of files) {
  const source = readFileSync(file, 'utf8');
  const rel = file.slice(root.length + 1);
  for (const { re, why } of banned) {
    test(`${rel}: ${why}`, () => {
      assert.doesNotMatch(source, re, `${rel} violates: ${why}`);
    });
  }
  if (!isServer(file)) {
    for (const { re, why } of networkBanned) {
      test(`${rel}: ${why}`, () => {
        assert.doesNotMatch(source, re, `${rel} violates: ${why}`);
      });
    }
  }
}

test('the server uses createServer, not an outbound client', () => {
  const serve = readFileSync(join(root, 'src', 'serve.ts'), 'utf8');
  assert.match(serve, /createServer\s*\(/, 'serve.ts should build an http server');
});

test('scan actually covered the source tree', () => {
  assert.ok(files.length >= 10, `expected to scan the src tree, saw ${files.length} files`);
});
