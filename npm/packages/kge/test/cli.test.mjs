// CLI tests for `kge optimize`, run against the compiled dist/ CLI with the
// deterministic fake binding injected (no native build, no network, no spawn).
// Verifies --budget passthrough, --receipts / --out file writing, and that the
// printed summary omits the (large) receipts blob.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const testDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(testDir, '..');
const mainPath = join(pkgDir, 'dist', 'cli', 'main.js');
const built = existsSync(mainPath);

const require = createRequire(import.meta.url);
const fakeBinding = require('./fixtures/fake-binding.cjs');

test('kge optimize writes receipts + model and prints a summary', { skip: !built && 'run npm run build first' }, async () => {
  const { main } = require(mainPath);
  const dir = mkdtempSync(join(tmpdir(), 'kge-cli-'));
  try {
    const receipts = join(dir, 'receipts.jsonl');
    const out = join(dir, 'model.json');
    const lines = [];
    const result = await main(
      ['optimize', '--scorer', 'hole', '--dims', '8', '--budget', '2', '--receipts', receipts, '--out', out],
      { binding: fakeBinding, stdout: (s) => lines.push(s), stderr: (s) => lines.push(s) },
    );
    assert.equal(result.code, 0, 'optimize exits 0');

    assert.ok(existsSync(receipts), 'receipts file written');
    assert.ok(readFileSync(receipts, 'utf8').includes('promote'), 'receipts JSONL has content');
    assert.ok(existsSync(out), 'model envelope written');

    const summary = JSON.parse(lines.join('\n'));
    assert.equal(summary.promoted, true, 'summary carries the report');
    assert.equal(summary.champion.scorer, 'hole', 'summary carries champion knobs');
    assert.equal(summary.receiptsWritten, receipts, 'summary notes the receipts path');
    assert.equal(summary.saved, out, 'summary notes the saved model');
    assert.equal(summary.receipts, undefined, 'the large receipts blob is not printed');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
