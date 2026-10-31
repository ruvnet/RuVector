/**
 * intelligence.json storage: compact serialization that stays plain JSON
 * (readable by the Rust CLI and @ruvector/cli) and project-scoped store
 * resolution shared by the CLI and MCP server.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const store = require('../bin/intel-store.js');

test('serializeStore keeps embeddings as number arrays within float32 precision', () => {
  const emb = Array.from({ length: 384 }, () => Math.random() * 2 - 1);
  const data = { memories: Array.from({ length: 50 }, (_, i) => ({ id: `m${i}`, embedding: emb.slice() })) };
  const compact = store.serializeStore(data);
  assert.ok(compact.length * 2 < JSON.stringify(data, null, 2).length, 'not smaller');
  assert.strictEqual(data.memories[0].embedding[0], emb[0], 'in-memory data mutated');
  const back = JSON.parse(compact);
  assert.ok(Array.isArray(back.memories[0].embedding));
  for (let i = 0; i < 384; i++) assert.ok(Math.abs(back.memories[0].embedding[i] - emb[i]) < 1e-6);
});

test('a repository root gets its own store even when ~/.ruvector exists', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-store-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const home = path.join(root, 'home');
  const proj = path.join(root, 'proj');
  fs.mkdirSync(path.join(home, '.ruvector'), { recursive: true });
  fs.writeFileSync(path.join(home, '.ruvector', 'intelligence.json'), '{}');
  fs.mkdirSync(proj);
  fs.writeFileSync(path.join(proj, 'package.json'), '{}');
  const env = { ...process.env, HOME: home, USERPROFILE: home, NO_COLOR: '1' };
  execFileSync(process.execPath, [CLI, 'hooks', 'remember', 'project fact', '-t', 'project'], { cwd: proj, env, stdio: 'ignore' });
  assert.ok(fs.existsSync(path.join(proj, '.ruvector', 'intelligence.json')), 'project store not created');
  assert.equal(fs.readFileSync(path.join(home, '.ruvector', 'intelligence.json'), 'utf8'), '{}', 'home store was written');
});
