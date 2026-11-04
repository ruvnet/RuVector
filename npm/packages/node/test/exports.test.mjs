// Every entry point in package.json "exports" must resolve, for both ESM
// `import` and CJS `require`, and expose the documented API.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

const ROOT_API = ['VectorDB', 'RuvectorLayer', 'TensorCompress', 'differentiableSearch',
  'hierarchicalForward', 'getCompressionLevel', 'initGnn'];
const GNN_API = ['RuvectorLayer', 'TensorCompress', 'differentiableSearch',
  'hierarchicalForward', 'getCompressionLevel', 'init'];

function assertApi(mod, names, label) {
  for (const n of names) assert.equal(typeof mod[n], 'function', `${label}.${n}`);
}

test('every file referenced by package.json exists after build', () => {
  const targets = [pkg.main, pkg.types, pkg.module].filter(Boolean);
  for (const entry of Object.values(pkg.exports)) targets.push(...Object.values(entry));
  for (const t of targets) assert.ok(existsSync(new URL(`../${t}`, import.meta.url)), `missing ${t}`);
});

test('"types" is the first condition of every export (TypeScript picks the first match)', () => {
  for (const [sub, entry] of Object.entries(pkg.exports)) {
    assert.equal(Object.keys(entry)[0], 'types', `exports["${sub}"]`);
  }
});

test('ESM import of the root entry exposes named and default exports', async () => {
  const m = await import('@ruvector/node');
  assertApi(m, ROOT_API, 'import');
  assertApi(m.default, ROOT_API, 'import default');
});

test('ESM import of ./gnn exposes named exports', async () => {
  assertApi(await import('@ruvector/node/gnn'), GNN_API, 'import gnn');
});

test('CJS require of both entries exposes the API', () => {
  assertApi(require('@ruvector/node'), ROOT_API, 'require');
  assertApi(require('@ruvector/node/gnn'), GNN_API, 'require gnn');
});
