/**
 * Issue #1217: `hooks graph-cluster` / `graph-mincut` keyed nodes by basename
 * and only followed `./x` imports, so same-named files in different packages
 * merged into one node and `../x` / subpath imports produced no edge.
 *
 * Fixture: two packages that each have a bytea.ts. pkg-a reaches shared/log
 * through `../../shared/log`; pkg-a/index imports `./util/helper`. The correct
 * graph has 7 nodes, 7 edges and splits into {pkg-a/*, shared/log} | {pkg-b/*}.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLI = path.join(__dirname, '..', 'bin', 'cli.js');

const FILES = {
  'pkg-a/index.ts': "import { x } from './bytea';\nimport { y } from './util/helper';\n",
  'pkg-a/bytea.ts': "import { y } from './util/helper.js';\n",
  'pkg-a/util/helper.ts': "import { s } from '../../shared/log';\n",
  'pkg-b/index.ts': "import { x } from './bytea';\nimport { z } from './codec';\n",
  'pkg-b/bytea.ts': "import { z } from './codec';\n",
  'pkg-b/codec.ts': 'export const z = 1;\n',
  'shared/log.ts': 'export const s = 1;\n',
};
const EXPECTED = [
  ['pkg-a/bytea', 'pkg-a/index', 'pkg-a/util/helper', 'shared/log'],
  ['pkg-b/bytea', 'pkg-b/codec', 'pkg-b/index'],
];
const norm = (groups) => groups.map((g) => [...g].sort().join(',')).sort();

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rv-graph-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  for (const [name, body] of Object.entries(FILES)) {
    fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    fs.writeFileSync(path.join(dir, name), body);
  }
  return dir;
}

const run = (dir, args) => JSON.parse(execFileSync(process.execPath, [CLI, 'hooks', ...args, ...Object.keys(FILES)],
  { cwd: dir, encoding: 'utf8', env: { ...process.env, NO_COLOR: '1' } }).trim().split('\n').pop());

for (const args of [['graph-cluster'], ['graph-cluster', '--method', 'spectral', '--clusters', '2'], ['graph-mincut']]) {
  test(`${args.join(' ')} resolves relative imports to distinct file nodes`, (t) => {
    const r = run(fixture(t), args);
    assert.equal(r.success, true, JSON.stringify(r));
    assert.equal(r.nodes, 7, `nodes: ${JSON.stringify(r.groups)}`);
    assert.equal(r.edges, 7);
    assert.deepEqual(norm(r.groups), norm(EXPECTED));
    assert.ok(r.modularity > 0.3 && r.modularity <= 1, `modularity ${r.modularity}`);
  });
}
