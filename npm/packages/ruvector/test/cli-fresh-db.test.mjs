// End-to-end CLI smoke for issue #417 — verifies that `ruvector create`,
// `insert`, `search`, and `stats` work on a fresh database (the old
// implementation crashed on every command after `create` because it
// JSON.parse()d the redb binary file).
//
// Run with:
//
//   node test/cli-fresh-db.test.mjs

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mkdtempSync, rmSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, '..');
const cli = join(repoRoot, 'bin/cli.js');
const tmp = mkdtempSync(join(tmpdir(), 'ruvector-417-'));
const dbPath = join(tmp, 'test.db');

let failures = 0;
function check(cond, msg, extra) {
  if (!cond) {
    console.error('FAIL:', msg);
    if (extra) console.error('  ', extra);
    failures++;
  } else {
    console.log('  ok:', msg);
  }
}

function runCli(args, env = {}) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, ...env },
  });
}

try {
  // 1. create — should succeed AND drop a sidecar.
  let res = runCli(['create', dbPath, '-d', '8', '-m', 'cosine']);
  check(res.status === 0, '`ruvector create` exits 0', res.stderr || res.stdout);
  check(existsSync(dbPath), 'redb file exists at dbPath');
  const sidecar = `${dbPath}.meta.json`;
  check(existsSync(sidecar), 'sidecar metadata file exists');
  const meta = JSON.parse(readFileSync(sidecar, 'utf8'));
  check(meta.dimensions === 8, 'sidecar.dimensions = 8');
  check(meta.metric === 'cosine', 'sidecar.metric = cosine');

  // 2. insert — should NOT crash with `Unexpected token 'r'` from JSON.parse(redb).
  const vectorsPath = join(tmp, 'vecs.json');
  const vectors = [
    { id: 'a', vector: [1, 0, 0, 0, 0, 0, 0, 0] },
    { id: 'b', vector: [0, 1, 0, 0, 0, 0, 0, 0] },
    { id: 'c', vector: [0, 0, 1, 0, 0, 0, 0, 0] },
  ];
  writeFileSync(vectorsPath, JSON.stringify(vectors));
  res = runCli(['insert', dbPath, vectorsPath]);
  check(res.status === 0, '`ruvector insert` exits 0', res.stderr || res.stdout);
  check(
    !res.stderr.includes('Unexpected token') && !res.stdout.includes('Unexpected token'),
    'insert does not crash JSON.parsing the redb binary',
    res.stderr || res.stdout,
  );

  // 3. search — should NOT crash, should return at least one hit.
  res = runCli([
    'search',
    dbPath,
    '-v',
    JSON.stringify([1, 0, 0, 0, 0, 0, 0, 0]),
    '-k',
    '3',
  ]);
  const searchOut = res.stdout + res.stderr;
  check(res.status === 0, '`ruvector search` exits 0', res.stderr || res.stdout);
  check(
    /Found\s+\d+\s+results?/.test(searchOut),
    'search prints `Found N results` (across stdout/stderr)',
    searchOut,
  );
  check(
    res.stdout.includes('ID: a') || res.stdout.includes('ID: b'),
    'search renders at least one hit row',
    res.stdout,
  );

  // 4. stats — should NOT crash, should report Vector Count.
  res = runCli(['stats', dbPath]);
  check(res.status === 0, '`ruvector stats` exits 0', res.stderr || res.stdout);
  check(res.stdout.includes('Vector Count'), 'stats prints Vector Count', res.stdout);

  // 5. helpful error when sidecar is absent (regression guard for the
  //    "user constructs DB without create" path).
  const orphanDb = join(tmp, 'orphan.db');
  writeFileSync(orphanDb, 'redb-fake-binary'); // pretend a redb file existed
  res = runCli(['stats', orphanDb]);
  check(res.status !== 0, 'stats fails fast on orphan DB without sidecar');
  check(
    (res.stderr + res.stdout).includes('sidecar'),
    'orphan-DB error message mentions sidecar',
    res.stderr || res.stdout,
  );
} finally {
  try { rmSync(tmp, { recursive: true, force: true }); } catch {}
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log(`\nruvector fresh-DB CLI smoke OK (issue #417)`);
