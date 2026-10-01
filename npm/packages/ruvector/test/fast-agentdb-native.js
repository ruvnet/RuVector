#!/usr/bin/env node

// A supported @ruvector/core release exports VectorDb without a VectorDB alias.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

async function runChild() {
  const { FastAgentDB } = require('../dist/core/agentdb-fast.js');
  const explicit = process.env.RUVECTOR_FAST_AGENTDB_TEST === 'explicit';
  const selectedPath = path.join(process.cwd(), 'owned', 'episodes.db');
  if (explicit) fs.mkdirSync(path.dirname(selectedPath), { recursive: true });
  const db = explicit ? new FastAgentDB(3, 10, selectedPath) : new FastAgentDB(3, 10);

  assert.deepStrictEqual(await db.searchByState([1, 0, 0], 1), []);
  assert.strictEqual(db.getStats().vectorDbAvailable, false, 'empty search opened an index');

  await db.storeEpisode({
    id: 'episode-1', state: [1, 0, 0], action: 'route', reward: 1,
    nextState: [0, 1, 0], done: true,
  });
  assert.strictEqual(db.getStats().vectorDbAvailable, true, 'native episode index never initialized');
  const indexed = await db.vectorDb.search({ vector: new Float32Array([1, 0, 0]), k: 1 });
  assert.strictEqual(indexed[0]?.id, 'episode-1', 'episode was not inserted into the native index');
  const matches = await db.searchByState([1, 0, 0], 1);
  assert.strictEqual(matches[0]?.episode.id, 'episode-1');
  assert.ok(!fs.existsSync('ruvector.db'), 'native index wrote into the caller cwd');
  if (explicit) assert.ok(fs.existsSync(selectedPath), 'explicit episode index path was ignored');
}

function runParent() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-fast-agentdb-test-'));
  try {
    for (const mode of ['explicit', 'default']) {
      const cwd = path.join(root, mode);
      fs.mkdirSync(cwd);
      const result = spawnSync(process.execPath, [__filename], {
        cwd,
        env: { ...process.env, RUVECTOR_FAST_AGENTDB_TEST: mode },
        encoding: 'utf8',
        timeout: 20_000,
      });
      assert.strictEqual(result.status, 0, `${mode} index failed:\n${result.stdout}\n${result.stderr}`);
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
  console.log('FastAgentDB native episode index and storage isolation passed');
}

if (process.env.RUVECTOR_FAST_AGENTDB_TEST) {
  runChild().catch((error) => { console.error(error); process.exitCode = 1; });
} else {
  runParent();
}
