#!/usr/bin/env node
/**
 * `hooks rag-context --rerank` reorders the recalled memories.
 * Before: a silent no-op (pool never loaded or started, rankContext arguments swapped, results would lose ids/scores).
 * Needs the built dist/ (npm run build), like the other tests.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { ExtendedWorkerPool } = require('../dist/core/parallel-workers.js');
const { rerankMemories } = require('../bin/rerank.js');

let passed = 0, failed = 0;
async function t(name, fn) {
  try { await fn(); passed++; console.log(`  ok   ${name}`); }
  catch (e) { failed++; console.log(`  FAIL ${name}: ${e.message}`); }
}

const memories = [
  { id: 'm1', content: 'notes about the weather in the valley', score: 0.91, memory_type: 'general' },
  { id: 'm2', content: 'the zebra drawbridge opens at dawn; the zebra drawbridge keeper rings a bell', score: 0.88, memory_type: 'lesson' },
  { id: 'm3', content: 'counting boats on the river', score: 0.85, memory_type: 'general' },
];

(async () => {
  console.log('rag-context --rerank');
  await t('reorders by relevance to the query: the zebra memory first', async () => {
    const r = await rerankMemories(memories, 'zebra drawbridge', ExtendedWorkerPool);
    assert.strictEqual(r.reranked, true, `reranked=${r.reranked} ${r.error || ''}`);
    assert.strictEqual(r.results[0].id, 'm2');
  });
  await t('keeps every field of each memory (id, score, type) and adds relevance', async () => {
    const r = await rerankMemories(memories, 'zebra drawbridge', ExtendedWorkerPool);
    const m2 = r.results.find((m) => m.id === 'm2');
    assert.deepStrictEqual([m2.score, m2.memory_type, m2.content], [0.88, 'lesson', memories[1].content]);
    assert.ok(m2.relevance > 0 && r.results.every((m) => typeof m.relevance === 'number'));
    assert.deepStrictEqual(r.results.map((m) => m.id).sort(), ['m1', 'm2', 'm3']);
  });
  await t('no query term matches: the recall order is kept', async () => {
    const r = await rerankMemories(memories, 'quantum', ExtendedWorkerPool);
    assert.deepStrictEqual(r.results.map((m) => m.id), ['m1', 'm2', 'm3']);
  });
  await t('string memories (older stores) become {content, relevance}', async () => {
    const r = await rerankMemories(['plain text', 'zebra text'], 'zebra', ExtendedWorkerPool);
    assert.deepStrictEqual(r.results.map((m) => m.content), ['zebra text', 'plain text']);
  });
  await t('no worker pool: the memories come back unchanged and the output says it did not rerank', async () => {
    const r = await rerankMemories(memories, 'zebra', undefined);
    assert.strictEqual(r.reranked, false);
    assert.strictEqual(r.results, memories);
  });
  await t('works without RUVECTOR_PARALLEL=1 (the pool is disabled by default)', async () => {
    assert.notStrictEqual(process.env.RUVECTOR_PARALLEL, '1');
    const r = await rerankMemories(memories, 'zebra', ExtendedWorkerPool);
    assert.strictEqual(r.reranked, true, r.error);
  });
  await t('cli.js wires it: rag-context loads the modules and calls rerankMemories', async () => {
    const src = fs.readFileSync(path.join(__dirname, '..', 'bin', 'cli.js'), 'utf8');
    const block = src.slice(src.indexOf("command('rag-context')"), src.indexOf("command('git-churn')"));
    assert.ok(/loadNewModules\(\)/.test(block) && /rerankMemories\(memories, queryStr, ExtendedWorkerPool\)/.test(block));
  });
  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
