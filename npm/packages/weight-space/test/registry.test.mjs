import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FlatVectorStore, RuvectorVectorStore, WeightRegistry, createMlp, setLayer } from '../dist/esm/index.js';

const tmp = () => mkdtempSync(join(tmpdir(), 'ws-test-'));

for (const store of ['flat', 'ruvector-core']) {
  test(`[${store}] save / get / nearest / lineage / outcomes`, async () => {
    const reg = await WeightRegistry.open({ dimensions: 4, store });
    assert.equal(reg.storeKind, store);
    const m = createMlp([2, 3, 1], { seed: 1 });
    const a = await reg.save(m, [1, 0, 0, 0], { task: 'and', tags: ['logic'], outcomes: [0.9] });
    const b = await reg.save(createMlp([2, 3, 1], { seed: 2 }), [0, 1, 0, 0], { task: 'or' });
    const c = await reg.save(createMlp([2, 5, 1], { seed: 3 }), [0.9, 0.1, 0, 0], { task: 'and-wide' });
    assert.equal(a.version, 1);
    assert.equal(a.hash.length, 64);
    assert.deepEqual(a.topology.sizes, [2, 3, 1]);
    assert.deepEqual(reg.get(a.id).outcomes.map((o) => o.score), [0.9]);
    assert.equal(reg.get('nope'), undefined);

    const hits = await reg.nearest([1, 0.05, 0, 0], 2);
    assert.deepEqual(hits.map((h) => h.id), [a.id, c.id]);
    assert.ok(hits[0].similarity > hits[1].similarity);
    const compatible = await reg.nearest([0.9, 0.1, 0, 0], 1, { compatibleWith: m });
    assert.equal(compatible[0].id, a.id, 'topology filter skips the closer but incompatible snapshot');

    const child = await reg.setLayer(a.id, 1, { bias: [0.25] }, { note: 'bias bump' });
    const grandchild = await reg.perturb(child.id, { scale: 0.01, seed: 9 });
    assert.equal(child.version, 2);
    assert.equal(grandchild.version, 3);
    assert.equal(child.parent, a.id);
    assert.equal(child.op, 'setLayer');
    assert.deepEqual(reg.lineage(grandchild.id).map((r) => r.id), [a.id, child.id, grandchild.id]);
    assert.deepEqual(reg.childrenOf(a.id).map((r) => r.id), [child.id]);
    assert.equal(reg.diff(a.id, child.id).changed, 1);
    assert.equal(reg.get(a.id).model.layers[1].bias[0], 0, 'parent unchanged');
    assert.ok(Object.isFrozen(reg.get(a.id).model.layers[0].weights), 'snapshots are frozen');

    const [top] = await reg.nearest([1, 0, 0, 0], 1, { compatibleWith: m });
    assert.equal(top.id, grandchild.id, 'similarity ties (shared embedding) resolve to the newest version');

    await reg.recordOutcome(b.id, 0.2, 'failed run');
    assert.equal(reg.get(b.id).outcomes[0].note, 'failed run');
    assert.equal((await reg.save(m, [1, 0, 0, 0], { task: 'and' })).id, a.id, 'identical save is idempotent');
    await reg.close();
  });
}

test('validation at the boundary', async () => {
  const reg = await WeightRegistry.open({ dimensions: 3, store: 'flat' });
  const m = createMlp([2, 1]);
  await assert.rejects(reg.save(m, [1, 0]), (e) => e.code === 'ESHAPE');
  await assert.rejects(reg.save(m, [0, 0, 0]), /all zeros/);
  await assert.rejects(reg.save(m, [1, NaN, 0]), /finite/);
  await assert.rejects(reg.save({ schema: 'x' }, [1, 0, 0]), /schema/);
  await assert.rejects(reg.save(m, [1, 0, 0], { parent: 'ws-missing' }), (e) => e.code === 'ENOTFOUND');
  await assert.rejects(reg.nearest([1, 0, 0], 0), /k must be/);
  assert.throws(() => reg.lineage('ws-missing'), (e) => e.code === 'ENOTFOUND');
  await assert.rejects(reg.recordOutcome('ws-missing', 1), (e) => e.code === 'ENOTFOUND');
  await assert.rejects(WeightRegistry.open({ dimensions: 0 }), /dimensions/);
  const wrong = new FlatVectorStore(5);
  const reg2 = await WeightRegistry.open({ store: wrong });
  await assert.rejects(reg2.save(m, [1, 0, 0]), (e) => e.code === 'ESHAPE');
});

test('event log persists and replays; tampering is detected', async () => {
  const dir = tmp();
  try {
    const path = join(dir, 'reg', 'weights.jsonl');
    const reg = await WeightRegistry.open({ path, store: 'flat' });
    assert.equal(reg.dimensions, null, 'dimensions inferred from first save');
    const a = await reg.save(createMlp([2, 2, 1], { seed: 4 }), [1, 2, 3], { task: 't' });
    const b = await reg.perturb(a.id, { scale: 0.5 });
    await reg.recordOutcome(b.id, 0.75);
    await reg.close();

    const again = await WeightRegistry.open({ path, store: 'flat' });
    assert.equal(again.size, 2);
    assert.equal(again.dimensions, 3);
    assert.deepEqual(again.lineage(b.id).map((r) => r.id), [a.id, b.id]);
    assert.deepEqual(again.get(b.id).outcomes.map((o) => o.score), [0.75]);
    assert.equal((await again.nearest([1, 2, 3], 1))[0].id, b.id, 'newest version wins the tie');
    await again.close();

    // A torn trailing line (crash mid-append) is tolerated.
    writeFileSync(path, readFileSync(path, 'utf8') + '{"v":1,"type":"snap');
    assert.equal((await WeightRegistry.open({ path, store: 'flat' })).size, 2);

    const lines = readFileSync(path, 'utf8').split('\n');
    const first = JSON.parse(lines[0]);
    first.record.model.layers[0].weights[0] += 1;
    lines[0] = JSON.stringify(first);
    writeFileSync(path, lines.join('\n'));
    await assert.rejects(WeightRegistry.open({ path, store: 'flat' }), (e) => e.code === 'EINTEGRITY');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('ruvector-core index persists next to the log', async () => {
  const dir = tmp();
  try {
    const path = join(dir, 'w.jsonl');
    const reg = await WeightRegistry.open({ path, store: 'ruvector-core' });
    const a = await reg.save(createMlp([1, 1]), [0, 1], {});
    await reg.close();
    const again = await WeightRegistry.open({ path, store: 'ruvector-core' });
    assert.equal((await again.nearest([0, 1], 1))[0].id, a.id);
    assert.ok(readFileSync(path + '.index.db').length > 0);
    await again.close();
    // The persisted index is reused when it already holds the log's ids; a fresh one is rebuilt.
    const kept = await RuvectorVectorStore.open(2, { storagePath: path + '.index.db' });
    assert.deepEqual(await kept.sync([{ id: a.id, vector: [0, 1] }]), { reindexed: 0 });
    assert.equal(kept.size(), 1);
    await kept.close();
    const fresh = await RuvectorVectorStore.open(2, { storagePath: join(dir, 'other.db') });
    assert.deepEqual(await fresh.sync([{ id: a.id, vector: [0, 1] }]), { reindexed: 1 });
    assert.equal((await fresh.search([0, 1], 1))[0].id, a.id);
    await fresh.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('setLayer on the registry rejects bad shapes without writing', async () => {
  const reg = await WeightRegistry.open({ dimensions: 2, store: 'flat' });
  const a = await reg.save(createMlp([2, 2]), [1, 1]);
  await assert.rejects(reg.setLayer(a.id, 0, { weights: [1] }), (e) => e.code === 'ESHAPE');
  assert.equal(reg.size, 1);
  assert.throws(() => setLayer(a.model, 3, {}), /layer index/);
});
