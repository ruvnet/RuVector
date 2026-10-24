/** Run with RUVECTOR_NATIVE_TEST_PREFIX pointing to an npm project containing ruvector. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, copyFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const prefix = process.env.RUVECTOR_NATIVE_TEST_PREFIX;
assert.ok(prefix, 'Set RUVECTOR_NATIVE_TEST_PREFIX to a project with the real native RuVector packages installed.');
test('native adapter preserves batch vectors and metadata across reopening', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'agentic-native-'));
  try {
    await copyFile(new URL('../src/adapters/ruvector.js', import.meta.url), join(dir, 'adapter.mjs'));
    await symlink(join(prefix, 'node_modules'), join(dir, 'node_modules'));
    const { RuvectorAdapter } = await import(pathToFileURL(join(dir, 'adapter.mjs')));
    const options = { dimensions: 3, inMemory: false, path: join(dir, 'vectors.db') };
    const first = new RuvectorAdapter(options);
    await first.initialize();
    assert.equal(first.isNative(), true, 'a memory fallback cannot satisfy this test');
    await first.insert([{ id: 'ticket:a', vector: [1, 0, 0], metadata: { department: 'support' } }]);
    await first.insertBatch([{ id: 'ticket:b', vector: [0, 1, 0] }, { id: 'ticket:c', vector: [0, 0, 1] }]);
    assert.equal((await first.stats()).count, 3);
    assert.equal((await first.search([1, 0, 0], 1))[0].id, 'ticket:a');
    await assert.rejects(first.insertBatch([{ id: 'bad', vector: [1, 2] }]), /dimension mismatch/);
    const reopened = new RuvectorAdapter(options);
    await reopened.initialize();
    assert.equal(reopened.isNative(), true);
    assert.equal((await reopened.stats()).count, 3);
    assert.deepEqual((await reopened.get('ticket:a')).metadata, { department: 'support' });
    assert.equal(await reopened.delete('ticket:b'), true);
    assert.equal((await reopened.stats()).count, 2);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
