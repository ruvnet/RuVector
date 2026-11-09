const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { DiskAnn } = require('./index.js');

console.log('Testing @ruvector/diskann...');

const DIM = 32;
const vecFor = (i) => {
  const vec = new Float32Array(DIM);
  for (let d = 0; d < DIM; d++) vec[d] = Math.sin(i * 0.1 + d * 0.3);
  return vec;
};

try {
  const index = new DiskAnn({ dim: DIM, maxDegree: 16, buildBeam: 32, searchBeam: 32, alpha: 1.2 });
  const n = 200;
  for (let i = 0; i < n; i++) index.insert(`vec-${i}`, vecFor(i));
  assert.strictEqual(index.count(), n);
  console.log(`✓ Inserted ${n} vectors`);

  index.build();
  const results = index.search(vecFor(42), 5);
  assert.strictEqual(results.length, 5);
  assert.strictEqual(results[0].id, 'vec-42');
  console.log(`✓ search() top result ${results[0].id} (distance ${results[0].distance.toFixed(6)})`);

  assert.throws(() => index.search(new Float32Array(DIM - 1), 5), /Dimension mismatch/);
  console.log('✓ dimension mismatch throws');

  assert.strictEqual(index.delete('vec-42'), true);
  assert.strictEqual(index.delete('vec-42'), false);
  assert.ok(!index.search(vecFor(42), 5).some((r) => r.id === 'vec-42'));
  console.log('✓ delete() removes the id from results');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'diskann-test-'));
  try {
    index.save(dir);
    const loaded = DiskAnn.load(dir);
    assert.strictEqual(loaded.count(), n - 1);
    assert.strictEqual(loaded.search(vecFor(43), 1)[0].id, 'vec-43');
    console.log('✓ save()/load() round trip');

    // A corrupt index file must throw a JS error, not abort the process.
    fs.writeFileSync(path.join(dir, 'graph.bin'), Buffer.alloc(3));
    assert.throws(() => DiskAnn.load(dir), /graph\.bin/);
    console.log('✓ corrupt graph.bin throws instead of crashing');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }

  console.log('\nAll tests passed!');
} catch (e) {
  console.error('✗ Test failed:', e.message);
  process.exitCode = 1;
}
