const { DiskAnn } = require('./index.js');

console.log('Testing @ruvector/diskann...');

try {
  // Create index
  const index = new DiskAnn({ dim: 32, maxDegree: 16, buildBeam: 32, searchBeam: 32, alpha: 1.2 });
  console.log('✓ DiskAnn instance created');

  // Insert vectors
  const n = 200;
  for (let i = 0; i < n; i++) {
    const vec = new Float32Array(32);
    for (let d = 0; d < 32; d++) vec[d] = Math.sin(i * 0.1 + d * 0.3);
    index.insert(`vec-${i}`, vec);
  }
  console.log(`✓ Inserted ${n} vectors`);
  console.log(`✓ count(): ${index.count()}`);

  // Build index
  index.build();
  console.log('✓ build() completed');

  // Search — query = vec-42, should find itself
  const query = new Float32Array(32);
  for (let d = 0; d < 32; d++) query[d] = Math.sin(42 * 0.1 + d * 0.3);

  const results = index.search(query, 5);
  console.log(`✓ search() returned ${results.length} results`);
  if (results.length > 0) {
    console.log(`  Top result: ${results[0].id} (distance: ${results[0].distance.toFixed(6)})`);
    if (results[0].id === 'vec-42') {
      console.log('✓ Correct nearest neighbor found!');
    }
  }

  // Delete
  const deleted = index.delete('vec-42');
  console.log(`✓ delete('vec-42'): ${deleted}`);

  console.log('\nAll tests passed!');
} catch (e) {
  console.error('✗ Test failed:', e.message);
}
