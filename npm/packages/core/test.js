const { VectorDB, VectorDb } = require('./index.js');

if (VectorDB !== VectorDb) {
  throw new Error('VectorDB compatibility alias does not match VectorDb');
}

async function test() {
  console.log('Testing native module...');

  try {
    const configured = new VectorDB({
      dimensions: 4,
      distanceMetric: 'Euclidean',
      storagePath: `memory://get-options-${process.pid}`
    });
    const effective = configured.getOptions();
    if (
      effective.dimensions !== 4 ||
      effective.distanceMetric !== 'euclidean' ||
      effective.indexType !== 'flat' ||
      effective.hnswConfig != null
    ) {
      throw new Error(`Unexpected effective options: ${JSON.stringify(effective)}`);
    }
    console.log('✓ Effective configuration getter');

    // Create database
    const db = VectorDB.withDimensions(128);
    console.log('✓ Created database');

    // Insert vector
    const id = await db.insert({
      vector: new Float32Array(128).fill(0.5)
    });
    console.log('✓ Inserted vector:', id);

    // Search
    const results = await db.search({
      vector: new Float32Array(128).fill(0.5),
      k: 1
    });
    console.log('✓ Search results:', results);

    // Check length
    const len = await db.len();
    console.log('✓ Database length:', len);

    console.log('\n✅ All tests passed!');
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

test();
