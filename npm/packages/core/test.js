const { VectorDB, VectorDb, version, nativeVersion } = require('./index.js');

if (VectorDB !== VectorDb) {
  throw new Error('VectorDB compatibility alias does not match VectorDb');
}

const expectedVersion = require('./package.json').version;
if (version() !== expectedVersion) {
  throw new Error(`version() returned ${version()}, expected ${expectedVersion}`);
}
if (typeof nativeVersion() !== 'string') {
  throw new Error('nativeVersion() must return a string');
}

async function test() {
  console.log('Testing native module...');

  try {
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
